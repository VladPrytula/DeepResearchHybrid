# research/synthesis.py
import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Set, Tuple

import numpy as np

from agent_config import PROMPTS, Settings
# WHAT: Added `extract_json_from_response` to the list of imported helper functions.
# WHY: This function is called within the `_reflexion_pass` method to parse JSON from the LLM's review response. It was missing from the import list, causing the `NameError` you observed.
from agent_helpers import (a_chat, a_embed, extract_json_from_response,
                           fetch_clean, hash_txt, searx_search,
                           sentence_chunks)

# Forward declarations for type hinting
class ResearchState:
    pass

class AnalysisComponent:
    pass


class SynthesisComponent:
    """
    Handles the final synthesis of the research report.

    This component takes the completed research state, including the outline and
    all collected information, and generates a structured, well-written report.
    It includes a self-correction (reflexion) mechanism to improve section quality
    and automatically generates a bibliography.
    """
    def __init__(self, state: 'ResearchState', analysis: 'AnalysisComponent', logger: logging.Logger):
        self.state = state
        self.analysis = analysis
        self.logger = logger
        self.source_lock = asyncio.Lock()

    async def synthesise(self) -> str:
        """Top-level method to generate the full research report."""
        if not self.state.outline or not any(item.get('topic') for item in self.state.outline):
            self.logger.error("Cannot synthesize report: Outline is empty or invalid.")
            return "# Report Generation Failed\n\nThe research outline could not be generated. Please try a different query or check logs."

        section_tasks = [self._synthesise_section_with_citations(block) for block in self.state.outline if block.get('topic')]
        section_texts = await asyncio.gather(*section_tasks)
        
        section_md_parts = []
        valid_outline_blocks = [block for block in self.state.outline if block.get('topic')]
        for i, text in enumerate(section_texts):
            if i < len(valid_outline_blocks):
                 section_md_parts.append(f"## {valid_outline_blocks[i]['topic']}\n\n{text}")
        section_md = "\n\n".join(section_md_parts)

        title_task = a_chat([{"role": "system", "content": "Create a concise, formal research report title for a report on the following topic. The title should be engaging and accurately reflect the core subject."}, {"role": "user", "content": self.state.query}], max_tokens=64, temp=0.3)
        abstract_task = a_chat([{"role": "system", "content": "Write a 200-250 word academic abstract for the report. Summarize the key findings and conclusions based on the provided section content."}, {"role": "user", "content": section_md[:Settings.MAX_ABSTRACT_CONTEXT_CHARS]}], max_tokens=400, temp=0.3)
        
        title, abstract = await asyncio.gather(title_task, abstract_task)
        
        bibliography = self._make_bibliography(section_md)
        return f"# {title}\n\n## Abstract\n\n{abstract}\n\n{section_md}\n\n{bibliography}"

    def _clean_section_text(self, text: str) -> str:
        """Removes hallucinated 'References' or 'Bibliography' sections from LLM output."""
        cleaned_text = re.sub(
            r'\n\s*((?:\*\*|##)?\s*(?:References|Bibliography|Works Cited)\s*(?:\*\*|##)?:?)[\s\S]*',
            '',
            text,
            flags=re.IGNORECASE
        )
        if len(cleaned_text) < len(text):
            self.logger.info("Cleaned a hallucinated bibliography from a generated section.")
        return cleaned_text.strip()

    async def _synthesise_section_with_citations(self, block: Dict[str, Any]) -> str:
        """Synthesizes a single section of the report based on a topic block from the outline."""
        topic_str = block.get('topic')
        if not topic_str:
            self.logger.warning("Skipping synthesis for block with no topic.")
            return ""

        self.logger.info(f"Synthesizing section: '{topic_str}'")
        
        subtopics_str = ", ".join(block.get('subtopics', []))
        section_focus_query = f"{topic_str}"
        if subtopics_str:
            section_focus_query += f": {subtopics_str}"

        if not self.state.all_chunks:
            return f"No information found in the knowledge base for the topic: {topic_str}."

        hyde_section_query_doc = await self.analysis._generate_hypothetical_document(section_focus_query)
        query_emb_list = await a_embed(hyde_section_query_doc)

        if not query_emb_list:
            self.logger.warning(f"Could not embed query for section '{topic_str}'. Skipping synthesis.")
            return f"Could not process query for section: {topic_str}."
        query_emb_np = np.array(query_emb_list)

        chunk_data_for_section = []
        for chunk_text, source_idx in self.state.all_chunks:
            chunk_hash = hash_txt(chunk_text)
            chunk_emb = self.state.chunk_embedding_cache.get(chunk_hash)
            if chunk_emb:
                chunk_data_for_section.append({'text': chunk_text, 'emb': np.array(chunk_emb), 'source_idx': source_idx})
        
        if not chunk_data_for_section:
            return f"No embedded chunks available for synthesizing section: {topic_str}."

        for chunk_item in chunk_data_for_section:
            chunk_item['similarity'] = (np.dot(query_emb_np, chunk_item['emb']) / 
                                       (np.linalg.norm(query_emb_np) * np.linalg.norm(chunk_item['emb'])))
        
        chunk_data_for_section.sort(key=lambda x: x['similarity'], reverse=True)
        top_k_chunks_data = chunk_data_for_section[:Settings.TOP_K_RESULTS_PER_SECTION]

        if not top_k_chunks_data:
            return f"No relevant information found for section: {topic_str} after similarity ranking."

        context_for_llm = ""
        initial_source_indices = set()
        for chunk_item in top_k_chunks_data:
            source_id = chunk_item['source_idx']
            initial_source_indices.add(source_id)
            context_for_llm += f"[Source {source_id + 1}]: {chunk_item['text']}\n\n"
        
        if not context_for_llm.strip():
             return f"No relevant context constructed for section: {topic_str}."

        prompt = [{"role": "system", "content": PROMPTS.SECTION_SYNTHESIZER}, 
                  {"role": "user", "content": f"Topic: {topic_str}\nSubtopics to consider: {subtopics_str}\n\nExcerpts:\n{context_for_llm}"}]
        
        raw_section = await a_chat(prompt, max_tokens=1500, temp=0.4) 
        if raw_section.startswith("Error:"):
            self.logger.error(f"LLM failed to synthesize section '{topic_str}': {raw_section}")
            return f"Failed to synthesize section: {topic_str}. LLM Error."

        cleaned_section = self._clean_section_text(raw_section)
        final_section = await self._reflexion_pass(block, cleaned_section, context_for_llm, initial_source_indices)
        final_section = re.sub(r'\[[Ss]ource\s*(\d+)\]', r'[\1]', final_section)
        return final_section

    async def _search_and_fetch_for_reflexion(self, query: str, existing_source_urls: set, topic: str) -> Tuple[str, Set[str]]:
        """Performs a targeted web search during the reflexion pass to fill knowledge gaps."""
        self.logger.info(f"Reflexion: Searching for '{query}' to enhance topic '{topic}'")
        hits = await searx_search(query, limit=2) 
        
        newly_added_urls = set()
        urls_to_fetch = []
        
        for hit in hits:
            url = hit.get('url')
            if url and url not in self.state.url_to_source_index and url not in existing_source_urls:
                 urls_to_fetch.append(url)
        
        if not urls_to_fetch:
            self.logger.info("Reflexion search: No new, unique URLs found.")
            return "", set()
            
        fetched_contents_tasks = [fetch_clean(url) for url in urls_to_fetch]
        fetched_contents = await asyncio.gather(*fetched_contents_tasks)
        
        new_context_str = ""
        
        relevant_hits_info = [h for h in hits if h.get('url') in urls_to_fetch]

        for url, content, hit_info in zip(urls_to_fetch, fetched_contents, relevant_hits_info): 
            if content and len(content) > 100: 
                source_idx = -1
                async with self.source_lock: 
                    if url not in self.state.url_to_source_index: 
                        source_idx = len(self.state.results)
                        self.state.results.append({"url": url, "title": hit_info.get('title', "Untitled Reflexion Source"), "query": f"reflexion: {query}"})
                        self.state.url_to_source_index[url] = source_idx
                        
                        new_source_chunks = sentence_chunks(content)
                        new_source_chunk_embeddings = await self.analysis._embed_texts_with_cache(new_source_chunks)
                        
                        for chunk_text, chunk_emb in zip(new_source_chunks, new_source_chunk_embeddings):
                            if chunk_emb:
                                chunk_hash = hash_txt(chunk_text)
                                self.state.all_chunks.append((chunk_text, source_idx))
                                self.state.chunk_embedding_cache[chunk_hash] = chunk_emb
                        
                        self.logger.info(f"Reflexion: Added {len(new_source_chunks)} chunks from new source: {url}")

                if source_idx != -1: 
                    new_context_str += f"\n[Source {source_idx + 1}]: {content[:1000]}...\n" 
                    newly_added_urls.add(url)
            else:
                self.logger.info(f"Reflexion: No useful content fetched from {url}")

        return new_context_str, newly_added_urls

    async def _reflexion_pass(self, block: Dict[str, Any], initial_text: str, context: str, initial_source_indices: set) -> str:
        """Performs a self-correction loop on a synthesized section of text."""
        current_text, current_context = initial_text, context
        all_source_urls_for_section = {self.state.results[i]['url'] for i in initial_source_indices if i < len(self.state.results)}
        
        topic_str = block.get('topic', "Current Section")

        for i in range(Settings.MAX_REFLEXION_LOOPS):
            self.logger.info(f"Reflexion Pass {i+1}/{Settings.MAX_REFLEXION_LOOPS} for section '{topic_str}'")
            review_prompt = [{"role": "system", "content": PROMPTS.REFLEXION_REVIEWER}, 
                             {"role": "user", "content": f"Topic: {topic_str}\n\nText to Review:\n{current_text}"}]
            
            raw_review = await a_chat(review_prompt, temp=0.4, max_tokens=512)
            if raw_review.startswith("Error:"):
                self.logger.error(f"Reflexion reviewer LLM failed: {raw_review}. Aborting reflexion for this section.")
                return current_text 

            json_to_parse = extract_json_from_response(raw_review)
            action = "REWRITE"
            critique = "No specific critique provided (or JSON parsing failed)."

            if not json_to_parse:
                self.logger.warning(f"Could not extract JSON from reflexion review response. Raw: {raw_review}. Defaulting to REWRITE.")
            else:
                self.logger.debug(f"Attempting to parse JSON string for reflexion review: >>>{json_to_parse}<<<")
                try:
                    review_json = json.loads(json_to_parse)
                    action = review_json.get("action", "REWRITE").upper()
                    critique = review_json.get("critique", "No specific critique provided.")
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse reflexion review: {e}. Extracted JSON: >>>{json_to_parse}<<< Raw: {raw_review}. Defaulting to REWRITE.")
                    critique = f"Draft needed revision (parsing error: {e})."
            
            if action == "NONE":
                self.logger.info(f"Reflexion for '{topic_str}': No issues found. Finalizing section.")
                return current_text

            self.logger.warning(f"Reflexion for '{topic_str}' (Action: {action}): {critique}")
            
            reflexion_query = None
            if action == "SEARCH":
                if json_to_parse:
                    try:
                        temp_review_json = json.loads(json_to_parse)
                        reflexion_query = temp_review_json.get("query")
                    except json.JSONDecodeError:
                        self.logger.warning("Could not re-parse review JSON for SEARCH query, reflexion_query remains None.")

                if reflexion_query:
                    new_evidence_context, newly_fetched_urls = await self._search_and_fetch_for_reflexion(reflexion_query, all_source_urls_for_section, topic_str)
                    if new_evidence_context:
                        current_context += "\n--- NEW EVIDENCE (from Reflexion Search) ---\n" + new_evidence_context
                        all_source_urls_for_section.update(newly_fetched_urls)
                        critique += "\n(Note: New evidence has been found and added to the context for revision.)"
                    else:
                        critique += "\n(Note: Reflexion search for more info was attempted but yielded no new usable content.)"
                else:
                    critique += "\n(Note: Reviewer suggested SEARCH but no query was provided or JSON was unparsable for query. Proceeding with REWRITE based on existing context.)"
            
            resynthesis_prompt = [{"role": "system", "content": PROMPTS.REFLEXION_REWRITER},
                                  {"role": "user", "content": f"Topic: {topic_str}\n\nFull Context (Original + New Evidence if any):\n{current_context}\n\nFlawed Draft:\n{current_text}\n\nReviewer's Feedback:\n{critique}\n\nRevised Section:"}]
            
            previous_text = current_text
            current_text = await a_chat(resynthesis_prompt, max_tokens=1500, temp=0.4) 
            if current_text.startswith("Error:"):
                self.logger.error(f"Reflexion rewriter LLM failed: {current_text}. Returning text from before this failed rewrite.")
                return previous_text
        
        self.logger.info(f"Finished reflexion for '{topic_str}' after {Settings.MAX_REFLEXION_LOOPS} loops.")
        return current_text

    def _make_bibliography(self, full_text: str) -> str:
        """Constructs a bibliography from all cited sources in the final text."""
        cited_indices_str = set(re.findall(r'\[(\d+)\]', full_text))
        if not cited_indices_str: return "## Bibliography\n\nNo sources were cited in this report."
        
        cited_indices = sorted([int(i) for i in cited_indices_str])
        
        entries = []
        for i in cited_indices:
            source_index_in_list = i - 1 
            if 0 <= source_index_in_list < len(self.state.results):
                res = self.state.results[source_index_in_list]
                title = res.get('title', "Untitled Source")
                url = res.get('url', "#") 
                entries.append(f"[{i}] {title}. <{url}>")
            else:
                self.logger.warning(f"Bibliography: Cited source index [{i}] is out of bounds for available results ({len(self.state.results)}).")
                entries.append(f"[{i}] Reference information not available (index out of bounds).")

        if not entries: return "## Bibliography\n\nNo valid cited sources found."
        return "## Bibliography\n\n" + "\n\n".join(entries)