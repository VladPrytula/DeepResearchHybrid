# research/actions.py
import asyncio
import logging
from typing import Any, Dict, List, Tuple

import numpy as np

from agent_config import Settings
from agent_helpers import (cosine_similarity, fetch_clean, hash_txt,
                           searx_search, sentence_chunks)

# Forward declarations for type hinting
class ResearchState:
    pass

class AnalysisComponent:
    pass

class ActionComponent:
    """
    Handles the execution of actions defined in the research plan.

    This component is responsible for performing web searches, fetching and cleaning
    web page content, scoring the relevance of new information chunks using HyDE,
    and adding the most valuable chunks to the research state's knowledge base.
    """
    def __init__(self, state: 'ResearchState', analysis: 'AnalysisComponent', logger: logging.Logger):
        self.state = state
        self.analysis = analysis
        self.logger = logger

    async def act(self, search_actions: List[Dict[str, Any]]) -> Tuple[Dict[str, List[str]], int]:
        """
        Executes a list of search actions, fetches content, scores it, and updates the state.

        Args:
            search_actions: A list of search action dictionaries from the plan.

        Returns:
            A tuple containing a dictionary mapping queries to new source titles,
            and the total number of new chunks added to the knowledge base.
        """
        self.logger.info(f"--- Agent Step: Acting on Plan (with HyDE) ---")
        if not search_actions:
            self.logger.info("Plan contains no search actions. Skipping action phase.")
            return {}, 0
        
        query_to_sources_map = {action['query']: [] for action in search_actions}
        
        target_topics = list(set(action['target_outline_topic'] for action in search_actions if action.get('target_outline_topic')))
        
        hyde_generation_tasks = [self.analysis._generate_hypothetical_document(topic) for topic in target_topics]
        hypothetical_docs = await asyncio.gather(*hyde_generation_tasks)
        
        hyde_embeddings_list = await self.analysis._embed_texts_with_cache(hypothetical_docs)
        
        hyde_embedding_map = {topic: emb for topic, emb in zip(target_topics, hyde_embeddings_list) if emb}
        self.logger.info(f"Generated and embedded {len(hyde_embedding_map)} hypothetical documents for relevance scoring.")

        scored_chunks = []
        for action in search_actions:
            query = action.get('query')
            if not query: continue

            target_topic = action.get('target_outline_topic') 
            
            utility_embedding = None
            if target_topic and target_topic in hyde_embedding_map:
                utility_embedding = hyde_embedding_map[target_topic]
            elif self.state.query_embedding: 
                utility_embedding = self.state.query_embedding
            
            if not utility_embedding:
                self.logger.warning(f"No utility embedding available for query '{query}' (target: '{target_topic}'). Skipping scoring for this action's results.")
                continue

            self.logger.info(f"Executing search for query: '{query}' (Target: '{target_topic or 'Overall Query'}')")
            hits = await searx_search(query)
            
            urls_to_fetch = { hit['url'] for hit in hits if hit.get('url') and hit['url'] not in self.state.url_to_source_index }
            if not urls_to_fetch:
                self.logger.info(f"All search results for query '{query}' have already been processed. Skipping.")
                continue
            
            fetch_tasks = {url: asyncio.create_task(fetch_clean(url)) for url in urls_to_fetch}
            chunks_to_embed, chunk_metadata = [], []

            for url, content_task in fetch_tasks.items():
                content = await content_task
                if content and len(content) > 100:
                    source_idx = len(self.state.results) 
                    title = next((h['title'] for h in hits if h.get('url') == url), "Untitled")
                    self.state.results.append({"url": url, "title": title, "query": query})
                    self.state.url_to_source_index[url] = source_idx
                    if query in query_to_sources_map: query_to_sources_map[query].append(title)
                    for chunk_text in sentence_chunks(content):
                        chunks_to_embed.append(chunk_text)
                        chunk_metadata.append({'original_chunk': chunk_text, 'source_idx': source_idx})
            
            if not chunks_to_embed: continue

            chunk_embeddings = await self.analysis._embed_texts_with_cache(chunks_to_embed)
            existing_embs_list = list(self.state.chunk_embedding_cache.values()) 

            for i, chunk_emb_list in enumerate(chunk_embeddings):
                if not chunk_emb_list: continue
                chunk_emb_np = np.array(chunk_emb_list)
                meta = chunk_metadata[i]
                chunk_text = meta['original_chunk']
                
                utility = cosine_similarity(np.array(utility_embedding), chunk_emb_np)
                redundancy = 0.0
                if existing_embs_list: 
                    similarities_to_existing = [cosine_similarity(chunk_emb_np, np.array(e_emb)) for e_emb in existing_embs_list if e_emb is not None] 
                    if similarities_to_existing: 
                         redundancy = max(similarities_to_existing)
                
                score = (Settings.NOVELTY_ALPHA * utility) - ((1 - Settings.NOVELTY_ALPHA) * redundancy)
                scored_chunks.append((score, chunk_text, meta['source_idx'], chunk_emb_list))

        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        top_chunks = scored_chunks[:Settings.NOVELTY_TOP_K]

        num_new_chunks_added = 0
        for score, chunk_text, source_idx, chunk_emb_list in top_chunks:
            chunk_hash = hash_txt(chunk_text)
            if chunk_hash not in self.state.chunk_embedding_cache: 
                self.state.all_chunks.append((chunk_text, source_idx))
                self.state.chunk_embedding_cache[chunk_hash] = chunk_emb_list
                num_new_chunks_added +=1
        
        self.logger.info(f"Added {num_new_chunks_added} new chunks to knowledge base (out of {len(scored_chunks)} candidates).")
        await asyncio.sleep(0.5)
        return query_to_sources_map, num_new_chunks_added