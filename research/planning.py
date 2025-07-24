# research/planning.py
import asyncio
import json
import logging
import re
from typing import Any, Dict, List

import numpy as np

from agent_config import PROMPTS, Settings
from agent_helpers import a_chat, cosine_similarity, extract_json_from_response

# Forward declarations for type hinting
class ResearchState:
    pass

class AnalysisComponent:
    pass


class PlanningComponent:
    """
    Handles the strategic planning and critique for the research agent.

    This component is responsible for drafting the initial research outline,
    generating search queries, and iteratively refining the research plan based
    on the current state, coverage analysis, and discovered latent topics.
    """
    def __init__(self, state: 'ResearchState', analysis: 'AnalysisComponent', logger: logging.Logger):
        self.state = state
        self.analysis = analysis
        self.logger = logger

    async def generate_queries(self, topic: str, count: int, purpose: str) -> List[str]:
        """Generates a specified number of search queries for a given topic and purpose."""
        prompt = [{"role": "system", "content": f"You are a research expert. Generate {count} {purpose}. Return a JSON list of strings."},
                  {"role": "user", "content": topic}]
        raw = await a_chat(prompt, max_tokens=384)
        try: 
            match = re.search(r"\[.*?\]", raw, re.DOTALL)
            if match:
                json_str = match.group(0)
                queries = json.loads(json_str)
                return [str(q) for q in queries if isinstance(q, str)][:count]
            else: 
                raise json.JSONDecodeError("No JSON list found", raw, 0)
        except json.JSONDecodeError: 
            self.logger.warning(f"Could not parse queries as JSON: {raw}. Falling back to line splitting.")
            return [l.strip("-•* ") for l in raw.splitlines() if l.strip()][:count]

    async def draft_outline(self) -> List[Dict[str, Any]]:
        """Drafts the initial research outline based on the first set of collected chunks."""
        if not self.state.all_chunks:
            self.logger.warning("No chunks available to draft outline. Using default query as single topic.")
            return [{"topic": self.state.query, "subtopics": []}]
        
        ctx = "\n\n".join(c[0] for c in self.state.all_chunks[:50]) 
        prompt = [{"role": "system", "content": PROMPTS.OUTLINE_DRAFTER},
                  {"role": "user", "content": f"User's Question: {self.state.query}\n\nContext:\n{ctx[:Settings.MAX_ABSTRACT_CONTEXT_CHARS]}"}] 
        
        raw_response = await a_chat(prompt, max_tokens=1536) 
        
        json_to_parse = extract_json_from_response(raw_response)
        if not json_to_parse:
            self.logger.error(f"Could not extract a JSON string for outline. Raw: {raw_response}. Using default.")
            return [{"topic": self.state.query, "subtopics": []}]

        self.logger.debug(f"Attempting to parse JSON string for outline: >>>{json_to_parse}<<<")
        try:
            parsed_json = json.loads(json_to_parse)
            raw_outline_data = parsed_json.get("outline", [])
            
            sanitized_outline = []
            if isinstance(raw_outline_data, list):
                for item in raw_outline_data:
                    topic_name = None
                    if isinstance(item, dict):
                        topic_name = item.get('topic') or item.get('title') 
                    
                    if topic_name and isinstance(topic_name, str) and topic_name.strip():
                        subtopics_list = item.get('subtopics', [])
                        valid_subtopics = [st for st in subtopics_list if isinstance(st, str) and st.strip()]
                        sanitized_outline.append({'topic': topic_name.strip(), 'subtopics': valid_subtopics})
            
            if not sanitized_outline:
                self.logger.warning(f"Sanitized outline is empty after processing. Raw outline data: {raw_outline_data}")
                raise ValueError("Sanitized outline is empty or invalid structure.")
            
            self.logger.info(f"Successfully drafted and sanitized outline with {len(sanitized_outline)} topics.")
            return sanitized_outline
            
        except (json.JSONDecodeError, AttributeError, ValueError) as e:
            self.logger.error(f"Failed to draft outline: {e}. Extracted JSON string attempt: >>>{json_to_parse}<<< . Raw response: {raw_response}. Using default.")
            return [{"topic": self.state.query, "subtopics": []}]

    async def plan_and_critique(self):
        """The core planning step that uses the current state to generate the next plan."""
        self.logger.info("--- Agent Step: Planning & Critiquing ---")
        _, coverage_summary = await self.analysis.calculate_topic_coverage()
        previous_queries = list(set(res['query'] for res in self.state.results if 'reflexion' not in res.get('query', '')))
        gain_trend = self.analysis.get_gain_trend_description()
        
        latent_topics_summary = "Not run."
        if Settings.ENABLE_EXPLORATION:
            self.logger.info("Exploration enabled. Discovering latent topics...")
            latent_topics = await self.analysis.get_latent_topics()
            if latent_topics:
                latent_topic_labels = [lt['label'] for lt in latent_topics]
                outline_topic_texts = [t.get('topic') for t in self.state.outline if isinstance(t, dict) and t.get('topic')]
                
                if outline_topic_texts:
                    hyde_docs_for_outline = await asyncio.gather(*(self.analysis._generate_hypothetical_document(ot) for ot in outline_topic_texts))
                    latent_topic_embs_list = await self.analysis._embed_texts_with_cache(latent_topic_labels)
                    outline_topic_embs_list = await self.analysis._embed_texts_with_cache(hyde_docs_for_outline)
                    
                    valid_latent_embs = [np.array(emb) for emb in latent_topic_embs_list if emb]
                    valid_outline_embs = [np.array(emb) for emb in outline_topic_embs_list if emb]

                    novel_topics = []
                    if valid_latent_embs and valid_outline_embs:
                        for i, lt_emb in enumerate(valid_latent_embs):
                            max_sim = max([cosine_similarity(lt_emb, ot_emb) for ot_emb in valid_outline_embs])
                            if max_sim < 0.65:
                                novel_topics.append(latent_topic_labels[i])
                    
                    if novel_topics:
                        latent_topics_summary = f"Found {len(novel_topics)} potentially new topics: {', '.join(novel_topics)}"
                        self.logger.info(f"Discovered novel latent topics: {novel_topics}")
                    else:
                        latent_topics_summary = "No new latent topics discovered; existing topics seem to cover the data well."
                else:
                    latent_topics_summary = f"Found {len(latent_topic_labels)} initial topics: {', '.join(latent_topic_labels)}"
            else:
                latent_topics_summary = "Not enough data to discover latent topics."

        state_summary = f"""
        Original Query: {self.state.query}
        Report Outline: {json.dumps(self.state.outline, indent=2)}
        Current Topic Coverage: {coverage_summary}
        Discovered Latent Topics: {latent_topics_summary}
        Research Cycles Completed: {self.state.cycles}
        Previous Critique: {self.state.critique_history[-1] if self.state.critique_history else 'None'}
        Information Gain Trend: {gain_trend}
        Previously Executed Search Queries (last 5): {json.dumps(previous_queries[-5:], indent=2)}
        """
        prompt = [{"role": "system", "content": PROMPTS.PLANNER_CRITIC}, {"role": "user", "content": state_summary}]
        raw_response = await a_chat(prompt, temp=0.2, max_tokens=1024)
        
        json_to_parse = extract_json_from_response(raw_response)
        if not json_to_parse:
            self.logger.error(f"Could not extract a JSON string from Planner/Critic response. Raw: {raw_response}")
            self.state.plan = {"critique": "Error: No JSON found in LLM response.", "thought": "Failed to generate a valid plan.", "plan": []}
            return

        self.logger.debug(f"Attempting to parse JSON string from planner: >>>{json_to_parse}<<<")
        try:
            response_json = json.loads(json_to_parse)
            self.state.plan = response_json
            
            if "plan" not in response_json or not isinstance(response_json["plan"], list):
                 self.logger.warning("Planner response missing 'plan' key or it's not a list. Assuming research complete.")
                 self.state.plan["plan"] = []

            critique = response_json.get("critique", "N/A (critique not provided by planner)")
            thought = response_json.get("thought", "N/A (thought not provided by planner)")
            
            self.logger.info(f"Agent Critique: {critique}")
            self.logger.info(f"Agent Thought: {thought}")
            self.logger.info(f"New Plan: {json.dumps(self.state.plan.get('plan'), indent=2)}")
            self.state.critique_history.append(critique)

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse Planner/Critic response: {e}. Extracted JSON string attempt: >>>{json_to_parse}<<< . Original Raw: {raw_response}")
            self.state.plan = {"critique": f"Error: {e} during JSON parsing.", "thought": "Failed to generate a valid plan due to parsing error.", "plan": []}

    async def generate_agent_summary(self) -> str:
        """Generates a human-readable summary of the agent's current plan."""
        if not self.state.plan: return "No plan available."
        thought = self.state.plan.get("thought", "N/A")
        plan_actions = self.state.plan.get("plan", [])
        if not plan_actions: return "Research is complete. Preparing to write the final report."
        
        action_summaries = []
        for action in plan_actions:
            action_type = action.get("action", "SEARCH").upper()
            if action_type == "SEARCH":
                action_summaries.append(f"Search for '{action.get('query')}'")
            elif action_type == "ADD_TO_OUTLINE":
                action_summaries.append(f"Add new topic '{action.get('topic')}' to the research outline")

        summary_context = f"Agent's Thought Process: {thought}\nNext Actions: {'; '.join(action_summaries)}"
        prompt = [{"role": "system", "content": PROMPTS.AGENT_SUMMARY},{"role": "user", "content": summary_context}]
        summary = await a_chat(prompt, model=Settings.AGENT_SUMMARY_MODEL, temp=0.3, max_tokens=256)
        return summary.strip()