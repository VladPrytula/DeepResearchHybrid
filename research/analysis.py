# research/analysis.py
import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from agent_config import PROMPTS, Settings
from agent_helpers import (a_chat, a_embed_batch, cosine_similarity,
                           hash_txt, EMBED_CACHE)

# Forward declaration for type hinting
class ResearchState:
    pass

class AnalysisComponent:
    """
    Handles all data analysis tasks for the research pipeline.

    This includes calculating topic coverage, discovering latent topics through
    unsupervised learning (PCA + KMeans), and tracking information gain to
    determine when research should conclude.
    """
    def __init__(self, state: 'ResearchState', logger: logging.Logger):
        self.state = state
        self.logger = logger

    async def _generate_hypothetical_document(self, topic: str) -> str:
        """Uses an LLM to generate a hypothetical document for a given topic (HyDE)."""
        self.logger.debug(f"Generating HyDE document for topic: '{topic}'")
        prompt = [{"role": "system", "content": PROMPTS.HYDE_GENERATOR}, {"role": "user", "content": topic}]
        doc = await a_chat(prompt, temp=0.4, max_tokens=512)
        if "Error:" in doc:
            self.logger.warning(f"Could not generate HyDE document for '{topic}'. Using topic string as fallback.")
            return topic
        return doc
    
    async def _embed_texts_with_cache(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Embeds a list of texts, utilizing a cache to avoid redundant API calls."""
        texts_to_embed, indices_to_embed, final_embeddings = [], [], [None] * len(texts)
        for i, text in enumerate(texts):
            if not text: continue
            h = hash_txt(text)
            # Check both the session-specific cache and the global cache
            cached_emb = self.state.chunk_embedding_cache.get(h) or EMBED_CACHE.get(h)
            if cached_emb:
                final_embeddings[i] = cached_emb
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)
        
        if texts_to_embed:
            all_new_embeddings: List[Optional[List[float]]] = []
            batches = [texts_to_embed[i:i + Settings.EMBEDDING_BATCH_SIZE] for i in range(0, len(texts_to_embed), Settings.EMBEDDING_BATCH_SIZE)]
            
            batch_results_futures = [a_embed_batch(batch) for batch in batches]
            batch_results_list_of_lists = await asyncio.gather(*batch_results_futures)
            
            for batch_result in batch_results_list_of_lists:
                all_new_embeddings.extend(batch_result)

            for i, emb in enumerate(all_new_embeddings):
                if emb:
                    original_index = indices_to_embed[i]
                    final_embeddings[original_index] = emb
                    # Put the new embedding into the global cache
                    EMBED_CACHE.put(hash_txt(texts_to_embed[i]), emb)
        return final_embeddings

    def get_gain_trend_description(self) -> str:
        """Analyzes the recent history of information gain to describe its trend."""
        history = self.state.information_gain_history
        if len(history) < 2: return "Just starting."
        recent_gains = history[-3:]
        if len(recent_gains) < 2: return "Stable."
        avg_gain = np.mean(recent_gains)
        if avg_gain < Settings.DIMINISHING_RETURNS_THRESHOLD: return "Stalling (very low gain)."
        if recent_gains[-1] > recent_gains[-2]: return "Increasing."
        elif recent_gains[-1] < recent_gains[-2] * 0.75: return "Decreasing."
        else: return "Stable."

    async def calculate_topic_coverage(self) -> Tuple[Optional[np.ndarray], str]:
        """
        Calculates the coverage of outline topics by the collected information chunks.

        Returns a vector of coverage scores and a human-readable summary.
        """
        if not self.state.outline or not self.state.all_chunks: return None, "Not enough data for coverage analysis."
        
        outline_topic_texts = [t.get('topic') for t in self.state.outline if isinstance(t, dict) and t.get('topic')]
        if not outline_topic_texts: return None, "Outline is malformed or empty (no topic strings)."

        hyde_docs_for_outline = await asyncio.gather(*(self._generate_hypothetical_document(ot) for ot in outline_topic_texts))
        outline_embeddings = await self._embed_texts_with_cache(hyde_docs_for_outline)
        
        valid_outline_data = [(ot, o_emb) for ot, o_emb in zip(outline_topic_texts, outline_embeddings) if o_emb]
        if not valid_outline_data: return None, "Could not embed outline topics."

        cached_chunk_embeddings_values = [emb for emb in self.state.chunk_embedding_cache.values() if emb is not None]
        if not cached_chunk_embeddings_values: return None, "No valid chunk embeddings in cache for coverage."
        
        cached_chunk_embeddings_np = [np.array(emb) for emb in cached_chunk_embeddings_values]

        coverage_scores = {}
        for ot_text, o_emb_list in valid_outline_data:
            o_emb_np = np.array(o_emb_list)
            sims_to_chunks = [cosine_similarity(o_emb_np, c_emb_np) for c_emb_np in cached_chunk_embeddings_np]
            coverage_scores[ot_text] = max(sims_to_chunks) if sims_to_chunks else 0.0
        
        summary = ", ".join([f"'{k}': {v:.2f}" for k, v in coverage_scores.items()])
        return np.array(list(coverage_scores.values())), summary

    async def get_latent_topics(self) -> List[Dict[str, Any]]: 
        """
        Discovers latent topics from chunk embeddings using PCA for dimensionality
        reduction and KMeans for clustering.
        """
        if len(self.state.chunk_embedding_cache) < Settings.N_CLUSTERS: return []
        
        texts_and_embs_for_clustering = []
        for chunk_text, _ in self.state.all_chunks: 
            text_hash = hash_txt(chunk_text)
            emb = self.state.chunk_embedding_cache.get(text_hash)
            if emb is not None:
                texts_and_embs_for_clustering.append({'text': chunk_text, 'emb': np.array(emb)})
        
        if len(texts_and_embs_for_clustering) < Settings.N_CLUSTERS: return []

        embeddings_np_array = np.array([item['emb'] for item in texts_and_embs_for_clustering])
        original_texts_ordered = [item['text'] for item in texts_and_embs_for_clustering]

        n_components = min(Settings.PCA_COMPONENTS, embeddings_np_array.shape[0], embeddings_np_array.shape[1])
        if n_components <= 1: return [] 

        pca = PCA(n_components=n_components)
        reduced_embeddings = pca.fit_transform(embeddings_np_array)
        
        actual_n_clusters = min(Settings.N_CLUSTERS, len(reduced_embeddings))
        if actual_n_clusters <= 1: return []

        kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(reduced_embeddings)
        
        cluster_tasks = []
        for i in range(actual_n_clusters):
            current_cluster_texts = [original_texts_ordered[j] for j, label in enumerate(cluster_labels) if label == i]
            if not current_cluster_texts: continue
            
            sample = "\n- ".join(current_cluster_texts[:5]) 
            prompt = [{"role": "system", "content": "Read these text snippets from a research cluster. Provide a concise, 3-5 word topic label for them."},
                      {"role": "user", "content": f"Snippets:\n- {sample[:3000]}"}]
            cluster_tasks.append(a_chat(prompt, temp=0.2, max_tokens=16))
        
        gathered_labels = await asyncio.gather(*cluster_tasks)
        return [{"label": label, "id": i} for i, label in enumerate(gathered_labels) if not label.startswith("Error:")]

    async def update_information_gain(self):
        """
        Calculates and records the information gain for the current cycle.
        
        Information gain is the Euclidean distance between the topic coverage vector
        of the current state and the previous state. It resets if the outline changes.
        """
        self.logger.debug("Updating information gain...")
        coverage_vector, _ = await self.calculate_topic_coverage()
        if coverage_vector is None: return

        if self.state.last_coverage_vector is not None:
            if coverage_vector.shape == self.state.last_coverage_vector.shape:
                gain = np.linalg.norm(coverage_vector - self.state.last_coverage_vector)
                self.state.information_gain_history.append(gain)
                self.logger.info(f"Information Gain this cycle: {gain:.4f}")
            else:
                self.logger.info("Outline has changed shape. Resetting information gain history.")
                self.state.information_gain_history = [1.0] # Reset with high gain
        
        self.state.last_coverage_vector = coverage_vector

    def check_diminishing_returns(self) -> bool:
        """
        Checks if the average information gain over a recent window has fallen
        below a predefined threshold, indicating that research should stop.
        """
        if len(self.state.information_gain_history) < Settings.DIMINISHING_RETURNS_WINDOW: return False
        avg_gain = np.mean(self.state.information_gain_history[-Settings.DIMINISHING_RETURNS_WINDOW:])
        if avg_gain < Settings.DIMINISHING_RETURNS_THRESHOLD:
            self.logger.warning(f"Avg gain ({avg_gain:.4f}) is below threshold. Stopping.")
            return True
        return False