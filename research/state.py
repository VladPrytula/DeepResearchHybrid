# research/state.py
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ResearchState:
    """
    Manages the state of a research task.

    This dataclass holds all the data generated and used during the research process,
    including the initial query, evolving outline, collected information, and various
    analytical artifacts.
    """
    query: str
    query_embedding: Optional[List[float]] = None
    cycles: int = 0
    outline: List[Dict[str, Any]] = field(default_factory=list)
    plan: Dict[str, Any] = field(default_factory=dict)
    critique_history: List[str] = field(default_factory=list)
    information_gain_history: List[float] = field(default_factory=list)
    last_coverage_vector: Optional[np.ndarray] = None
    results: List[Dict[str, Any]] = field(default_factory=list)
    all_chunks: List[Tuple[str, int]] = field(default_factory=list)
    chunk_embedding_cache: Dict[str, List[float]] = field(default_factory=dict)
    url_to_source_index: Dict[str, int] = field(default_factory=dict)