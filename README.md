# Deep-Research Agent

This project implements an autonomous AI agent designed to perform in-depth research on a given query. It transforms a simple question into a comprehensive, structured, and cited Markdown report. The agent operates through a cyclical process of planning, acting, and analyzing, leveraging both large language models and unsupervised machine learning to explore topics, gather information, and synthesize coherent narratives.

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-blue.svg)](https://www.python.org/)
[![Powered by PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Uses on Azure](https://img.shields.io/badge/Azure-Cloud-0078D4?logo=azure-devops&logoColor=white)](https://azure.microsoft.com/)
[![Uses scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

## Core Methodology

The agent's intelligence is rooted in a modular architecture where distinct components handle different phases of the research process. The core workflow is an agentic loop:

1.  **Plan & Critique:** The agent analyzes its current knowledge base, identifies gaps using topic coverage vectors, and discovers novel research avenues by analyzing latent topics in the collected data. It formulates a step-by-step plan and critiques its own previous actions.
2.  **Act (Search & Acquire):** The agent executes the plan by performing targeted web searches. It uses **Hypothetical Document Embeddings (HyDE)** to generate rich contextual embeddings for its search queries, leading to more semantically relevant results.
3.  **Analyze & Update:** New information is chunked, embedded, and scored based on a combination of relevance to the target topic and novelty (penalizing redundancy against existing knowledge). The agent updates its knowledge base and recalculates its **information gain**, a metric based on the change in its topic coverage vector. The process terminates when this gain falls below a defined threshold, signaling diminishing returns.
4.  **Synthesize & Refine:** Once the research cycles conclude, the agent synthesizes the final report section by section, retrieving the most relevant information chunks for each topic. Each section undergoes a **reflexion pass**—a self-correction loop where an LLM reviews the draft, identifies potential flaws (e.g., lack of detail, logical gaps), and either rewrites the section or triggers further web searches to find missing information before the final rewrite.

## Algorithmic Pillars

-   **PCA + KMeans Clustering:** To facilitate exploration, the agent performs dimensionality reduction (**Principal Component Analysis**) on the embeddings of all collected text chunks. It then applies **KMeans clustering** to this reduced space to identify emergent, latent thematic groups. These discovered topics are fed back into the planner, allowing the agent to dynamically expand the scope of its research beyond the initial outline.
-   **HyDE (Hypothetical Document Embeddings):** For information retrieval, queries are first expanded by an LLM into a full, hypothetical document that answers the question. The embedding of this richer document serves as a superior vector for measuring **cosine similarity** against potential source chunks, improving retrieval accuracy.
-   **Information Gain:** To autonomously decide when to stop researching, the agent models its understanding as a "topic coverage vector," where each dimension represents an outline topic. The information gain for a cycle is the **Euclidean distance** between the coverage vectors of the current and previous states. This quantifies how much new "ground" was covered in a cycle.

## System Architecture & Integration

This research agent is designed to be run both as a standalone CLI tool and as a service integrated into a larger AI ecosystem like **OpenWebUI**. The library exposes a single primary function, `run_deep_research`, for this purpose.

A common integration pattern is to use a model-as-a-tool server, enabling LLMs hosted on platforms like Azure to call this research agent. The data flow is as follows:

```
 User via OpenWebUI         MCPo Proxy              Your Server            This Code
───────────────────      ────────────────      ───────────────────      ───────────────
       │                        │                        │                        │
       │─── HTTP Request ─────>│                        │                        │
       │   (OpenAPI Tool Call)  │                        │                        │
       │                        │─── stdio ────────────>│                        │
       │                        │                        │─── Python import ───>│ `run_deep_research()`
       │                        │                        │                        │
       │<──── Report───────────│<─────────── stdio ─────│<──────────Return──────│
       │       (as text)        │                        │                        │
```

This setup effectively connects a powerful UI and custom LLMs to the specialized research capabilities of this agent.

## How to Run (CLI)

1.  Ensure all dependencies from `requirements.txt` are installed.
2.  Execute the agent from your terminal:

    ```bash
    python main.py "Your research query here"
    ```
3.  A detailed Markdown report will be generated in the root directory.

## Documentation

For a more in-depth exploration of the theoretical depths and a granular breakdown of the agent's architecture, a detailed, paper-style document is available. This documentation is generated using LaTeX and provides a formal overview of the project's design and methodology.

You can find the PDF and source `.tex` files in the `/documentation` directory.

## License

This project is licensed under the MIT License.