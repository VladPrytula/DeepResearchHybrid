### **Chapter 2: HyDE - Bridging the Semantic Gap with Hypothetical Documents**

In the closing of our previous chapter, we were left with a conundrum: the **Query-Document Mismatch**. A user's query and a comprehensive document, though semantically linked, are structurally alien to one another. Representing them with the same embedding function $E$ can place them in distant regions of our vector space $V$, confounding our cosine similarity metric. How, then, can we build a bridge across this semantic gap?

#### **2.1 Intuition: Answering a Question by First Imagining the Perfect Answer**

Let us engage in a thought experiment. Imagine a legal scholar, let us call her Eve, is asked, "What are the jurisprudential arguments against natural law theory?" Instead of immediately marching to the library and pulling books on jurisprudence, Eva first pauses. In her mind, she constructs an ideal, hypothetical answer. She envisions a paragraph from a legal philosophy textbook, one that might begin, *"The primary critiques of natural law theory often stem from legal positivism, which argues for a separation between law and morality. Positivists like H.L.A. Hart contend that the validity of a law depends on its social sources, not its moral content. Furthermore, the is-ought problem, articulated by David Hume, poses a significant challenge, questioning how one can derive normative legal principles ('ought') from factual observations of human nature ('is')..."*

This imagined text, this *hypothetical document*, has not yet been found. It may not even exist in this exact form. But it serves a powerful purpose. It is a perfect template of relevance. It is dense with the concepts, terminology, and argumentative structure of a truly relevant document.

The scholar can now take this hypothetical answer to the library. Instead of searching for texts that match her terse initial question, she searches for texts that "sound like" her imagined answer. The search is no longer a comparison between a question and a potential answer, but a comparison between an *ideal answer* and *real answers*. This is a far more robust and direct comparison.

This is the core intuition behind **Hypothetical Document Embeddings (HyDE)**. We use the generative power of a Large Language Model (LLM) to perform the same creative leap as our legal scholar. Given a user's query, we first instruct the LLM to generate a short, high-quality document that it believes would perfectly answer that query. We then embed this hypothetical document to get an "ideal answer" vector. This vector, now located in the region of the vector space populated by well-formed answers, becomes our new query vector for searching the corpus.

#### **2.2 Rigor: Formalizing the HyDE Process**

To implement this intuition, we introduce a new component into our system: a generator function.

**Definition 2.1: Hypothetical Document Generator ($G$)**
Let $\mathcal{T}$ be the space of all text strings. The **Hypothetical Document Generator**, denoted by $G$, is a mapping $G: \mathcal{T} \to \mathcal{T}$. In practice, $G$ is a Large Language Model (LLM) prompted to take a query string $q \in \mathcal{T}$ as input and produce a document string $d_h \in \mathcal{T}$ that constitutes a plausible answer to $q$. We call $d_h = G(q)$ a **hypothetical document**.

With this new function, we can now redefine our search process to incorporate the HyDE strategy.

**Definition 2.2: The HyDE Search Process**
Given a query $q \in \mathcal{T}$ and a corpus of documents $C = \{d_1, \dots, d_n\}$:
1.  Generate a hypothetical document from the query:
    $$ d_h = G(q) $$
2.  Compute the embedding of this hypothetical document. This is our new query vector, which we will call the **utility embedding**, $v_u$:
    $$ v_u = E(d_h) = (E \circ G)(q) $$
3.  For each real document $d_i \in C$, compute its embedding:
    $$ v_i = E(d_i) $$
4.  The relevance of document $d_i$ to the query $q$ is now calculated as the similarity between the utility embedding and the document embedding:
    $$ \text{relevance}(d_i, q) = \text{sim}(v_u, v_i) = \text{sim}((E \circ G)(q), E(d_i)) $$
5.  The result of the search is the document $d_k$ that maximizes this new similarity score.

**Remark 2.1:** The beauty of this approach lies in its elegance. We have not altered our embedding function $E$ or our similarity metric $\text{sim}$. Instead, we have introduced a preparatory transformation $G$ on the query itself. The composition $E \circ G$ can be seen as a new, more effective query embedding function, one that maps the query not to its own location in the vector space, but to the location of its ideal answer.

#### **2.3 Application: Translating HyDE Theory into Code**

Now, we shall see how these formal definitions are realized in the agent's Python code. The implementation is split between two components: the `AnalysisComponent`, which provides the tool for generating hypothetical documents, and the `ActionComponent`, which uses this tool to score information during a search.

##### **Implementing the Generator $G$**

The `AnalysisComponent` in `research/analysis.py` contains the method `_generate_hypothetical_document`. This is our concrete implementation of the generator function $G$.

*   **Motivation:** We need a function that takes a string (the query or topic) and returns a new string (the hypothetical document), exactly as specified by Definition 2.1.

*   **Code Introduction & Dissection:**
    ```python
    # In research/analysis.py -> AnalysisComponent

    async def _generate_hypothetical_document(self, topic: str) -> str:
        """Uses an LLM to generate a hypothetical document for a given topic (HyDE)."""
        self.logger.debug(f"Generating HyDE document for topic: '{topic}'")
        prompt = [{"role": "system", "content": PROMPTS.HYDE_GENERATOR}, {"role": "user", "content": topic}]
        doc = await a_chat(prompt, temp=0.4, max_tokens=512)
        if "Error:" in doc:
            self.logger.warning(f"Could not generate HyDE document for '{topic}'. Using topic string as fallback.")
            return topic
        return doc
    ```

*   **Explanation:**
    *   **`async def _generate_hypothetical_document(self, topic: str) -> str:`**: This function signature directly mirrors the definition $G: \mathcal{T} \to \mathcal{T}$, taking a string `topic` and returning a string `doc`.
    *   **`PROMPTS.HYDE_GENERATOR`**: This is a crucial detail. The system prompt given to the LLM explicitly instructs it on its role: *"Write a concise, one-paragraph hypothetical document that answers the following research query... like an encyclopedia entry..."*. This prompt engineers the behavior of the LLM to act as our function $G$.
    *   **`await a_chat(...)`**: This is the asynchronous call to the LLM, which executes the generation.
    *   **`if "Error:" in doc:`**: This is a robustness check. If the LLM fails, the function does not crash. It logs a warning and returns the original topic string. In this failure mode, the HyDE process gracefully degrades to the naive search process from Chapter 1, as $G(q)$ becomes $q$.

##### **Executing the HyDE Search Process**

The `ActionComponent` in `research/actions.py` is responsible for executing searches. Its `act` method orchestrates the full HyDE process (Definition 2.2) to score and select the most relevant chunks of text retrieved from the web.

*   **Motivation:** After performing a web search for a set of queries, the `act` method receives a large number of text chunks from various sources. It cannot add all of them to its knowledge base. It needs a principled way to score each chunk's relevance to the research goal. HyDE provides this principle.

*   **Code Introduction & Dissection:** Let us walk through the key steps in the `act` method.

    1.  **Step 1: Generate all necessary hypothetical documents.**
        The agent plans to search for information related to several `target_outline_topic`s. It begins by generating a hypothetical document for each one.
        ```python
        # In research/actions.py -> ActionComponent.act()

        target_topics = list(set(action['target_outline_topic'] for action in search_actions if action.get('target_outline_topic')))
        
        hyde_generation_tasks = [self.analysis._generate_hypothetical_document(topic) for topic in target_topics]
        hypothetical_docs = await asyncio.gather(*hyde_generation_tasks)
        ```
        This code collects all unique topics, calls our $G$ function (`_generate_hypothetical_document`) for each, and gathers the results. We now have a set of $d_h$ strings.

    2.  **Step 2: Compute the utility embeddings.**
        Next, the agent embeds these hypothetical documents to create the utility vectors, $v_u$.
        ```python
        # In research/actions.py -> ActionComponent.act()

        hyde_embeddings_list = await self.analysis._embed_texts_with_cache(hypothetical_docs)
        
        hyde_embedding_map = {topic: emb for topic, emb in zip(target_topics, hyde_embeddings_list) if emb}
        ```
        This applies our embedding function $E$ (via `_embed_texts_with_cache`) to each $d_h$. The result is a dictionary mapping each topic string to its corresponding utility embedding $v_u$. This is the pre-computation of $(E \circ G)(q)$ for all relevant queries $q$.

    3.  **Step 3 & 4: Fetch real documents and score them against the utility embedding.**
        The agent then proceeds with its web search. For each chunk of text it finds, it computes its relevance.
        ```python
        # In research/actions.py -> ActionComponent.act()

        # ... (code for fetching web content and getting chunk_embeddings) ...

        for i, chunk_emb_list in enumerate(chunk_embeddings):
            # ...
            # Get the correct utility embedding for this action's target topic
            utility_embedding = hyde_embedding_map[target_topic] 
            # ...
            chunk_emb_np = np.array(chunk_emb_list) # This is v_i = E(d_i)
            
            # Here is the core calculation from Definition 2.2
            utility = cosine_similarity(np.array(utility_embedding), chunk_emb_np)
            # ... (the rest of the scoring logic) ...
        ```
        Here, the theory manifests directly. For each new chunk of text, its embedding (`chunk_emb_np`, our $v_i$) is compared against the pre-computed `utility_embedding` (our $v_u$) using `cosine_similarity`. This `utility` score is the concrete implementation of $\text{sim}((E \circ G)(q), E(d_i))$. It is this score that determines which pieces of information are valuable enough to be added to the agent's knowledge base.

By following this though flow from the intuitive idea of imagining an answer, to its formal mathematical definition, to its precise implementation in the agent's action cycle—we see how an abstract concept for improving search can be transformed into a robust and effective software component.