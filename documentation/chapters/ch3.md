### **Chapter 3: The Agentic Loop: A Framework for Iterative Inquiry**

The HyDE process, as described, provides a sophisticated method for a one-shot search. But a true researcher does not simply ask one question, receive an answer, and consider the work done. They engage in a cycle: they assess what they have learned, identify gaps, formulate a new plan of attack, and execute it. This iterative cycle is the very essence of inquiry. Our agent must embody this process.

#### **3.1 The Abstract Machine: The Plan-Critique-Act-Analyze Cycle**

To build an autonomous researcher, we must first define its "mental" process. We can model this as a state machine that perpetually cycles through four distinct phases. This is the **agentic loop**, the cognitive engine of our system.

1.  **Plan & Critique:** The cycle begins with introspection. The agent examines its current state of knowledge. What is the overall research goal? What topics in the outline are well-supported with information? Which are sparse? Are there any newly discovered concepts that warrant investigation? Based on this analysis, the agent formulates a concrete plan for the next cycle, consisting of a series of actions (e.g., "search for 'quantum computing and cryptography'") designed to fill the most pressing knowledge gaps. This phase includes a crucial **critique** step, where the agent evaluates its own progress, much like a human researcher pausing to ask, "Am I on the right track?"

2.  **Act:** With a plan in hand, the agent interacts with the external world. It executes the actions defined in the plan—primarily performing web searches, fetching content from URLs, and parsing it into usable text. This is the information-gathering phase, where the agent acquires the raw material for its knowledge base.

3.  **Analyze & Update:** The raw material from the `Act` phase is not yet knowledge. It must be processed and integrated. In this phase, the agent takes the newly fetched text chunks, scores them for relevance and novelty (using techniques like HyDE), and decides which ones to add to its persistent knowledge base. It then updates its internal metrics, such as its understanding of topic coverage and its progress over time.

This cycle, `Plan → Act → Analyze → (new) Plan...`, is not a simple loop. It is a spiral. With each iteration, the agent's internal state of knowledge expands, its understanding deepens, and its subsequent plans become more sophisticated and targeted.

#### **3.2 The State Monad: The Carrier of Knowledge**

For the agentic loop to function, there must be a persistent, unified representation of the agent's knowledge and intentions. This is the **research state**. All phases of the loop read from and write to this central state.

From a computer science perspective, we can think of this state as a monadic-like container. Each major function in our cycle (plan, act, analyze) takes the current state as input and produces a new, updated state as output. This ensures a clean, predictable flow of data and prevents the kind of scattered, implicit state management that plagues complex systems.

Our implementation codifies this concept in the `ResearchState` dataclass. This is not merely a data structure; it is the embodiment of the agent's "mind" at any given moment.

*   **Motivation:** We need a single, well-defined Python object that can hold every piece of information relevant to the research task as it evolves over multiple cycles.

*   **Code Introduction & Dissection:** Let us examine the definition in `research/state.py`.
    ```python
    # In research/state.py

    from dataclasses import dataclass, field
    from typing import Any, Dict, List, Optional, Tuple
    import numpy as np

    @dataclass
    class ResearchState:
        """
        Manages the state of a research task.
        ...
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
    ```

*   **Explanation of Key Fields:**
    *   `query`: The immutable, original question that initiated the research.
    *   `cycles`: A simple integer counter, tracking our position in the agentic loop.
    *   `outline`: This is the agent's high-level strategic map. It's a list of topics that the final report should cover. This `outline` can be modified and expanded during the `Plan & Critique` phase.
    *   `plan`: The output of the `Plan & Critique` phase for the *current* cycle. It contains the specific actions to be performed next.
    *   `critique_history`: A list of the critiques generated in each cycle, allowing the agent to avoid repeating past mistakes.
    *   `all_chunks`, `results`, `chunk_embedding_cache`, `url_to_source_index`: These fields collectively form the agent's **knowledge base**. They store the raw text chunks, their vector embeddings, and their source metadata. This is the data that grows with each cycle.
    *   `information_gain_history`, `last_coverage_vector`: These are metacognitive fields. They store data about the *rate of learning*, which is crucial for the agent to decide when its research is complete, a topic we will explore in depth in Chapter 5.

Every component we will discuss from now on will interact with this `ResearchState` object, reading from it to make decisions and writing to it to record new knowledge.

#### **3.3 Architectural Blueprint: From Abstract Loop to Concrete Code**

We have the abstract loop and the state object. Now, let's map this theory onto the project's file structure. The system's architecture is a direct reflection of the agentic loop. The primary orchestrator is the `ResearchPipeline` class, and it coordinates three specialist components: `PlanningComponent`, `ActionComponent`, and `AnalysisComponent`.

1.  **The Orchestrator (`ResearchPipeline` in `pipeline.py`):** This class owns the main `while` loop. Its `run` method is the top-level entry point that executes the entire research process from start to finish.

    *   **Leading the Reader to the Code:** Observe the `run` method within `ResearchPipeline`. You will find a `while self.state.cycles < Settings.MAX_CYCLES:` loop. This is the literal implementation of our agentic loop. Inside this loop, you will see a sequence of calls that perfectly match our abstract machine.

    ```python
    # In research/pipeline.py -> ResearchPipeline.run()

    # ... inside the while loop ...

    # Phase 1: Plan & Critique
    self.ui.start_phase("Planning")
    await self.planning.plan_and_critique()
    self.ui.end_phase()

    # ... (logic to handle the plan) ...

    # Phase 2: Act
    if search_actions:
        self.ui.start_phase("Executing Searches & Analyzing with HyDE")
        newly_added_info, num_new_chunks = await self.actions.act(search_actions)
        self.ui.end_phase()
    
    # Phase 3: Analyze & Update
    await self.analysis.update_information_gain()

    # Check for completion and increment the cycle
    if self.analysis.check_diminishing_returns():
        break
    self.state.cycles += 1
    ```

2.  **The Components (`planning.py`, `actions.py`, `analysis.py`):**
    *   **`PlanningComponent`:** This component is responsible for the "Plan & Critique" phase. Its core method, `plan_and_critique`, is the focus of our next section.
    *   **`ActionComponent`:** This component handles the "Act" phase. Its `act` method reads the `plan` from the state, executes the specified web searches, and uses HyDE to score and add new information to the knowledge base.
    *   **`AnalysisComponent`:** This component is a collection of analytical tools used throughout the loop. It provides `_generate_hypothetical_document` to the `ActionComponent`, `calculate_topic_coverage` to the `PlanningComponent`, and `update_information_gain` to the main `ResearchPipeline`.

This clean separation of concerns, mirroring the abstract agentic loop, is the hallmark of a well-designed agentic system. It allows for modularity and comprehension, enabling us to analyze and improve each cognitive faculty—planning, acting, analyzing—in isolation.

#### **3.4 A Deeper Look into the Planning & Critique Phase**

The "Plan & Critique" phase is where the agent's intelligence is most apparent. Unlike the `Action` phase, which executes a well-defined search-and-score algorithm, the `Planning` phase must reason under uncertainty. It must synthesize diverse inputs—topic coverage, information gain trends, latent topic discovery, and its own past actions—to make a strategic decision about what to do next.

##### **Intuition: Mimicking the Metacognitive Researcher**

A human researcher does not plan their next steps in a vacuum. They spread out their notes, look at their outline, and engage in a form of metacognition. They ask themselves:
*   "Which parts of my argument are weak?" (Critique)
*   "What do I know, and what do I need to find out?" (Gap Analysis)
*   "Have I been going in circles? Should I try a different angle?" (Self-Correction)
*   "This unexpected idea keeps coming up; maybe I should investigate it." (Exploration)

Our agent's planning phase is designed to emulate this exact process. We do not provide the agent with a rigid, deterministic planning algorithm. Instead, we provide it with all the necessary information—a comprehensive summary of its current state—and instruct a Large Language Model (LLM) to act as a "meticulous research strategist." The rigor in this approach comes not from a mathematical proof, but from the formal, structured way we constrain this reasoning process.

##### **Rigor: Formalizing the Planner's Task**

To make the LLM's reasoning reliable and its output usable, we must rigorously define its inputs and outputs.

**Definition 3.1: The State Summary ($S_{summary}$)**
The **State Summary** is a structured text string that serves as the complete informational input to the planning LLM. It is a serialization of the agent's current "mental state," constructed from the `ResearchState` object. It includes:
*   The original research query.
*   The current `outline` of the report.
*   A quantitative summary of `Topic Coverage` for each outline topic.
*   A qualitative summary of the `Information Gain Trend` (e.g., "Increasing," "Stalling").
*   A list of `Discovered Latent Topics` that are not yet in the outline (if exploration is enabled).
*   A history of previously executed search queries to avoid repetition.
*   The critique from the previous cycle to encourage progress.

**Definition 3.2: The Planner's Output Schema**
The output of the planning LLM is not free-form text. It is constrained to be a JSON object conforming to a strict schema. This ensures the agent's "thoughts" can be parsed and acted upon by the system. The schema is as follows:
```json
{
  "critique": "A brief evaluation of the previous cycle's progress and the current state.",
  "thought": "A step-by-step reasoning process explaining the rationale for the new plan.",
  "plan": [
    {"action": "ACTION_TYPE", "parameters": "..."},
    ...
  ]
}
```
*   `critique`: A string containing the LLM's evaluation. This is stored in `critique_history`.
*   `thought`: A string containing the LLM's reasoning. This is for logging and debugging, providing insight into the agent's "mind."
*   `plan`: A list of action objects. The primary `ACTION_TYPE` is `SEARCH`, which includes a `query` and a `target_outline_topic`. Another is `ADD_TO_OUTLINE`, which allows the agent to dynamically expand its own research goals.

The explicit instruction to the LLM to perform these three tasks (`Critique`, `Thought`, `Plan`) and to structure the output in this specific JSON format is the core of our "planning function." Let us examine the prompt that enforces this behavior.

*   **The Planner's "Source Code": The System Prompt**
    The function that governs the planner's behavior is the system prompt stored in `agent_config.py` as `PROMPTS.PLANNER_CRITIC`.

    ```python
    # In agent_config.py -> class PROMPTS
    
    PLANNER_CRITIC = f"""You are a meticulous research strategist. Your role is to act as a Planner and Critic.
Based on the current research state, perform three tasks:
1.  **Critique**: Briefly evaluate the progress. Identify the most significant gaps...
2.  **Thought**: Reason step-by-step about what to do next...
3.  **Plan**: Formulate a concrete plan as a JSON object...
    - For a web search, use `{{"action": "SEARCH", ...}}`.
    - To add a new topic to the outline, use `{{"action": "ADD_TO_OUTLINE", ...}}`.
...
Output ONLY the JSON object. The entire response must be a single valid JSON object."""
    ```
    This prompt is not merely a suggestion; it is a contract. It defines the LLM's persona, its required tasks, and, most critically, the precise format of its output. The instruction "Output ONLY the JSON object" is crucial for robust parsing.

##### **Application: Implementation in the `PlanningComponent`**

We now turn our attention to `research/planning.py` to see how these formal concepts are translated into running code within the `plan_and_critique` method.

1.  **Step 1: Assembling the State Summary.** The method begins by gathering all the necessary information from the `ResearchState` object to construct the `state_summary` string, precisely as specified in Definition 3.1.
    ```python
    # In research/planning.py -> PlanningComponent.plan_and_critique()

    self.logger.info("--- Agent Step: Planning & Critiquing ---")
    _, coverage_summary = await self.analysis.calculate_topic_coverage()
    previous_queries = list(set(res['query'] for res in self.state.results if 'reflexion' not in res.get('query', '')))
    gain_trend = self.analysis.get_gain_trend_description()
    
    # ... (code for latent topic discovery) ...

    state_summary = f"""
    Original Query: {self.state.query}
    Report Outline: {json.dumps(self.state.outline, indent=2)}
    Current Topic Coverage: {coverage_summary}
    Discovered Latent Topics: {latent_topics_summary}
    ...
    """
    ```
    This block is the practical implementation of creating $S_{summary}$. It calls methods on the `AnalysisComponent` to get metrics like coverage and gain, and then formats them into a comprehensive string.

2.  **Step 2: Invoking the Planner LLM.** The `state_summary` is then packaged with the system prompt and sent to the LLM.
    ```python
    # In research/planning.py -> PlanningComponent.plan_and_critique()
    
    prompt = [{"role": "system", "content": PROMPTS.PLANNER_CRITIC}, {"role": "user", "content": state_summary}]
    raw_response = await a_chat(prompt, temp=0.2, max_tokens=1024)
    ```
    This is the execution of our planning function. The `a_chat` helper (from `agent_helpers.py`) handles the API call.

3.  **Step 3: Parsing the Response and Updating State.** The raw text response from the LLM must be parsed into the structured JSON defined by our schema. This is a potential point of failure, so a robust helper function, `extract_json_from_response`, is used.
    ```python
    # In research/planning.py -> PlanningComponent.plan_and_critique()

    json_to_parse = extract_json_from_response(raw_response)
    if not json_to_parse:
        # ... (error handling) ...
        return

    try:
        response_json = json.loads(json_to_parse)
        self.state.plan = response_json # Update the state
        
        critique = response_json.get("critique", "N/A")
        self.state.critique_history.append(critique) # Update the state

    except json.JSONDecodeError as e:
        # ... (error handling) ...
    ```
    This code block completes the cycle. It first uses a regular-expression-based helper to find the JSON block in the LLM's response, making the system resilient to extraneous text. It then parses this JSON and uses the result to update the `self.state.plan` and `self.state.critique_history` fields. This new state is now ready for the `ActionComponent` to execute in the next phase of the agentic loop.

By formalizing the LLM's task as a mapping from a well-defined state summary to a structured JSON output, we transform the fuzzy notion of "planning" into a concrete, repeatable, and surprisingly effective computational process. This concludes our deep dive into the agent's cognitive engine. We now turn to a more advanced capability: the agent's ability to not just follow its plan, but to discover entirely new avenues of inquiry.