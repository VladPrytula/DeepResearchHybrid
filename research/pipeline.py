# research/pipeline.py
import asyncio
import logging
import numpy as np

from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from agent_config import SCRIPT_VERSION, Settings
from agent_helpers import a_embed, hash_txt
from research.actions import ActionComponent
from research.analysis import AnalysisComponent
from research.planning import PlanningComponent
from research.state import ResearchState
from research.synthesis import SynthesisComponent
from research.ui import UIMonitor


class ResearchPipeline:
    """
    The main orchestrator for the deep research process.

    This class initializes and coordinates the different components of the research
    agent (UI, State, Planning, Action, Analysis, Synthesis) to execute a research
    task from start to finish.
    """
    def __init__(self, query: str):
        self.state = ResearchState(query=query)
        self.logger = logging.getLogger(f"deep-research.{hash_txt(query)[:6]}")
        self.ui = UIMonitor(Settings.OUTPUT_STYLE)
        
        # Initialize components
        self.analysis = AnalysisComponent(self.state, self.logger)
        self.planning = PlanningComponent(self.state, self.analysis, self.logger)
        self.actions = ActionComponent(self.state, self.analysis, self.logger)
        self.synthesis = SynthesisComponent(self.state, self.analysis, self.logger)
        
        self.logger.info(f"--- Research-Engine v{SCRIPT_VERSION} initialized for query: '{self.state.query}' ---")

    async def run(self) -> str:
        """Executes the entire research pipeline from start to finish."""
        self.logger.info("--- Starting Research Pipeline ---")
        self.ui.start(self.state.query)
        
        self.state.query_embedding = await a_embed(self.state.query)
        if not self.state.query_embedding:
            self.logger.error("Could not embed initial query. Aborting.")
            return "Error: Could not process the initial query due to an embedding failure."

        await self._initial_setup()
        
        main_progress = None
        if Settings.OUTPUT_STYLE == 'progress':
            main_progress = Progress(SpinnerColumn(), *Progress.get_default_columns(), BarColumn(), TextColumn("Research Cycles"), console=self.ui.console)
            main_progress.start()
            cycle_task = main_progress.add_task("cycles", total=Settings.MAX_CYCLES)

        while self.state.cycles < Settings.MAX_CYCLES:
            self.logger.info(f"--- Starting Agentic Cycle {self.state.cycles + 1}/{Settings.MAX_CYCLES} ---")
            self.ui.update_cycle_start(self.state.cycles + 1, Settings.MAX_CYCLES)
            
            self.ui.start_phase("Planning")
            await self.planning.plan_and_critique()
            self.ui.end_phase()

            if not self.state.plan.get("plan"): 
                self.logger.info("Planner has concluded the research. Moving to synthesis.")
                break
            
            plan_actions = self.state.plan.get("plan", [])
            search_actions = []
            for action in plan_actions:
                action_type = action.get("action", "SEARCH").upper()
                if action_type == "ADD_TO_OUTLINE":
                    new_topic = action.get("topic")
                    if new_topic and isinstance(new_topic, str):
                        self.logger.info(f"Planner requested to add topic to outline: '{new_topic}'")
                        if not any(o.get('topic') == new_topic for o in self.state.outline):
                            self.state.outline.append({
                                "topic": new_topic,
                                "subtopics": action.get("subtopics", [])
                            })
                            if self.ui._is_active():
                                self.ui.console.print(Panel(f"New research area added to outline: [bold cyan]{new_topic}[/bold cyan]", title="[bold yellow]🧬 Outline Evolved[/bold yellow]", border_style="yellow"))
                        else:
                            self.logger.warning(f"Skipping request to add duplicate topic to outline: '{new_topic}'")
                elif action_type == "SEARCH":
                    search_actions.append(action)

            self.ui.show_agent_plan(await self.planning.generate_agent_summary())
            
            if search_actions:
                self.ui.start_phase("Executing Searches & Analyzing with HyDE")
                newly_added_info, num_new_chunks = await self.actions.act(search_actions)
                self.ui.end_phase()
                self.ui.show_action_summary(newly_added_info, num_new_chunks)
            else:
                self.logger.info("No search actions in this cycle. Skipping action phase.")
                await asyncio.sleep(0.5)

            await self.analysis.update_information_gain()
            
            if main_progress: main_progress.update(cycle_task, advance=1)

            if self.analysis.check_diminishing_returns():
                self.ui.show_diminishing_returns(np.mean(self.state.information_gain_history[-Settings.DIMINISHING_RETURNS_WINDOW:]))
                self.logger.info("Diminishing returns detected. Concluding research phase.")
                break

            self.state.cycles += 1
        
        if main_progress: main_progress.stop()
        
        self.logger.info("--- Maximum cycles reached or stopping criteria met. Moving to Synthesis. ---")
        self.ui.start_synthesis()
        return await self.synthesis.synthesise()

    async def _initial_setup(self):
        """Performs the initial search and outline drafting."""
        self.logger.info("--- Performing Initial Setup ---")
        self.ui.start_phase("Performing initial search & drafting outline")
        
        boot_queries = await self.planning.generate_queries(self.state.query, 3, "diverse, broad web search queries to get an overview")
        initial_plan_actions = [{"action": "SEARCH", "query": q, "target_outline_topic": self.state.query} for q in boot_queries]
        self.state.plan = {"plan": initial_plan_actions, "thought": "Initial exploratory search.", "critique": "N/A for initial setup."}
        
        newly_added_info, num_new_chunks = await self.actions.act(initial_plan_actions)
        self.state.outline = await self.planning.draft_outline()
        self.logger.info(f"Drafted outline with {len(self.state.outline)} main topics.")
        if not self.state.outline or not any(item.get('topic') for item in self.state.outline):
            self.logger.warning("Outline is empty or malformed after initial draft. Using default query as single topic.")
            self.state.outline = [{"topic": self.state.query, "subtopics": []}]

        self.state.cycles = 1 
        
        self.ui.end_phase()
        self.ui.show_action_summary(newly_added_info, num_new_chunks)

# WHAT: The public API function is moved here from main.py.
# WHY: This function is the primary entry point for using the pipeline. Placing it here alongside the ResearchPipeline class is more logical and makes the package's structure cleaner.
async def run_deep_research(query: str, output_style: str = "summary") -> str:
    """
    Public API function to run the deep research process.
    Returns a Markdown report as a string.
    """
    Settings.OUTPUT_STYLE = output_style
    engine = ResearchPipeline(query)
    report = await engine.run()
    return report        