# research/ui.py
from pathlib import Path
from typing import Dict, List

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from agent_config import SCRIPT_VERSION, Settings


class UIMonitor:
    """Handles all user-facing, non-detailed logging using the `rich` library."""
    def __init__(self, output_style: str):
        self.style = output_style
        self.console = Console()
        self.live_progress = None

    def _is_active(self) -> bool:
        return self.style in ["summary", "progress"]

    def start(self, query: str):
        if not self._is_active(): return
        self.console.print(Panel(
            f"[bold magenta]🔬 Starting Deep Research[/bold magenta]\n[cyan]Query:[/] \"{query}\"",
            title=f"[bold green]Research Engine v{SCRIPT_VERSION}[/bold green]",
            border_style="green"
        ))

    def start_phase(self, text: str):
        if self.style != 'summary': return
        self.live_progress = Progress(SpinnerColumn(), TextColumn(f"[bold blue]{text}..."), console=self.console)
        self.live_progress.start()
    
    def end_phase(self):
        if self.live_progress:
            self.live_progress.stop()
            self.live_progress = None

    def update_cycle_start(self, cycle: int, max_cycles: int):
        if self.style == 'summary':
            self.console.print(f"\n[bold]🔄 Cycle {cycle}/{max_cycles}: Planning next steps...[/bold]")
    
    def show_agent_plan(self, summary: str):
        if self.style == 'summary':
            self.console.print(Panel(summary, title="[bold yellow]🧠 Agent Plan[/bold yellow]", border_style="yellow"))

    def show_action_summary(self, newly_added_info: Dict[str, List[str]], num_new_chunks: int):
        if self.style != 'summary': return
        total_sources = sum(len(titles) for titles in newly_added_info.values())
        if num_new_chunks == 0:
            self.console.print("[yellow]⚠️ No new information was added in this cycle.[/yellow]")
            return
        table = Table(title=f"[bold green]📚 Added {num_new_chunks} new info snippets from {total_sources} source(s)[/bold green]", show_header=True, header_style="bold cyan")
        table.add_column("Search Query", style="dim", width=40)
        table.add_column("Found Source Title")
        for query, titles in newly_added_info.items():
            if not titles: continue
            for i, title in enumerate(titles):
                table.add_row(f'"{query}"' if i == 0 else "", f"• {title}")
        self.console.print(table)

    def show_diminishing_returns(self, gain: float):
        if not self._is_active(): return
        self.console.print(f"[bold yellow]⚠️ Diminishing returns detected (avg. gain {gain:.4f} < {Settings.DIMINISHING_RETURNS_THRESHOLD}). Moving to synthesis.[/bold yellow]")

    def start_synthesis(self):
        if not self._is_active(): return
        self.console.print(Panel("[bold blue]✍️ All research cycles complete. Synthesizing final report...[/bold blue]", border_style="blue"))

    def end(self, report_path: Path):
        if not self._is_active(): return
        self.console.print(Panel(
            f"[bold green]🎉 Report Finished![/bold green]\n[cyan]Full report saved to:[/] {report_path.resolve()}",
            title="[bold green]Synthesis Complete[/bold green]",
            border_style="green"
        ))