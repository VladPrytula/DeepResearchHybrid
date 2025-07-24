# main.py
import argparse
import asyncio
import logging
import re
from pathlib import Path

from rich.console import Console

from agent_config import SCRIPT_VERSION, Settings, log
from agent_helpers import hash_txt
# WHAT: The ResearchPipeline is now imported directly from the package.
# WHY: This simplifies the import statement and aligns with the new package structure.
from research.pipeline import ResearchPipeline


async def main_cli():
    """The main command-line interface function."""
    parser = argparse.ArgumentParser(description=f"Deep-Research Agent v{SCRIPT_VERSION}", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("question", nargs="+", help="Your research prompt")
    parser.add_argument("-s", "--output-style", choices=["detailed", "summary", "progress"], default="summary", help="""Choose the output style:
  - detailed: Show all verbose logs.
  - summary: Show high-level agent summaries per cycle (default).
  - progress: Show a minimal progress bar.""")
    args = parser.parse_args()
    
    Settings.OUTPUT_STYLE = args.output_style
    if Settings.OUTPUT_STYLE == "detailed":
        logging.getLogger().setLevel(Settings.LOG_LEVEL) 
        log.setLevel(Settings.LOG_LEVEL) 
    else: 
        logging.getLogger().setLevel(logging.CRITICAL + 10) 
        log.setLevel(logging.INFO) 

    question_str = " ".join(args.question)
    
    # Basic cleaning for common CLI argument parsing issues
    cleaned_question = question_str
    if " -- output-style " in question_str:
        log.warning("Detected ' -- output-style ' in query. This might be a CLI usage error. Attempting to clean.")
        cleaned_question = question_str.split(" -- output-style ")[0].strip()
    elif question_str.endswith(" output-style summary") or question_str.endswith(" output-style detailed") or question_str.endswith(" output-style progress"):
        parts = question_str.split(" output-style ")
        if len(parts) > 1 and parts[-1] in ["summary", "detailed", "progress"]:
            log.warning(f"Detected 'output-style {parts[-1]}' at the end of the query. Cleaning it.")
            cleaned_question = " output-style ".join(parts[:-1]).strip()

    engine = ResearchPipeline(cleaned_question)
    report = await engine.run()
    
    report_filename_base = re.sub(r'[^\w\s-]', '', cleaned_question.lower())
    report_filename_base = re.sub(r'[-\s]+', '_', report_filename_base)[:50] 
    report_path = Path(f"report_{report_filename_base}_{hash_txt(cleaned_question)[:8]}.md")
    
    try:
        report_path.write_text(report, encoding="utf-8")
    except Exception as e:
        print(f"[ERROR] Failed to write report to {report_path}: {e}") 
        log.error(f"Failed to write report to {report_path}: {e}")
        report_path = Path(f"report_fallback_{hash_txt(cleaned_question)[:8]}.md")
        try:
            report_path.write_text(report, encoding="utf-8")
            print(f"[INFO] Report saved to fallback path: {report_path.resolve()}")
            log.info(f"Report saved to fallback path: {report_path.resolve()}")
        except Exception as e_fb:
            print(f"[ERROR] Failed to write report to fallback path {report_path}: {e_fb}")
            log.error(f"Failed to write report to fallback path {report_path}: {e_fb}")

    if Settings.OUTPUT_STYLE == 'detailed':
        log.info("--- FINAL REPORT ---") 
        print("\n" + ("="*80) + "\nFINAL REPORT\n" + ("="*80) + "\n")
        print(report)
        print("\n" + ("="*80))
        print(f"\n[INFO] Full report saved to: {report_path.resolve()}")
    else:
        engine.ui.end(report_path)

async def run_deep_research(query: str, output_style: str = "summary") -> str:
    """
    Public API function to run the deep research process.
    Returns a Markdown report as a string.
    """
    Settings.OUTPUT_STYLE = output_style
    engine = ResearchPipeline(query)
    report = await engine.run()
    return report

# WHAT: The local definition of run_deep_research is removed.
# WHY: This function has been moved into the `research` package to serve as the official
# public API, making it accessible to other parts of your application, like mcp_server.py.
# We will rely on the package's version for any future use.


if __name__ == "__main__":
    try:
        asyncio.run(main_cli())
    except KeyboardInterrupt:
        print("\n--- Process interrupted by user. Shutting down. ---")
        console = Console()
        console.show_cursor(True)
    except Exception as e:
        print(f"\n--- A critical error occurred in the main process: {e} ---")
        log.error("--- A critical error occurred in the main process ---", exc_info=True) 
        console = Console() 
        console.print(f"\n[bold red]A critical error occurred: {e}. Please check the logs.[/bold red]")
        console.print_exception(show_locals=True)