# mcp_server.py
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field
import asyncio

from research import run_deep_research

mcp = FastMCP("DeepResearchMCP")

class Report(BaseModel):
    report_markdown: str = Field(description="Full research report in Markdown")

@mcp.tool()
async def deep_research(query: str, output_style: str = "summary") -> Report:
    """
    Multi-hop web research. Returns Markdown.
    output_style: 'summary' | 'detailed' | 'progress'
    """
    md = await run_deep_research(query, output_style)
    return Report(report_markdown=md)

if __name__ == "__main__":
    # stdio is what mcpo wraps by default
    mcp.run()