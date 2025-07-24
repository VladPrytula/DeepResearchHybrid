# agent_config.py
import os
import logging
import dotenv

dotenv.load_dotenv()

# --------------------------------------------------------------------------- #
#  Configuration
# --------------------------------------------------------------------------- #
class Settings:
    # --- MODEL & API ---
    AZURE_CHAT_ENDPOINT        = os.getenv("AZURE_CHAT_ENDPOINT")
    AZURE_CHAT_API_KEY         = os.getenv("AZURE_CHAT_API_KEY")
    AZURE_DEPLOYMENT           = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    AZURE_EMBEDDING_ENDPOINT   = os.getenv("AZURE_EMBEDDING_ENDPOINT")
    AZURE_EMBEDDING_API_KEY    = os.getenv("AZURE_EMBEDDING_API_KEY")
    AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
    AZURE_API_VERSION          = os.getenv("AZURE_OPENAI_API_VERSION", "2023-07-01-preview")

    CLIENT_TIMEOUT             = 30.0 # seconds
    CLIENT_MAX_RETRIES         = 3    # Number of retries for API calls
    EMBEDDING_BATCH_SIZE       = 16   # Azure's limit for text-embedding-ada-002

    # --- SEARCH ---
    SEARX_URL                  = os.getenv("SEARX_URL", "http://127.0.0.1:8080/search?q=")
    SEARCH_RESULTS             = int(os.getenv("SEARCH_RESULTS", "8"))
    TOP_K_RESULTS_PER_SECTION  = int(os.getenv("TOP_K_RESULTS_PER_SECTION", "12"))

    # --- PIPELINE ---
    MAX_CYCLES                 = int(os.getenv("MAX_CYCLES", "5"))
    CHUNK_SENTENCES            = 4
    MAX_PAGE_CHARS             = 40_000
    MAX_EMBED_CHARS            = 8_191
    MAX_ABSTRACT_CONTEXT_CHARS = 10_000
    PDF_MIN_TEXT_LENGTH_FALLBACK = 250

    # --- ANALYSIS & AGENT BEHAVIOR ---
    PCA_COMPONENTS             = int(os.getenv("PCA_COMPONENTS", "10"))
    N_CLUSTERS                 = int(os.getenv("N_CLUSTERS", "8"))
    NOVELTY_ALPHA              = 0.65
    NOVELTY_TOP_K              = 15
    DIMINISHING_RETURNS_THRESHOLD = 0.005
    DIMINISHING_RETURNS_WINDOW = 2
    MAX_REFLEXION_LOOPS        = 2
    # WHAT: Adds a flag to enable/disable the latent topic discovery feature. 
    # WHY: This allows for easy comparison between the baseline agent and the more advanced version for educational purposes.
    ENABLE_EXPLORATION         = os.getenv("ENABLE_EXPLORATION", "true").lower() == "true" 
    PCA_COMPONENTS             = int(os.getenv("PCA_COMPONENTS", "10"))
    N_CLUSTERS                 = int(os.getenv("N_CLUSTERS", "8"))
    NOVELTY_ALPHA              = 0.65    

    # --- LOGGING & UI ---
    LOG_LEVEL                  = os.getenv("LOG_LEVEL", "INFO").upper()
    OUTPUT_STYLE               = "summary" # 'detailed', 'summary', or 'progress'
    AGENT_SUMMARY_MODEL        = os.getenv("AGENT_SUMMARY_MODEL", "gpt-4o")

SCRIPT_VERSION = "4.2.3"

# --------------------------------------------------------------------------- #
#  Logging
# --------------------------------------------------------------------------- #
log_format = "%(asctime)s | %(name)-22s | %(levelname)-8s | %(message)s"
logging.basicConfig(level=getattr(logging, Settings.LOG_LEVEL), format=log_format)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("anyio").setLevel(logging.WARNING)
log = logging.getLogger("deep-research")


# --------------------------------------------------------------------------- #
#  Prompts
# --------------------------------------------------------------------------- #
class PROMPTS:
    """Centralized repository for all system and user prompts."""
    PLANNER_CRITIC = f"""You are a meticulous research strategist. Your role is to act as a Planner and Critic.
Based on the current research state, perform three tasks:
1.  **Critique**: Briefly evaluate the progress. Identify the most significant gaps in the research based on the outline. Consider any `Discovered_Latent_Topics` as potential new areas of inquiry.
2.  **Thought**: Reason step-by-step about what to do next. Your goal is to fill the identified gaps. If a discovered latent topic seems highly relevant and unaddressed, you should propose adding it to the outline. If information gain is stalling, propose creative or tangential queries. Consider "stepping back" to formulate a more general query that could provide foundational context. AVOID re-using or creating queries very similar to those already executed.
3.  **Plan**: Formulate a concrete plan as a JSON object. This object MUST have a `plan` key, which is a list of actions. Each action is an object with an `action` key and its parameters.
    - For a web search, use `{{"action": "SEARCH", "query": "...", "target_outline_topic": "..."}}`.
    - To add a new topic to the outline, use `{{"action": "ADD_TO_OUTLINE", "topic": "New Topic Name", "subtopics": ["subtopic1", "subtopic2"]}}`.
The plan MUST also include `critique` (your evaluation) and `thought` (your reasoning) keys.
If you believe the research is complete and all topics are well-covered, return a JSON object like:
`{{"critique": "Research deemed complete.", "thought": "All topics appear to be well-covered, and information gain is low.", "plan": []}}`.
Output ONLY the JSON object. The entire response must be a single valid JSON object. Ensure standard JSON formatting with no trailing commas."""
    AGENT_SUMMARY = "You are a helpful assistant. Your job is to summarize an AI agent's internal monologue into a short (1-2 sentence), user-friendly status update. Explain what the agent just decided and what it's about to do next in simple terms."
    PDF_OCR = [{"type": "text", "text": "Please perform OCR and extract all textual content from this PDF document. Focus on the main body of text, ignoring headers, footers, and page numbers where possible. Present the extracted text as a single, continuous block of plain text. If the document is unreadable, return only an empty string."}, {"type": "image_url", "image_url": {"url": ""}}]
    OUTLINE_DRAFTER = "You are a research analyst. Create a structured JSON outline for a report based on the provided text. It should have 4-5 main topics, each with 2-4 subtopics. Output must be a JSON object with a single key 'outline', where each item in the 'outline' list is an object with 'topic' (or 'title') and 'subtopics' keys."
    SECTION_SYNTHESIZER = """You are a research writer. Your task is to synthesize the provided excerpts into a coherent, detailed, and expressive section for a research report.
The section must cover the given topic comprehensively, drawing from the multiple provided excerpts. Ensure a logical flow and clear explanations.

**CRITICAL INSTRUCTIONS:**
1.  At the end of each sentence, you **MUST** add a citation marker like `[Source ID]` referring to the source of the information.
2.  Use multiple sources if necessary to support a single point.
3.  **Only cite sources that are explicitly provided in the excerpts** (e.g., `[Source 1]`, `[Source 5]`, etc.). Do not invent source numbers.
4.  **DO NOT** add a "References", "Bibliography", or "Works Cited" section at the end of your response. A master bibliography will be generated later. Your output should contain only the synthesized text for the section itself."""
    REFLEXION_REVIEWER = """You are an adversarial reviewer. Your task is to find flaws in the given text and propose a concrete action.
Critically analyze the text for: Logical Gaps, Unsourced Claims, and Vagueness/Overgeneralization.
Respond with a JSON object containing `critique` and `action`.
- If a knowledge gap exists, set `action` to "SEARCH" and provide a `query`.
- If the issue is purely style or logic, set `action` to "REWRITE".
- If no major issues, set `action` to "NONE"."""
    REFLEXION_REWRITER = "You are a research writer. Your previous draft had issues. Revise it based on the reviewer's feedback, using the full context provided (including any new evidence). Ensure every claim is cited correctly. Aim for a comprehensive and well-supported revision."
    HYDE_GENERATOR = "You are a helpful assistant. Write a concise, one-paragraph hypothetical document that answers the following research query or topic. This document should be factual in tone and structure, like an encyclopedia entry or a paragraph from a research paper. It will be used for a vector search to find similar real documents."