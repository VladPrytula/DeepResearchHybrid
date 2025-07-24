# agent_helpers.py
import asyncio
import base64
import hashlib
import json
import logging
import re
from typing import Any, Dict, List, Optional

import aiohttp
import fitz
from bs4 import BeautifulSoup
from curl_cffi.requests import AsyncSession
from openai import AsyncAzureOpenAI, Timeout

from agent_config import Settings, PROMPTS, log

# --------------------------------------------------------------------------- #
# 1.  API Clients, Wrappers & Caching
# --------------------------------------------------------------------------- #
_chat_client: Optional[AsyncAzureOpenAI] = None
_embedding_client: Optional[AsyncAzureOpenAI] = None

class _Cache(dict):
    def __init__(self, cap: int = 10_000): super().__init__(); self.cap = cap
    def __getitem__(self, k): return super().get(k)
    def put(self, k, v):
        if len(self) >= self.cap: self.pop(next(iter(self)))
        self[k] = v
EMBED_CACHE = _Cache(50_000)
CONTENT_CACHE = _Cache(2_000)

def get_chat_client() -> AsyncAzureOpenAI:
    global _chat_client
    if _chat_client is None:
        log.debug("Initializing Azure Chat Client...")
        _chat_client = AsyncAzureOpenAI(api_key=Settings.AZURE_CHAT_API_KEY, api_version=Settings.AZURE_API_VERSION, azure_endpoint=Settings.AZURE_CHAT_ENDPOINT, timeout=Timeout(Settings.CLIENT_TIMEOUT), max_retries=Settings.CLIENT_MAX_RETRIES)
    return _chat_client

def get_embedding_client() -> AsyncAzureOpenAI:
    global _embedding_client
    if _embedding_client is None:
        log.debug("Initializing Azure Embedding Client...")
        _embedding_client = AsyncAzureOpenAI(api_key=Settings.AZURE_EMBEDDING_API_KEY, api_version=Settings.AZURE_API_VERSION, azure_endpoint=Settings.AZURE_EMBEDDING_ENDPOINT, timeout=Timeout(Settings.CLIENT_TIMEOUT), max_retries=Settings.CLIENT_MAX_RETRIES)
    return _embedding_client

async def a_chat(messages: List[Dict[str, Any]], model: str = Settings.AZURE_DEPLOYMENT, temp: float = 0.5, max_tokens: int = 1024) -> str:
    log.debug(f"Sending chat request to model '{model}' with {len(messages)} messages. Max tokens: {max_tokens}")
    client = get_chat_client()
    try:
        rsp = await client.chat.completions.create(model=model, temperature=temp, max_tokens=max_tokens, messages=messages)
        log.debug("Chat request successful.")
        return rsp.choices[0].message.content.strip()
    except Exception as e:
        log.error(f"Chat request failed: {e}")
        return f"Error: Could not get response from language model. {e}"

async def a_embed_batch(texts: List[str], model: str = Settings.AZURE_EMBEDDING_DEPLOYMENT) -> List[Optional[List[float]]]:
    if not texts: return []
    log.debug(f"Sending batch embedding request for {len(texts)} texts.")
    client = get_embedding_client()
    try:
        truncated_texts = [t[:Settings.MAX_EMBED_CHARS] for t in texts]
        rsp = await client.embeddings.create(model=model, input=truncated_texts)
        log.debug(f"Batch embedding request successful, received {len(rsp.data)} embeddings.")
        return [d.embedding for d in rsp.data]
    except Exception as e:
        log.error(f"Batch embedding request failed for {len(texts)} texts: {e}")
        return [None] * len(texts)

async def a_embed(text: str, model: str = Settings.AZURE_EMBEDDING_DEPLOYMENT) -> Optional[List[float]]:
    results = await a_embed_batch([text], model=model)
    return results[0] if results and results[0] is not None else None

# --------------------------------------------------------------------------- #
# 2.  Content Fetching & Parsing
# --------------------------------------------------------------------------- #
async def parse_pdf_with_gpt4o(pdf_bytes: bytes) -> str:
    log.info("Attempting PDF text extraction with multimodal model (e.g., GPT-4o).")
    try:
        base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
        messages = [{"role": "user","content": [dict(PROMPTS.PDF_OCR[0]), dict(PROMPTS.PDF_OCR[1])]}]
        messages[0]['content'][1]['image_url']['url'] = f"data:application/pdf;base64,{base64_pdf}"
        
        extracted_text = await a_chat(messages, model=Settings.AZURE_DEPLOYMENT, temp=0.05, max_tokens=4000)
        if "Error:" in extracted_text: log.error(f"Multimodal model failed to parse PDF: {extracted_text}"); return ""
        log.info(f"Successfully extracted {len(extracted_text)} characters from PDF using multimodal model.")
        return extracted_text
    except Exception as e:
        log.error(f"An unexpected error occurred during multimodal PDF parsing: {e}", exc_info=True)
        return ""

async def parse_pdf_bytes(pdf_bytes: bytes) -> str:
    fitz_text = ""
    try:
        def parse_fitz():
            text = ""
            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                for page in doc: text += page.get_text()
            return text
        fitz_text = await asyncio.to_thread(parse_fitz)
        log.debug(f"PyMuPDF successfully extracted {len(fitz_text)} characters.")
    except Exception as e:
        log.warning(f"PyMuPDF (fitz) failed to parse PDF. Error: {e}. Will attempt fallback.")
        fitz_text = ""
    if len(fitz_text) < Settings.PDF_MIN_TEXT_LENGTH_FALLBACK:
        if not fitz_text: log.info("PyMuPDF extracted no text, initiating multimodal model fallback.")
        else: log.info(f"PyMuPDF extracted only {len(fitz_text)} chars (threshold: {Settings.PDF_MIN_TEXT_LENGTH_FALLBACK}). This might be a scanned PDF. Initiating multimodal model fallback.")
        if "gpt-4o" not in Settings.AZURE_DEPLOYMENT.lower():
            log.error(f"Cannot use PDF parsing fallback. Your configured deployment '{Settings.AZURE_DEPLOYMENT}' does not appear to be 'gpt-4o'. Returning PyMuPDF result.")
            return fitz_text
        gpt4o_text = await parse_pdf_with_gpt4o(pdf_bytes)
        if len(gpt4o_text) > len(fitz_text) + 100:
            log.info("Multimodal model provided a substantially longer extraction. Using its result.")
            return gpt4o_text
        else:
            log.warning("Multimodal model did not provide a better result. Sticking with the original (short) PyMuPDF extraction.")
            return fitz_text
    return fitz_text

async def fetch_clean(url: str) -> str:
    if not url or CONTENT_CACHE[url]:
        if CONTENT_CACHE[url]: log.debug(f"Cache HIT for URL: {url[:80]}...")
        return CONTENT_CACHE[url] or ""
    log.debug(f"Cache MISS. Fetching URL: {url[:80]}...")
    TOUGH_DOMAINS = ['sciencedirect.com', 'onlinelibrary.wiley.com', 'mdpi.com', 'ieee.org', 'acs.org', 'researchgate.net', 'diamond.ac.uk']
    use_impersonation = any(domain in url for domain in TOUGH_DOMAINS)
    try:
        content = ""
        if use_impersonation:
            log.debug(f"Using impersonation (curl_cffi) for tough domain: {url[:80]}...")
            async with AsyncSession(impersonate="chrome110", timeout=30) as ses:
                resp = await ses.get(url)
                resp.raise_for_status()
                content_type = resp.headers.get('Content-Type', '').lower()
                if 'application/pdf' in content_type: content = await parse_pdf_bytes(resp.content)
                else: content = resp.text
        else:
            log.debug(f"Using standard fetch (aiohttp) for: {url[:80]}...")
            headers = {'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8','User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Gecko/20100101 Firefox/115.0'}
            async with aiohttp.ClientSession(headers=headers) as ses:
                async with ses.get(url, timeout=30, allow_redirects=True) as resp:
                    resp.raise_for_status()
                    content_type = resp.headers.get('Content-Type', '').lower()
                    if 'application/pdf' in content_type: content = await parse_pdf_bytes(await resp.read())
                    else: content = await resp.text()
        soup = BeautifulSoup(content, "html.parser")
        for bad in soup(["script", "style", "nav", "header", "footer", "aside", "form"]): bad.decompose()
        text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
        log.info(f"Successfully fetched and cleaned URL. Content length: {len(text)}. URL: {url[:80]}...")
        text = text[:Settings.MAX_PAGE_CHARS]
        CONTENT_CACHE.put(url, text)
        return text
    except Exception as e:
        log.warning(f"Fetch/Parse error for {url[:80]}... ({e})")
        return ""

async def searx_search(query: str, limit: int = Settings.SEARCH_RESULTS) -> List[Dict[str, str]]:
    url = Settings.SEARX_URL + aiohttp.helpers.quote(query) + "&format=json"
    log.debug(f"Sending search request to SearXNG for query: '{query}'")
    try:
        async with aiohttp.ClientSession() as ses:
            async with ses.get(url, timeout=20) as r:
                r.raise_for_status()
                j = await r.json()
                results = [{"title": res.get("title", ""), "url": res.get("url", ""), "snippet": res.get("content", "")} for res in (j.get("results") or [])[:limit]]
                log.debug(f"SearXNG returned {len(results)} results.")
                return results
    except Exception as e:
        log.error(f"SearXNG search failed for query '{query}': {e}")
        return []

# --------------------------------------------------------------------------- #
# 3.  Utilities
# --------------------------------------------------------------------------- #
def hash_txt(txt: str) -> str: return hashlib.sha1(txt.encode()).hexdigest()

def sentence_chunks(text: str) -> List[str]:
    sents = re.split(r"(?<=[.!?])\s+", text)
    chunks = [" ".join(sents[i:i+Settings.CHUNK_SENTENCES]) for i in range(0, len(sents), Settings.CHUNK_SENTENCES) if sents[i:i+Settings.CHUNK_SENTENCES]]
    return chunks

def cosine_similarity(a, b) -> float: 
    # Handles numpy arrays
    import numpy as np
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def extract_json_from_response(raw_response: str) -> Optional[str]:
    """Extracts a JSON string from an LLM response, handling markdown and other noise."""
    # 1. Check for ```json ... ``` markdown block
    markdown_match = re.search(r"```json\s*(.*?)\s*```", raw_response, re.DOTALL | re.IGNORECASE)
    if markdown_match:
        log.debug("Extracted JSON from markdown block.")
        return markdown_match.group(1).strip()

    # 2. If no markdown, assume the entire response is the JSON or contains it.
    potential_json_str = raw_response.strip()
    try:
        # Test if the stripped raw response is valid JSON
        json.loads(potential_json_str)
        log.debug("Identified raw LLM response as JSON.")
        return potential_json_str
    except json.JSONDecodeError:
        # 3. If direct parsing fails, try greedy regex for an embedded object
        log.debug("Raw LLM response is not direct JSON, attempting greedy regex.")
        object_match = re.search(r'(\{.*\})', raw_response, re.DOTALL) # Greedy
        if object_match:
            log.debug("Extracted JSON object using greedy regex.")
            return object_match.group(1).strip()

    log.error(f"Could not extract a JSON string from response. Raw: {raw_response}")
    return None