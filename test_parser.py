import asyncio
from pathlib import Path

# Important: This imports the functions from your main script
from research_engine_R1_01 import _parse_pdf_bytes, fetch_clean, log

# We need a dummy URL to use the cache key mechanism in fetch_clean
# The file path will be used to read the bytes directly.
TEXT_PDF_PATH = Path("./test_assets/text_based.pdf")
SCANNED_PDF_PATH = Path("./test_assets/scanned.pdf")
HTML_TEST_URL = "https://www.w3.org/TR/html52/" # A simple HTML page for regression testing

async def test_pdf_parsing(pdf_path: Path):
    """Reads a local PDF and runs it through the parsing logic."""
    log.info(f"--- TESTING PDF: {pdf_path.name} ---")
    if not pdf_path.exists():
        log.error(f"Test file not found: {pdf_path.resolve()}")
        return

    # Read the local PDF file as bytes
    pdf_bytes = pdf_path.read_bytes()

    # Call the core parsing function directly
    extracted_text = await _parse_pdf_bytes(pdf_bytes)

    print("\n--- EXTRACTED TEXT (first 500 chars) ---")
    print(extracted_text[:500])
    print("-------------------------------------------\n")

async def test_html_parsing(url: str):
    """Tests the standard HTML fetching to ensure it wasn't broken."""
    log.info(f"--- TESTING HTML URL: {url} ---")
    
    # Use fetch_clean which handles both HTML and remote PDFs
    extracted_text = await fetch_clean(url)

    print("\n--- EXTRACTED TEXT (first 500 chars) ---")
    print(extracted_text[:500])
    print("-------------------------------------------\n")


async def main():
    # --- Test Case 1: Standard Text-Based PDF ---
    # EXPECTED: PyMuPDF succeeds, no fallback to GPT-4o.
    await test_pdf_parsing(TEXT_PDF_PATH)

    # --- Test Case 2: Scanned/Image-Based PDF ---
    # EXPECTED: PyMuPDF extracts little/no text, triggers the GPT-4o fallback.
    await test_pdf_parsing(SCANNED_PDF_PATH)
    
    # --- Test Case 3: Standard HTML Page (Regression Test) ---
    # EXPECTED: BeautifulSoup succeeds, no PDF logic is triggered.
    await test_html_parsing(HTML_TEST_URL)


if __name__ == "__main__":
    asyncio.run(main())