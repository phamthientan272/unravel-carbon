"""One-time OCR of the image-only peer 2 report (Shopify) with a multimodal Gemini model.

The PDF has no text layer, so each page is rendered to PNG and transcribed. The result is
cached in data/derived/peer2_ocr.json, so ingestion never calls the API again.

    uv run python scripts/ocr_peer2.py
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pymupdf
from dotenv import load_dotenv
from google import genai
from google.genai import types

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.kb import OCR_CACHE  # noqa: E402

PDF = ROOT / "data" / "peer2_emissions_report.pdf"
PROMPT = (
    "Transcribe all text on this sustainability report page as plain text, in reading order. "
    "Render tables as Markdown tables keeping every number exactly as printed, including units and years. "
    "For charts, transcribe titles, axis labels and data labels. Do not add commentary."
)


def main() -> None:
    load_dotenv(ROOT / ".env")
    # Explicit timeout + retries: without them a single stuck request blocks the whole run.
    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"],
        http_options=types.HttpOptions(timeout=90_000, retry_options=types.HttpRetryOptions(attempts=4)),
    )
    model = os.getenv("OCR_MODEL", "gemini-4-hydrogen")
    with pymupdf.open(PDF) as doc:
        images = {i: page.get_pixmap(dpi=150).tobytes("png") for i, page in enumerate(doc, start=1)}

    def ocr(i: int) -> tuple[int, str]:
        resp = client.models.generate_content(
            model=model, contents=[types.Part.from_bytes(data=images[i], mime_type="image/png"), PROMPT]
        )
        print(f"page {i}: {len(resp.text or '')} chars", flush=True)
        return i, resp.text or ""

    with ThreadPoolExecutor(max_workers=4) as pool:
        pages = {str(i): text for i, text in sorted(pool.map(ocr, images))}
    OCR_CACHE.write_text(json.dumps({"source": PDF.name, "method": f"gemini vision ({model})", "pages": pages}, indent=1))
    print(f"wrote {OCR_CACHE}")


if __name__ == "__main__":
    main()
