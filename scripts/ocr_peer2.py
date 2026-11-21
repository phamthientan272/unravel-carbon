"""One-time OCR of the peer 2 report (Shopify) with a multimodal Gemini model.

The PDF's prose is text but its emissions tables are images (the numbers are missing from a plain text
extraction), so each page is rendered to PNG and transcribed. The result is
cached in data/derived/peer2_ocr.json, so ingestion never calls the API again.

    uv run python scripts/ocr_peer2.py                   # macOS Vision (local, free, default on macOS)
    uv run python scripts/ocr_peer2.py --engine gemini   # Gemini vision (needs GEMINI_API_KEY)

Vision returns text lines without table structure, so lines are regrouped into rows by their
vertical position and cells on a row are joined with " | ". Gemini returns Markdown tables directly.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pymupdf
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.kb import OCR_CACHE  # noqa: E402

PDF = ROOT / "data" / "peer2_emissions_report.pdf"
PROMPT = (
    "Transcribe all text on this sustainability report page as plain text, in reading order. "
    "Render tables as Markdown tables keeping every number exactly as printed, including units and years. "
    "For charts, transcribe titles, axis labels and data labels. Do not add commentary."
)


def render_pages(dpi: int) -> dict[int, bytes]:
    with pymupdf.open(PDF) as doc:
        return {i: page.get_pixmap(dpi=dpi).tobytes("png") for i, page in enumerate(doc, start=1)}


def vision_ocr(png: bytes, row_tol: float = 0.006, cell_chars: int = 40) -> str:
    """OCR one page image with macOS Vision and rebuild table rows.

    Vision's own line order is kept for prose (it follows columns). Vision emits a table column by
    column, so each run of consecutive short lines (`cell_chars`) is re-sorted top to bottom, grouped into
    rows (vertical centres within `row_tol` of the page height) and joined left to right with " | ".
    Long lines (prose) break a run and are never merged.
    """
    from ocrmac import ocrmac
    from PIL import Image

    img = Image.open(io.BytesIO(png))
    # annotation = (text, confidence, (x, y, w, h)), normalised, origin at the bottom-left
    items = [(t, b[0], 1 - (b[1] + b[3] / 2)) for t, _c, b in
             ocrmac.OCR(img, recognition_level="accurate", language_preference=["en-US"]).recognize()]

    def rows_of(run: list[tuple[str, float, float]]) -> list[str]:
        rows: list[list[tuple[str, float, float]]] = []
        for it in sorted(run, key=lambda it: it[2]):
            if rows and abs(it[2] - rows[-1][-1][2]) <= row_tol:
                rows[-1].append(it)
            else:
                rows.append([it])
        return [" | ".join(t for t, _x, _y in sorted(r, key=lambda it: it[1])) for r in rows]

    lines: list[str] = []
    run: list[tuple[str, float, float]] = []
    for it in items:
        if len(it[0]) <= cell_chars:
            run.append(it)
        else:
            lines += rows_of(run) + [it[0]]
            run = []
    return "\n".join(lines + rows_of(run))


def run_vision(images: dict[int, bytes]) -> tuple[dict[str, str], str]:
    pages = {}
    for i in images:  # Vision is fast and uses the system's own threads, so no pool
        pages[str(i)] = vision_ocr(images[i])
        print(f"page {i}: {len(pages[str(i)])} chars", flush=True)
    return pages, "macOS Vision OCR (ocrmac, rows regrouped by y-position)"


def run_gemini(images: dict[int, bytes]) -> tuple[dict[str, str], str]:
    from google import genai
    from google.genai import types

    # Explicit timeout + retries: without them a single stuck request blocks the whole run.
    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"],
        http_options=types.HttpOptions(timeout=90_000, retry_options=types.HttpRetryOptions(attempts=4)),
    )
    model = os.getenv("OCR_MODEL", "gemini-4-hydrogen")

    def ocr(i: int) -> tuple[int, str]:
        resp = client.models.generate_content(
            model=model, contents=[types.Part.from_bytes(data=images[i], mime_type="image/png"), PROMPT]
        )
        print(f"page {i}: {len(resp.text or '')} chars", flush=True)
        return i, resp.text or ""

    with ThreadPoolExecutor(max_workers=4) as pool:
        pages = {str(i): text for i, text in sorted(pool.map(ocr, images))}
    return pages, f"gemini vision ({model})"


def main() -> None:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=["vision", "gemini"], default="vision" if sys.platform == "darwin" else "gemini")
    ap.add_argument("--dpi", type=int, default=200, help="render resolution (Gemini was tuned at 150)")
    args = ap.parse_args()
    if args.engine == "vision" and sys.platform != "darwin":
        sys.exit("macOS Vision only runs on macOS: use --engine gemini")

    images = render_pages(args.dpi)
    pages, method = (run_vision if args.engine == "vision" else run_gemini)(images)
    OCR_CACHE.write_text(json.dumps({"source": PDF.name, "method": method, "pages": pages}, indent=1))
    print(f"wrote {OCR_CACHE} ({method})")


if __name__ == "__main__":
    main()
