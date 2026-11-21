"""Knowledge base over the PDFs: Chroma (vector) + BM25 (keyword), merged with reciprocal rank fusion.

Only unstructured text is indexed here. Tabular data goes through the analytics/SQL tools.
Every chunk carries `source` and `page` metadata so answers can cite pages.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache


import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from rank_bm25 import BM25Okapi

from src.data import DATA_DIR, ROOT

CHROMA_DIR = ROOT / ".chroma"
COLLECTION = "emissions_kb"
OCR_CACHE = DATA_DIR / "derived" / "peer2_ocr.json"

DOCUMENTS = {
    "ghg-protocol-revised.pdf": "GHG Protocol Corporate Accounting and Reporting Standard (Revised Edition)",
    "peer1_emissions_report.pdf": "Peer 1: Docusign FY25 Impact Report",
    "peer2_emissions_report.pdf": "Peer 2: Shopify 2024 Climate Report",
}

CHUNK_CHARS = 1500
OVERLAP_CHARS = 250


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    title: str
    page: int


def _clean(text: str) -> str:
    text = re.sub(r"-\n(\w)", r"\1", text)  # re-join hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{2,}", "\n", text).strip()


def chunk_page(text: str, source: str, page: int) -> list[Chunk]:
    """Split one page into overlapping chunks on paragraph/sentence boundaries. Chunks never span pages."""
    text = _clean(text)
    chunks, start, i = [], 0, 0
    while start < len(text):
        end = min(start + CHUNK_CHARS, len(text))
        if end < len(text):  # prefer to break at a paragraph or sentence end
            cut = max(text.rfind("\n", start + CHUNK_CHARS // 2, end), text.rfind(". ", start + CHUNK_CHARS // 2, end))
            end = cut + 1 if cut > 0 else end
        body = text[start:end].strip()
        if len(body) > 80:
            chunks.append(Chunk(f"{source}:p{page}:{i}", body, source, DOCUMENTS[source], page))
            i += 1
        if end >= len(text):
            break
        start = max(end - OVERLAP_CHARS, start + 1)
    return chunks


def extract_pages(source: str) -> list[tuple[int, str]]:
    """Return (page_number, text) for a PDF. peer2 uses the cached OCR output (its tables are images)."""
    if source == "peer2_emissions_report.pdf":
        if not OCR_CACHE.exists():
            raise FileNotFoundError(f"{OCR_CACHE} missing: run `python scripts/ocr_peer2.py` first (needs GEMINI_API_KEY).")
        return [(int(p), t) for p, t in json.loads(OCR_CACHE.read_text())["pages"].items()]
    import pymupdf

    with pymupdf.open(DATA_DIR / source) as doc:
        return [(i + 1, page.get_text()) for i, page in enumerate(doc)]


def build_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []
    for source in DOCUMENTS:
        for page, text in extract_pages(source):
            chunks.extend(chunk_page(text, source, page))
    return chunks


def build_index(reset: bool = True) -> int:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if reset and COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)
    col = client.get_or_create_collection(COLLECTION, embedding_function=DefaultEmbeddingFunction(),
                                          metadata={"hnsw:space": "cosine"})
    chunks = build_chunks()
    for i in range(0, len(chunks), 128):
        batch = chunks[i:i + 128]
        col.add(ids=[c.id for c in batch], documents=[c.text for c in batch],
                metadatas=[{"source": c.source, "title": c.title, "page": c.page} for c in batch])
    return len(chunks)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())


class KnowledgeBase:
    def __init__(self) -> None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.col = client.get_collection(COLLECTION, embedding_function=DefaultEmbeddingFunction())
        everything = self.col.get(include=["documents", "metadatas"])
        self.ids: list[str] = everything["ids"]
        self.docs = dict(zip(self.ids, everything["documents"]))
        self.meta = dict(zip(self.ids, everything["metadatas"]))
        self.bm25 = BM25Okapi([_tokens(self.docs[i]) for i in self.ids])

    def search(self, query: str, k: int = 6, source: str | None = None, rrf_k: int = 60) -> list[dict]:
        """Hybrid search: vector + BM25 ranks fused with reciprocal rank fusion."""
        where = {"source": source} if source else None
        n = min(30, len(self.ids))
        vec = self.col.query(query_texts=[query], n_results=n, where=where)["ids"][0]

        scores = self.bm25.get_scores(_tokens(query))
        ranked = sorted(range(len(self.ids)), key=lambda i: -scores[i])
        kw = [self.ids[i] for i in ranked if (not source or self.meta[self.ids[i]]["source"] == source)][:n]

        fused: dict[str, float] = {}
        for ranking in (vec, kw):
            for rank, cid in enumerate(ranking):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
        top = sorted(fused, key=fused.get, reverse=True)[:k]
        return [{"source": self.meta[c]["source"], "title": self.meta[c]["title"], "page": self.meta[c]["page"],
                 "score": round(fused[c], 4), "text": self.docs[c]} for c in top]


    def page(self, source: str, page: int, max_chunks: int = 1) -> list[dict]:
        """The first chunk(s) of one page, for citing a page the quality catalog points to."""
        ids = [i for i in self.ids if self.meta[i]["source"] == source and self.meta[i]["page"] == page]
        ids.sort(key=lambda i: int(i.rsplit(":", 1)[1]))
        return [{"source": source, "title": self.meta[i]["title"], "page": page, "score": 1.0, "text": self.docs[i]}
                for i in ids[:max_chunks]]


@lru_cache(maxsize=1)
def get_kb() -> KnowledgeBase:
    return KnowledgeBase()


def index_exists() -> bool:
    try:
        return chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(COLLECTION).count() > 0
    except Exception:
        return False


if __name__ == "__main__":
    print(f"Indexed {build_index()} chunks into {CHROMA_DIR}")
