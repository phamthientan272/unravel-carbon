"""Knowledge base: chunking, PDF extraction, indexing and hybrid search, with a fake embedder (no model download)."""

import json

import pymupdf
import pytest

from src import kb


class FakeEmbedder:
    """Bag-of-letters embedding: deterministic and good enough to rank by shared words."""

    def __init__(self, *_a, **_k):
        pass

    def __call__(self, input):  # noqa: A002 (chroma's signature)
        return [[float(t.lower().count(c)) for c in "abcdefghijklmnopqrstuvwxyz"] or [1.0] for t in input]

    @staticmethod
    def name():
        return "fake"

    def get_config(self):
        return {}

    @staticmethod
    def build_from_config(_config):
        return FakeEmbedder()

    def embed_query(self, input):  # noqa: A002
        return self(input)

    def is_legacy(self):
        return False


@pytest.fixture
def chroma(tmp_path, monkeypatch):
    monkeypatch.setattr(kb, "CHROMA_DIR", tmp_path / "chroma")
    monkeypatch.setattr(kb, "DefaultEmbeddingFunction", FakeEmbedder)
    kb.get_kb.cache_clear()
    yield
    kb.get_kb.cache_clear()


SOURCE = "ghg-protocol-revised.pdf"
LONG = " ".join(f"Sentence number {i} about scope two electricity emissions." for i in range(80))


def test_clean_joins_hyphenated_breaks_and_collapses_whitespace():
    assert kb._clean("emis-\nsions   are\n\n\nhigh") == "emissions are\nhigh"


def test_chunk_page_splits_long_pages_with_overlap_and_skips_tiny_ones():
    chunks = kb.chunk_page(LONG, SOURCE, 7)
    assert len(chunks) > 1 and all(c.page == 7 and c.source == SOURCE for c in chunks)
    assert chunks[0].id == f"{SOURCE}:p7:0" and chunks[1].id.endswith(":1")
    assert all(len(c.text) <= kb.CHUNK_CHARS for c in chunks)
    assert kb.chunk_page("too short", SOURCE, 1) == []
    one = kb.chunk_page("x" * 200, SOURCE, 1)  # no break point: hard cut at the limit still terminates
    assert len(one) == 1
    unbroken = kb.chunk_page("y" * (kb.CHUNK_CHARS * 2), SOURCE, 1)
    assert len(unbroken) >= 2


def test_extract_pages_reads_a_pdf(tmp_path, monkeypatch):
    pdf = tmp_path / "tiny.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "hello page one")
    doc.new_page().insert_text((72, 72), "hello page two")
    doc.save(pdf)
    doc.close()
    monkeypatch.setattr(kb, "DATA_DIR", tmp_path)
    pages = kb.extract_pages("tiny.pdf")
    assert [p for p, _ in pages] == [1, 2] and "page two" in pages[1][1]


def test_extract_pages_uses_the_ocr_cache_for_peer2(tmp_path, monkeypatch):
    monkeypatch.setattr(kb, "OCR_CACHE", tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError, match="ocr_peer2"):
        kb.extract_pages("peer2_emissions_report.pdf")
    cache = tmp_path / "ocr.json"
    cache.write_text(json.dumps({"pages": {"3": "ocr text"}}))
    monkeypatch.setattr(kb, "OCR_CACHE", cache)
    assert kb.extract_pages("peer2_emissions_report.pdf") == [(3, "ocr text")]


def _fake_pages(monkeypatch):
    texts = {
        "ghg-protocol-revised.pdf": [(1, "Scope two covers purchased electricity and the location based method. " * 3),
                                      (2, "Scope three covers value chain emissions such as business travel. " * 3)],
        "peer1_emissions_report.pdf": [(5, "Docusign reports renewable energy purchases and total emissions. " * 3)],
        "peer2_emissions_report.pdf": [(9, "Shopify reports carbon removal and scope one totals in tables. " * 3)],
    }
    monkeypatch.setattr(kb, "extract_pages", lambda source: texts[source])


def test_build_chunks_and_index_and_search(chroma, monkeypatch):
    _fake_pages(monkeypatch)
    assert not kb.index_exists()
    n = kb.build_index()
    assert n == len(kb.build_chunks()) == 4 and kb.index_exists()
    assert kb.build_index(reset=True) == 4  # rebuilding drops the old collection first

    base = kb.get_kb()
    hits = base.search("purchased electricity location based", k=2)
    assert hits[0]["page"] == 1 and {"source", "title", "page", "score", "text"} <= set(hits[0])
    only = base.search("emissions", k=5, source="peer1_emissions_report.pdf")
    assert only and {h["source"] for h in only} == {"peer1_emissions_report.pdf"}
    assert kb.get_kb() is base  # cached
