
from src.papertrail.chunking import clean_text, chunk_pages


def test_clean_text_normalizes_ligatures():
    assert clean_text("scientiﬁc") == "scientific"


def test_chunking_preserves_page_metadata():
    pages = [
        {
            "paper_id": "demo",
            "source": "demo.pdf",
            "page": 2,
            "text": " ".join(["research"] * 1200),
        }
    ]

    chunks = chunk_pages(pages, chunk_size=500, overlap=80)

    assert len(chunks) == 3
    assert all(chunk["page"] == 2 for chunk in chunks)
    assert all(chunk["paper_id"] == "demo" for chunk in chunks)
    assert all(len(chunk["text"].split()) <= 500 for chunk in chunks)


def test_invalid_overlap_is_rejected():
    try:
        chunk_pages([], chunk_size=100, overlap=100)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for invalid overlap")
