from pathlib import Path
from src.papertrail.ingestion import extract_pdf

def test_extract_pdf_preserves_page_metadata():
    pdf_path=Path("data/raw/sample_paper.pdf")

    pages=extract_pdf(pdf_path)

    assert len(pages) > 0
    assert pages[0]["page"] == 1
    assert pages[0]["paper_id"] == "sample_paper"
    assert pages[0]["source"] == "sample_paper.pdf"
    assert pages[0]["text"].strip()