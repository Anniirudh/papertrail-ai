from pathlib import Path
import pymupdf

def extract_pdf(pdf_path:str |Path)->list[dict]:
    pdf_path=Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found :{pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Input file must be a PDF")

    pages=[]

    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            text=page.get_text("text").strip()

            if not text:
                continue

            pages.append(
                {
                    "paper_id":pdf_path.stem,
                    "source":pdf_path.name,
                    "page":page_number,
                    "text":text,
                }
            )
        if not pages:
            raise ValueError(
                "No extractable text found. "
                "The PDF may be scanned and require OCR."
            )
    return pages

if __name__=='__main__':
    pdf_file=Path("data/raw/sample_paper.pdf")
    extracted_pages=extract_pdf(pdf_file)
    print(f"Extracted {len(extracted_pages)} pages with text.")

    for page in extracted_pages[:2]:
        print(f"\n--- Page {page['page']} ---")
        print(page["text"][:1000])