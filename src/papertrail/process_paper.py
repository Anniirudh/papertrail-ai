import json
from pathlib import Path

from .ingestion import extract_pdf
from .chunking import chunk_pages

def main():
    pdf_path=Path("data/raw/sample_paper.pdf")
    output_path=Path("data/processed/sample_paper_chunks.jsonl")

    pages=extract_pdf(pdf_path)

    chunks=chunk_pages(
        pages,
        chunk_size=500,
        overlap=80,
    )

    output_path.parent.mkdir(parents=True,exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Pages extracted: {len(pages)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Output saved to: {output_path}")

    for chunk in chunks[:3]:
        print(f"\n--- {chunk['chunk_id']} ---")
        print(f"Page: {chunk['page']}")
        print(f"Word count: {len(chunk['text'].split())}")
        print(chunk["text"][:300])


if __name__ == "__main__":
    main()
