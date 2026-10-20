
import re
import unicodedata


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_pages(
    pages: list[dict],
    chunk_size: int = 500,
    overlap: int = 80,
) -> list[dict]:
    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Overlap must be >= 0 and smaller than chunk_size.")

    chunks = []
    step = chunk_size - overlap

    for page_record in pages:
        text = clean_text(page_record["text"])
        words = text.split()

        if not words:
            continue

        for start in range(0, len(words), step):
            end = min(start + chunk_size, len(words))

            chunks.append({
                "paper_id": page_record["paper_id"],
                "source": page_record["source"],
                "page": page_record["page"],
                "chunk_id": (
                    f"{page_record['paper_id']}"
                    f"_p{page_record['page']}"
                    f"_c{start // step + 1}"
                ),
                "start_word": start,
                "end_word": end,
                "text": " ".join(words[start:end]),
            })

            if end == len(words):
                break

    return chunks
