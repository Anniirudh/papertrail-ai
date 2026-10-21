import json
import re
from pathlib import Path
from rank_bm25 import BM25Okapi

CHUNKS_PATH=Path("data/processed/sample_paper_chunks.jsonl")

def tokenize(text:str)->list[str]:
    return re.findall(r"\b\w+\b",text.lower())

def load_chunks(path:Path=CHUNKS_PATH)->list[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {path}. "
            "Run python -m src.papertrail.process_paper first."
        )
    chunks=[]

    with path.open("r",encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))

    if not chunks:
        raise ValueError("No chunks found in the JSONL file. ")

    return chunks 

def build_bm25(chunks:list[dict])->BM25Okapi:
    corpus=[tokenize(chunk["text"]) for chunk in chunks]

    if not any(corpus):
        raise ValueError("No search text found in the chunks.")

    return BM25Okapi(corpus)

def retrieve(
        query:str,
        chunks:list[dict],
        bm25:BM25Okapi,
        top_k:int=5,
)->list[dict]:
    if not query.strip():
        raise ValueError("Query cannot be empty.")
    if top_k<=0:
        raise ValueError("top_k must be positive.")

    query_tokens=tokenize(query)
    if not query_tokens:
        raise ValueError("Query contains no searchable words.")

    scores=bm25.get_scores(query_tokens)

    ranked_indices=sorted(
        range(len(scores)),
        key=lambda i:scores[i],
        reverse=True,
    )

    results=[]

    for index in ranked_indices[:min(top_k,len(chunks))]:
        result=chunks[index].copy()
        result["bm25_score"]=float(scores[index])
        results.append(result)

    return results

def main():
    chunks=load_chunks()
    bm25=build_bm25(chunks)

    print(f"Loaded {len(chunks)} chunks.")
    print("BM25 index built successfully.")

    while True:
        query=input("\n Enter a question or 'exit' to quit").strip()
        if query.lower() in {'exit','quit'}:
            break

        try:
            results=retrieve(query,chunks,bm25,top_k=5)

            for rank,result in enumerate(results,start=1):
                print(f"\n--- Result {rank} ---")
                print(f"BM25 score: {result['bm25_score']:.4f}")
                print(f"Source: {result['source']}")
                print(f"Page: {result['page']}")
                print(f"Chunk ID: {result['chunk_id']}")
                print(result["text"][:700])

        except ValueError as error:
            print(f"Invalid query: {error}")


if __name__ == "__main__":
    main()