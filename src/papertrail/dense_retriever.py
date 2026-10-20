import json
from pathlib import Path
import faiss
import numpy as np
from .embeddings import load_embedding_model, generate_embeddings

CHUNKS_PATH=Path("data/processed/sample_paper_chunks.jsonl")
INDEX_PATH=Path("artifacts/papertrail.faiss")

def load_chunks(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Chunks not founds at {path}"
        )

    with path.open("r",encoding="utf-8") as file:
        chunks=[json.loads(line) for line in file if line.strip()]
    if not chunks:
        raise ValueError("The chunk file is empty.")

    return chunks

def build_index(chunks: list[dict],model):
    texts=[chunk["text"] for chunk in chunks]
    embeddings=generate_embeddings(texts,model)
    embeddings=np.asarray(embeddings,dtype=np.float32)
    index=faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

def search(
        query:str,
        chunks:list[dict],
        model,
        index,
        top_k:int =5,
)-> list[dict]:
    if not query.strip():
        raise ValueError("Query cannot be empty")

    if top_k<=0:
        raise ValueError("top_k must be positive")
    if index.ntotal==0:
        return []

    
    query_vector=generate_embeddings([query],model)
    query_vector=np.asarray(query_vector,dtype=np.float32)

    k=min(top_k,index.ntotal)
    scores,indices=index.search(query_vector,k)

    results=[]

    for score,idx in zip(scores[0],indices[0]):
        if idx<0:
            continue

        result=chunks[int(idx)].copy()
        result["similarity_score"]=float(score)
        results.append(result)


    return results

def main():
    chunks=load_chunks(CHUNKS_PATH)
    model=load_embedding_model()

    print(f"Loaded {len(chunks)} chunks.")
    index=build_index(chunks,model)

    INDEX_PATH.parent.mkdir(parents=True,exist_ok=True)
    faiss.write_index(index,str(INDEX_PATH))

    print(f"FAISS index saved to {INDEX_PATH}")
    print("Enter a research question or type 'exit' to quit.")
    while True:
        query=input("\nQuestion: ").strip()

        if query.lower()=="exit":
            break
        try:
            results=search(query,chunks,model,index,top_k=5)

            for rank,result in enumerate(results,start=1):
                print(f"\n--- Result {rank} ---")
                print(
                    f"Source: {result['source']} | "
                    f"Page: {result['page']}"
                )
                print(
                    f"Similarity: "
                    f"{result['similarity_score']:.4f}"
                )
                print(result["text"][:700])

        except ValueError as error:
            print(f"Invalid query: {error}")


if __name__ == "__main__":
    main()
