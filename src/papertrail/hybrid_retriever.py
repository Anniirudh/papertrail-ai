
import numpy as np
import faiss

from .bm25_retriever import load_chunks, build_bm25, retrieve
from .embeddings import load_embedding_model, generate_embeddings
from .reranker import load_reranker,rerank


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k: int = 60,
    source_names: list[str] | None = None,
) -> list[dict]:
    """Combine ranked retrieval results using Reciprocal Rank Fusion."""

    if k <= 0:
        raise ValueError("RRF k must be positive.")

    if source_names is None:
        source_names = [
            f"retriever_{i + 1}" for i in range(len(ranked_lists))
        ]

    if len(source_names) != len(ranked_lists):
        raise ValueError(
            "source_names must match the number of ranked lists."
        )

    fused = {}

    for source_name, ranked_list in zip(source_names, ranked_lists):
        seen_in_list = set()

        for rank, item in enumerate(ranked_list, start=1):
            chunk_id = item["chunk_id"]

            # Avoid counting duplicate chunks from the same retriever.
            if chunk_id in seen_in_list:
                continue

            seen_in_list.add(chunk_id)

            if chunk_id not in fused:
                fused[chunk_id] = {
                    "chunk": item.copy(),
                    "rrf_score": 0.0,
                    "retrieval_sources": [],
                }

            fused[chunk_id]["rrf_score"] += 1.0 / (k + rank)

            if source_name not in fused[chunk_id]["retrieval_sources"]:
                fused[chunk_id]["retrieval_sources"].append(source_name)

    results = sorted(
        fused.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )

    output = []

    for item in results:
        chunk = item["chunk"].copy()
        chunk["rrf_score"] = item["rrf_score"]
        chunk["retrieval_sources"] = item["retrieval_sources"]
        output.append(chunk)

    return output


def build_dense_index(chunks: list[dict], model):
    """Build a FAISS index using normalized embeddings."""

    if not chunks:
        raise ValueError("Cannot build a dense index with no chunks.")

    texts = [chunk["text"] for chunk in chunks]

    embeddings = generate_embeddings(texts, model)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
        raise ValueError("Invalid embedding dimensions or chunk count.")

    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index


def dense_search(
    query: str,
    chunks: list[dict],
    model,
    index,
    top_k: int = 10,
) -> list[dict]:
    """Retrieve semantically similar chunks using FAISS."""

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    if not chunks:
        return []

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    query_embedding = np.asarray(query_embedding, dtype=np.float32)

    limit = min(top_k, len(chunks))
    scores, indices = index.search(query_embedding, limit)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        item = chunks[int(idx)].copy()
        item["dense_score"] = float(score)
        results.append(item)

    return results


def hybrid_search(
    query: str,
    chunks: list[dict],
    bm25,
    model,
    index,
    candidate_k: int = 10,
    top_k: int = 5,
) -> list[dict]:
    """Combine BM25 and dense retrieval results."""

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if candidate_k <= 0 or top_k <= 0:
        raise ValueError("candidate_k and top_k must be positive.")

    lexical_results = retrieve(
        query,
        chunks,
        bm25,
        top_k=candidate_k,
    )

    semantic_results = dense_search(
        query,
        chunks,
        model,
        index,
        top_k=candidate_k,
    )

    fused_results = reciprocal_rank_fusion(
        [lexical_results, semantic_results],
        source_names=["bm25", "dense"],
    )

    return fused_results[:top_k]


def main():
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks.")

    if not chunks:
        raise ValueError("No chunks found. Check your processed data file.")

    print("Building BM25 index...")
    bm25 = build_bm25(chunks)

    print("Loading embedding model...")
    model = load_embedding_model()

    print("Loading cross-encoder reranker...")
    reranker=load_reranker()

    print("Building FAISS dense index...")
    index = build_dense_index(chunks, model)

    print("Hybrid retrieval ready.")

    while True:
        query = input("\nQuestion (type 'exit' to quit): ").strip()

        if query.lower() in {"exit", "quit"}:
            print("Exiting hybrid retriever.")
            break

        if not query:
            print("Please enter a question.")
            continue

        try:
            candidates = hybrid_search(
                query,
                chunks,
                bm25,
                model,
                index,
                candidate_k=10,
                top_k=10,
            )

            results=rerank(
                query,
                candidates,
                reranker,
                top_k=5,
            )

            if not results:
                print("No relevant chunks found.")
                continue

            for rank, result in enumerate(results, start=1):
                print(f"\n--- Hybrid result {rank} ---")
                print(f"RRF score: {result['rrf_score']:.6f}")
                print(f"Retrieved by: {', '.join(result['retrieval_sources'])}")
                print(f"Source: {result.get('source', 'Unknown')}")
                print(f"Page: {result.get('page', 'Unknown')}")
                print(f"Chunk: {result['chunk_id']}")
                print(f"Text: {result['text'][:700]}")

        except ValueError as error:
            print(f"Invalid query: {error}")


if __name__ == "__main__":
    main()
