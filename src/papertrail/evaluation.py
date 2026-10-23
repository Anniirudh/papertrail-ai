from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean

from .bm25_retriever import build_bm25,load_chunks
from .embeddings import load_embedding_model
from .hybrid_retriever import build_dense_index, hybrid_search
from .reranker import load_reranker,rerank

PROJECT_ROOT=Path(__file__).resolve().parents[2]
DATASET_PATH=PROJECT_ROOT/"eval"/"data"/"sample_paper_questions.json"
RESULTS_DIR=PROJECT_ROOT/"eval"/"results"

def recall_at_k(retrieved:list[int],relevant:list[int],k:int)->float:
    if k <= 0:
        raise ValueError("k must be greater than 0")

    if not relevant:
        return 0.0

    retrieved_at_k = retrieved[:k]
    relevant_retrieved = set(retrieved_at_k) & set(relevant)

    return len(relevant_retrieved) / len(set(relevant))

def reciprocal_rank(retrieved:list[int],relevant:list[int])->float:
    relevant_set = set(relevant)

    for rank, page_id in enumerate(retrieved, start=1):
        if page_id in relevant_set:
            return 1.0 / rank

    return 0.0

def ndcg_at_k(retrieved_pages:list[int],relevant_pages:list[int],k:int)->float:
    if k<1:
        raise ValueError("k must be >=1")

    if not relevant_pages:
        return 0.0

    relevant=set(relevant_pages)
    dcg=sum(
        1.0/math.log2(rank+1)
        for rank,page in enumerate(retrieved_pages[:k],start=1)
        if page in relevant
    )

    ideal_hits=min(len(relevant),k)
    idcg=sum(
        1.0/math.log2(rank+1)
        for rank in range(1,ideal_hits+1)
    )
    return dcg/idcg if idcg else 0.0

def load_dataset(path:Path)->list[dict]:
    with path.open("r",encoding="utf-8") as file:
        dataset=json.load(file)

    if not isinstance(dataset,list) or not dataset:
        raise ValueError("Evaluation dataset must be a non empty JSON list")

    seen_ids=set()

    for item in dataset:
        for key in ("id","question","relevant_pages","answerable"):
            if key not in item:
                raise ValueError(f"Dataset entry is missing '{key}'.")

        if item["id"] in seen_ids:
            raise ValueError(f"Duplicate evaluation ID: {item['id']}")
        seen_ids.add(item["id"])

        if not isinstance(item["question"],str) or not item["question"].strip():
            raise ValueError(f"{item["id"]}: question cannot be empty")
        if not isinstance(item["relevant_pages"],list):
            raise ValueError(f"{item["id"]}: relevant_pages must be a list")

        if any(
            not isinstance(page,int) or isinstance(page,bool) or page<1
            for page in item["relevant_pages"]
        ):
            raise ValueError(f"{item["id"]}: page numbers must be a positive integers")

        if not isinstance(item["answerable"],bool):
            raise ValueError(f"{item['id']}: answerable must be a boolean")

        if item["answerable"] and not item["relevant_pages"]:
            raise ValueError(
                f"{item['id']}: answerable questions need relevant pages."
            )

        if not item["answerable"] and item["relevant_pages"]:
            raise ValueError(
                f"{item['id']}: unanswerable questions must have no relevant pages."
            )

    return dataset


def evaluate_retrieval(
    dataset: list[dict],
    chunks: list[dict],
    bm25,
    embedding_model,
    dense_index,
    reranker,
    k: int = 5,
) -> list[dict]:
    results = []

    for item in dataset:
        candidates = hybrid_search(
            item["question"],
            chunks,
            bm25,
            embedding_model,
            dense_index,
            candidate_k=20,
            top_k=20,
        )

        ranked = rerank(
            item["question"],
            candidates,
            reranker,
            top_k=20,
        )

        retrieved_pages = list(
            dict.fromkeys(int(chunk["page"]) for chunk in ranked)
        )
        relevant_pages = item["relevant_pages"]

        results.append({
            "id": item["id"],
            "question": item["question"],
            "answerable": item["answerable"],
            "relevant_pages": relevant_pages,
            "retrieved_pages": retrieved_pages[:k],
            "retrieved_chunk_ids": [
                chunk["chunk_id"] for chunk in ranked[:k]
            ],
            "recall_at_k": (
                recall_at_k(retrieved_pages, relevant_pages, k)
                if item["answerable"] else None
            ),
            "reciprocal_rank": (
                reciprocal_rank(retrieved_pages, relevant_pages)
                if item["answerable"] else None
            ),
            "ndcg_at_k": (
                ndcg_at_k(retrieved_pages, relevant_pages, k)
                if item["answerable"] else None
            ),
        })

    return results


def save_results(results: list[dict], k: int) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    json_path = RESULTS_DIR / "sample_paper_results.json"
    csv_path = RESULTS_DIR / "sample_paper_results.csv"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    answerable = [row for row in results if row["answerable"]]

    print(f"\nRetrieval evaluation (k={k})")
    print(f"Answerable questions: {len(answerable)}")
    print(f"Recall@{k}: {mean(r['recall_at_k'] for r in answerable):.3f}")
    print(f"MRR: {mean(r['reciprocal_rank'] for r in answerable):.3f}")
    print(f"nDCG@{k}: {mean(r['ndcg_at_k'] for r in answerable):.3f}")
    print(f"\nJSON report: {json_path}")
    print(f"CSV report:  {csv_path}")


def main() -> None:
    k = 5
    dataset = load_dataset(DATASET_PATH)

    print("Loading paper chunks...")
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks.")

    print("Building BM25 index...")
    bm25 = build_bm25(chunks)

    print("Loading embedding model...")
    embedding_model = load_embedding_model()

    print("Building dense index...")
    dense_index = build_dense_index(chunks, embedding_model)

    print("Loading reranker...")
    reranker = load_reranker()

    print("Evaluating retrieval...")
    results = evaluate_retrieval(
        dataset,
        chunks,
        bm25,
        embedding_model,
        dense_index,
        reranker,
        k=k,
    )

    save_results(results, k)


if __name__ == "__main__":
    main()

