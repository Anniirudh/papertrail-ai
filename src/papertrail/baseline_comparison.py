from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean

from .bm25_retriever import build_bm25, load_chunks, retrieve
from .dense_retriever import build_index, search
from .embeddings import load_embedding_model
from .hybrid_retriever import build_dense_index, hybrid_search
from .reranker import load_reranker, rerank
from .evaluation import recall_at_k, reciprocal_rank, ndcg_at_k, load_dataset

PROJECT_ROOT=Path(__file__).resolve().parents[2]
DATASET_PATH=PROJECT_ROOT/"eval"/"data"/"sample_paper_questions.json"
RESULTS_DIR=PROJECT_ROOT/"eval"/"results"

K=5
CANDIDATE_K=20

def distinct_pages(ranked_chunks:list[dict])->list[int]:
    return list(
        dict.fromkeys(int(chunk["page"]) for chunk in ranked_chunks)
    )

def run_comparison()->list[dict]:
    dataset=load_dataset(DATASET_PATH)

    print("Loadinf paper chunks...")
    chunks=load_chunks()
    print(f"Loaded {len(chunks)} chunks.")

    print("Building BM25 index...")
    bm25=build_bm25(chunks)

    print("Loading embedding model...")
    embedding_model=load_embedding_model()

    print("Building dense index...")
    dense_index = build_index(chunks, embedding_model)

    print("Building hybrid dense index...")
    hybrid_dense_index = build_dense_index(chunks, embedding_model)

    print("Loading reranker...")
    reranker = load_reranker()

    results = []

    for item in dataset:
        question=item["question"]
        relevant_pages=item["relevant_pages"]

        print(f"Evaluating {item["id"]}...")

        bm25_ranked=retrieve(
            question,chunks,bm25,top_k=CANDIDATE_K
        )

        dense_ranked = search(
            question,
            chunks,
            embedding_model,
            dense_index,
            top_k=CANDIDATE_K,
        )

        hybrid_ranked = hybrid_search(
            question,
            chunks,
            bm25,
            embedding_model,
            hybrid_dense_index,
            candidate_k=CANDIDATE_K,
            top_k=CANDIDATE_K,
        )

        hybrid_reranked = rerank(
            question,
            hybrid_ranked,
            reranker,
            top_k=CANDIDATE_K,
        )

        configurations={
            "bm25":bm25_ranked,
            "dense":dense_ranked,
            "hybrid":hybrid_ranked,
            "hybrid_reranked":hybrid_reranked,
        }

        row={
            "id":item["id"],
            "question":question,
            "answerable":item["answerable"],
            "relevant_pages":relevant_pages,
        }

        for name, ranked_chunks in configurations.items():
            pages=distinct_pages(ranked_chunks)
            row[f"{name}_pages_at_5"]=pages[:K]
            row[f"{name}_chunks_ids_at_5"]=[
                chunk["chunk_id"] for chunk in ranked_chunks[:K]
            ]

            if item["answerable"]:
                row[f"{name}_recall_at_5"]=recall_at_k(
                    pages,relevant_pages,K
                )
                row[f"{name}_mrr"] = reciprocal_rank(
                    pages[:K], relevant_pages
                )
                row[f"{name}_ndcg_at_5"] = ndcg_at_k(
                    pages, relevant_pages, K
                )
            else:
                row[f"{name}_recall_at_5"] = None
                row[f"{name}_mrr"] = None
                row[f"{name}_ndcg_at_5"] = None

        results.append(row)
    return results

def save_comparison(results: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    json_path = RESULTS_DIR / "baseline_comparison.json"
    csv_path = RESULTS_DIR / "baseline_comparison.csv"
    summary_path = RESULTS_DIR / "baseline_summary.csv"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    answerable = [row for row in results if row["answerable"]]
    methods = ["bm25", "dense", "hybrid", "hybrid_reranked"]

    summary = []
    for method in methods:
        summary.append({
            "method": method,
            "answerable_questions": len(answerable),
            "recall_at_5": mean(
                row[f"{method}_recall_at_5"] for row in answerable
            ),
            "mrr": mean(row[f"{method}_mrr"] for row in answerable),
            "ndcg_at_5": mean(
                row[f"{method}_ndcg_at_5"] for row in answerable
            ),
        })

    with summary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    print("\nBASELINE COMPARISON (page-level metrics)")
    print(f"{'Method':<20} {'Recall@5':>10} {'MRR':>10} {'nDCG@5':>10}")
    print("-" * 54)

    for row in summary:
        print(
            f"{row['method']:<20} "
            f"{row['recall_at_5']:>10.3f} "
            f"{row['mrr']:>10.3f} "
            f"{row['ndcg_at_5']:>10.3f}"
        )

    print(f"\nDetailed JSON: {json_path}")
    print(f"Detailed CSV:  {csv_path}")
    print(f"Summary CSV:   {summary_path}")


def main() -> None:
    results = run_comparison()
    save_comparison(results)


if __name__ == "__main__":
    main()
