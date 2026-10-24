
from __future__ import annotations

import csv
import json
from pathlib import Path

from .bm25_retriever import build_bm25, load_chunks
from .embeddings import load_embedding_model
from .hybrid_retriever import build_dense_index, hybrid_search
from .reranker import load_reranker, rerank
from .rag_qa import create_client, generate_answer
from .answer_quality import evaluate_answer, extract_citations


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT / "eval" / "data" / "answer_quality_questions.json"
)
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"

CANDIDATE_K = 10
FINAL_K = 5


def load_questions() -> list[dict]:
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    if not isinstance(questions, list) or not questions:
        raise ValueError("Evaluation dataset must be a non-empty JSON list.")

    required = {"id", "question", "answerable", "expected_pages"}

    for item in questions:
        missing = required - item.keys()
        if missing:
            raise ValueError(
                f"Question {item.get('id', '<unknown>')} "
                f"is missing fields: {sorted(missing)}"
            )

        if not isinstance(item["answerable"], bool):
            raise ValueError(
                f"{item['id']}: answerable must be true or false."
            )

        if not item["question"].strip():
            raise ValueError(f"{item['id']}: question cannot be empty.")

    return questions


def run_evaluation() -> list[dict]:
    questions = load_questions()

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

    print("Connecting to Gemini...")
    client = create_client()

    results = []

    for item in questions:
        question_id = item["id"]
        question = item["question"]
        print(f"\nEvaluating {question_id}: {question}")

        try:
            candidates = hybrid_search(
                question,
                chunks,
                bm25,
                embedding_model,
                dense_index,
                candidate_k=CANDIDATE_K,
                top_k=CANDIDATE_K,
            )

            passages = rerank(
                question,
                candidates,
                reranker,
                top_k=FINAL_K,
            )

            answer = generate_answer(question, passages, client)

            quality = evaluate_answer(
                question=question,
                answer=answer,
                passages=passages,
                expected_answerable=item["answerable"],
            )

            citations = extract_citations(answer)
            expected_pages = set(item["expected_pages"])
            cited_pages = {citation["page"] for citation in citations}

            result = {
                "id": question_id,
                **quality,
                "expected_pages": sorted(expected_pages),
                "cited_expected_page": bool(
                    expected_pages & cited_pages
                ),
                "retrieved_passages": [
                    {
                        "source": passage.get("source"),
                        "page": passage.get("page"),
                        "chunk_id": passage.get("chunk_id"),
                        "text": passage.get("text"),
                    }
                    for passage in passages
                ],
                "error": None,
            }

            print(f"  Citations: {quality['citation_count']}")
            print(
                "  All citations match retrieved sources/pages: "
                f"{quality['all_citations_supported']}"
            )
            print(f"  Appears to abstain: {quality['appears_to_abstain']}")

        except Exception as error:
            result = {
                "id": question_id,
                "question": question,
                "expected_answerable": item["answerable"],
                "error": f"{type(error).__name__}: {error}",
            }
            print(f"  ERROR: {result['error']}")

        results.append(result)

    return results


def save_results(results: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    json_path = RESULTS_DIR / "answer_quality_results.json"
    csv_path = RESULTS_DIR / "answer_quality_results.csv"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    # Store nested evidence and lists as JSON strings in the CSV.
    flattened = []
    for result in results:
        flattened.append({
            key: (
                json.dumps(value, ensure_ascii=False)
                if isinstance(value, (dict, list))
                else value
            )
            for key, value in result.items()
        })

    if flattened:
        fieldnames = list(dict.fromkeys(
            key for row in flattened for key in row
        ))

        with csv_path.open(
            "w", encoding="utf-8", newline=""
        ) as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flattened)

    successful = [r for r in results if not r.get("error")]
    answerable = [
        r for r in successful if r["expected_answerable"]
    ]
    unanswerable = [
        r for r in successful if not r["expected_answerable"]
    ]

    print("\nANSWER-QUALITY CHECK SUMMARY")
    print(f"Questions processed: {len(results)}")
    print(f"Successful evaluations: {len(successful)}")
    print(f"Failed evaluations: {len(results) - len(successful)}")

    if successful:
        citation_rate = sum(
            r["all_citations_supported"] for r in successful
        ) / len(successful)

        print(
            "Answers whose citations match retrieved evidence: "
            f"{citation_rate:.1%}"
        )

    if answerable:
        abstentions = sum(
            r["appears_to_abstain"] for r in answerable
        )
        print(
            "Answerable questions that appear to abstain: "
            f"{abstentions}/{len(answerable)}"
        )

    if unanswerable:
        correct_abstentions = sum(
            r["abstention_correct"] is True for r in unanswerable
        )
        print(
            "Unanswerable questions with detected abstention: "
            f"{correct_abstentions}/{len(unanswerable)}"
        )

    print(f"\nJSON report: {json_path}")
    print(f"CSV report:  {csv_path}")


def main() -> None:
    results = run_evaluation()
    save_results(results)


if __name__ == "__main__":
    main()
