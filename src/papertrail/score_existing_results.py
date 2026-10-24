import json
from pathlib import Path

from .answer_quality import evaluate_answer

ROOT=Path(__file__).resolve().parents[2]

DATASET_PATH = (
    ROOT / "eval" / "data" / "answer_quality_questions.json"
)
RESULTS_PATH = (
    ROOT / "eval" / "results" / "answer_quality_results.json"
)
OUTPUT_PATH = (
    ROOT / "eval" / "results" / "answer_quality_scored.json"
)

def main():
    with DATASET_PATH.open(encoding="utf-8") as f:
        questions=json.load(f)

    with RESULTS_PATH.open(encoding="utf-8") as f:
        existing_results=json.load(f)

    question_map={item["id"]: item for item in questions}
    scored_results=[]

    for result in existing_results:
        question_id=result.get("id")
        question=question_map.get(question_id)

        if "answer" not in result:
            print(f"\nMissing answer for question ID: {question_id}")
            print("Available keys:", list(result.keys()))
            print("Result:", result)
            continue

        if question is None:
            print(f"Skipping unknown question ID: {question_id}")
            continue
        scored=evaluate_answer(
            question=result["question"],
            
            answer=result["answer"],
            passages=result.get("retrieved_passages", []),
            expected_answerable=question["answerable"],
            reference_answer=question.get("reference_answer", ""),
        )

        scored_results.append({
            **result,
            **scored,
            "id": question_id,
            "expected_pages": question.get("expected_pages", []),
            "manual_factual_review": "pending",
        })
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(scored_results, f, indent=2, ensure_ascii=False)

    successful = [
        row for row in scored_results
        if not row.get("error")
    ]

    answerable = [
        row for row in successful
        if row["expected_answerable"]
    ]

    unanswerable = [
        row for row in successful
        if not row["expected_answerable"]
    ]

    scored = [
        row for row in answerable
        if row["reference_coverage"] is not None
    ]

    print("\nREFERENCE-ANSWER EVALUATION")
    print(f"Successful answers: {len(successful)}")
    print(
        f"Answerable questions with reference answers: "
        f"{len(scored)}/{len(answerable)}"
    )

    if scored:
        mean_coverage = sum(
            row["reference_coverage"] for row in scored
        ) / len(scored)
        print(f"Mean reference-token coverage: {mean_coverage:.1%}")

    print("\nABSTENTION EVALUATION")
    if successful:
        correct = sum(
            row["abstention_correct"] is True
            for row in successful
        )
        print(f"Correct answer/abstention behavior: {correct}/{len(successful)}")

    print("\nCITATION EVALUATION")
    citation_checked = [
        row for row in successful
        if row.get("citation_check_applicable")
    ]

    if citation_checked:
        citation_rate = sum(
            row["all_citations_match_retrieved_passages"]
            for row in citation_checked
        ) / len(citation_checked)

        print(
            "Answers with citations matching retrieved passages: "
            f"{citation_rate:.1%}"
        )

    print("\nManual claim-support review is still required.")
    print(f"Scored report: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
