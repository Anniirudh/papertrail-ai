from __future__ import annotations

import re
from pathlib import Path


CITATION_PATTERN = re.compile(
    r"\[([^,\]]+),\s*p\.\s*(\d+)\]",
    re.IGNORECASE,
)

ABSTENTION_PHRASES = (
    "i could not find relevant evidence",
    "the paper does not provide",
    "the provided evidence does not",
    "the provided paper evidence is insufficient",
    "the evidence is insufficient",
    "there is insufficient evidence",
    "i cannot determine from the paper",
    "i cannot answer this question from the paper",
    "the paper does not contain information",
    "insufficient information to answer",
    "not enough information to answer",
    "cannot be determined from the provided evidence",
)


def extract_citations(answer: str) -> list[dict]:
    return [
        {
            "source": match.group(1).strip(),
            "page": int(match.group(2)),
        }
        for match in CITATION_PATTERN.finditer(answer)
    ]


def citation_is_supported(
    citation: dict,
    passages: list[dict],
) -> bool:
    cited_source = Path(citation["source"]).name.casefold()

    for passage in passages:
        passage_source = Path(
            str(passage.get("source", ""))
        ).name.casefold()

        try:
            passage_page = int(passage.get("page", -1))
        except (TypeError, ValueError):
            continue

        if (
            passage_source == cited_source
            and passage_page == citation["page"]
        ):
            return True

    return False


def check_citations(answer: str, passages: list[dict], abstained: bool = False) -> dict:
    citations = extract_citations(answer)

    supported_count = sum(
        citation_is_supported(citation, passages)
        for citation in citations
    )
    all_citations_match = (
        bool(citations) and supported_count == len(citations)
    )

    if abstained:
        return {
            "citation_count": len(citations),
            "has_citation": bool(citations),
            "all_citations_supported": all_citations_match,
            "all_citations_match_retrieved_passages": all_citations_match,
            "citation_check_applicable": False,
            "citation_review": "not_applicable_abstention",
        }

    return {
        "citation_count": len(citations),
        "has_citation": bool(citations),
        "all_citations_supported": all_citations_match,
        "all_citations_match_retrieved_passages": all_citations_match,
        "citation_check_applicable": True,
        "citation_review": "pending_manual_claim_support_review",
    }

def appears_to_abstain(answer: str) -> bool:
    normalized = answer.casefold()
    return any(
        phrase in normalized
        for phrase in ABSTENTION_PHRASES
    )


def normalize_text(text: str) -> str:
    return " ".join(
        re.findall(r"[a-z0-9]+", text.casefold())
    )


def reference_answer_coverage(
    answer: str,
    reference_answer: str,
) -> dict:
    """
    Measures reference-token coverage, not semantic correctness.
    Ignore very short tokens to reduce noise.
    """
    reference_tokens = set(
        token
        for token in normalize_text(reference_answer).split()
        if len(token) > 2
    )

    answer_tokens = set(
        normalize_text(answer).split()
    )

    if not reference_tokens:
        return {
            "reference_coverage": None,
            "reference_tokens_missing": [],
            "reference_check_status": "missing_reference_answer",
        }

    missing = sorted(reference_tokens - answer_tokens)
    coverage = (
        len(reference_tokens - set(missing))
        / len(reference_tokens)
    )

    return {
        "reference_coverage": round(coverage, 4),
        "reference_tokens_missing": missing,
        "reference_check_status": "automated_screen_only",
    }


def evaluate_answer(
    question: str,
    answer: str,
    passages: list[dict],
    expected_answerable: bool,
    reference_answer: str = "",
) -> dict:
    abstained = appears_to_abstain(answer)

    if expected_answerable:
        abstention_correct = not abstained
    else:
        abstention_correct = abstained

    citation_result = check_citations(
        answer,
        passages,
        abstained=abstained ,
    )

    coverage_result = reference_answer_coverage(
        answer,
        reference_answer,
    ) if expected_answerable else {
        "reference_coverage": None,
        "reference_tokens_missing": [],
        "reference_check_status": "not_applicable_unanswerable",
    }

    return {
        "question": question,
        "answer": answer,
        "expected_answerable": expected_answerable,
        "appears_to_abstain": abstained,
        "abstention_correct": abstention_correct,
        **citation_result,
        **coverage_result,
    }
