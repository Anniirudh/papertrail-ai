
from src.papertrail.answer_quality import (
    extract_citations,
    citation_is_supported,
    check_citations,
    appears_to_abstain,
    evaluate_answer,
)


def test_extract_citations():
    answer = "The experiment reports this result [sample_paper.pdf, p. 7]."

    assert extract_citations(answer) == [
        {"source": "sample_paper.pdf", "page": 7}
    ]


def test_citation_matches_retrieved_evidence():
    passages = [
        {
            "source": "sample_paper.pdf",
            "page": 7,
            "chunk_id": "chunk_12",
            "text": "Example evidence.",
        }
    ]

    assert citation_is_supported(
        {"source": "sample_paper.pdf", "page": 7},
        passages,
    )


def test_wrong_page_is_not_supported():
    passages = [{"source": "sample_paper.pdf", "page": 7}]

    assert not citation_is_supported(
        {"source": "sample_paper.pdf", "page": 3},
        passages,
    )


def test_missing_citation_is_detected():
    result = check_citations("The paper reports a result.", [])

    assert result["has_citation"] is False
    assert result["citation_count"] == 0


def test_explicit_abstention_is_detected():
    assert appears_to_abstain(
        "I could not find relevant evidence in the indexed paper."
    )


def test_unanswerable_question_requires_abstention():
    result = evaluate_answer(
        question="Who will win the next UEFA Champions League?",
        answer="I could not find relevant evidence in the indexed paper.",
        passages=[],
        expected_answerable=False,
    )

    assert result["abstention_correct"] is True
