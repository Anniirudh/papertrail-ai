from __future__ import annotations

import re
from pathlib import Path

CITATION_PATTERN=re.compile(
    r"\[([^,\]]+),\s*p\.\s*(\d+)\]",
    re.IGNORECASE,
)

ABSTENTION_PHRASES = (
    "i could not find relevant evidence",
    "the paper does not provide",
    "the provided evidence does not",
    "i cannot determine from the paper",
    "there is insufficient evidence",
    "the evidence is insufficient",
)

def extract_citations(answer:str)->list[dict]:
    citations=[]
    for match in CITATION_PATTERN.finditer(answer):
        citations.append({
            "source":match.group(1).strip(),
            "page":int(match.group(2)),
        })
    return citations

def citation_is_supported(
        citation:str,
        passages:list[dict],
)->bool:
    cited_source=Path(citation["source"]).name.casefold()
    cited_pages=citation["page"]

    return any(
        Path(str(passage.get("source",""))).name.casefold()
        ==cited_source
        and int(passage.get("page",-1))==cited_pages
        for passage in passages
    )

def check_citations(
        answer:str,
        passages:list[dict],
)->dict:
    citations=extract_citations(answer)

    if not citations:
        return {
            "citation_count": 0,
            "has_citation": False,
            "all_citations_supported": False,
            "supported_citation_count": 0,
        }

    supported_count=sum(
        citation_is_supported(citation,passages)
        for citation in citations
    )

    return {
        "citation_count": len(citations),
        "has_citation": True,
        "all_citations_supported": supported_count == len(citations),
        "supported_citation_count": supported_count,
    }

def appears_to_abstain(answer: str) -> bool:
    """Heuristic only; this does not establish semantic abstention."""
    normalized = answer.casefold()
    return any(
        phrase in normalized
        for phrase in ABSTENTION_PHRASES
    )


def evaluate_answer(
        question:str,
        answer:str,
        passages:list[dict],
        expected_answerable:bool,
)->dict:
    citation_result = check_citations(answer, passages)
    abstained = appears_to_abstain(answer)

    if expected_answerable:
        abstention_correct = None
    else:
        abstention_correct = abstained

    return {
        "question": question,
        "answer": answer,
        "expected_answerable": expected_answerable,
        "appears_to_abstain": abstained,
        "abstention_correct": abstention_correct,
        **citation_result,
    }