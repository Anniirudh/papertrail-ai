
from src.papertrail.baseline_comparison import distinct_pages


def test_distinct_pages_removes_duplicates_in_rank_order():
    chunks = [
        {"page": 3},
        {"page": 3},
        {"page": 7},
        {"page": 2},
        {"page": 7},
    ]

    assert distinct_pages(chunks) == [3, 7, 2]


def test_distinct_pages_handles_empty_results():
    assert distinct_pages([]) == []
