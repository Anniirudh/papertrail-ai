
import pytest

from src.papertrail.evaluation import (
    load_dataset,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k_finds_relevant_page():
    assert recall_at_k([3, 7, 8], [7, 8], 3) == 1.0


def test_recall_at_k_is_zero_when_not_found():
    assert recall_at_k([1, 2, 3], [8], 3) == 0.0


def test_recall_at_k_only_considers_first_k():
    assert recall_at_k([1, 2, 8], [8], 2) == 0.0


def test_reciprocal_rank_first_result():
    assert reciprocal_rank([8, 7, 4], [7, 8]) == 1.0


def test_reciprocal_rank_second_result():
    assert reciprocal_rank([4, 7, 8], [7, 8]) == 0.5


def test_reciprocal_rank_no_match():
    assert reciprocal_rank([1, 2, 3], [8]) == 0.0


def test_ndcg_perfect_ranking():
    assert ndcg_at_k([7, 8, 2], [7, 8], 3) == pytest.approx(1.0)


def test_ndcg_relevant_page_ranked_lower():
    score = ndcg_at_k([1, 7, 2], [7], 3)
    assert 0.0 < score < 1.0


def test_metrics_reject_invalid_k():
    with pytest.raises(ValueError):
        recall_at_k([1], [1], 0)

    with pytest.raises(ValueError):
        ndcg_at_k([1], [1], 0)


def test_dataset_loads():
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    dataset_path = (
        project_root / "eval" / "data" / "sample_paper_questions.json"
    )

    dataset = load_dataset(dataset_path)

    assert len(dataset) == 8
    assert any(item["answerable"] is False for item in dataset)
