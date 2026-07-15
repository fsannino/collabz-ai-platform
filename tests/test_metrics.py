from rag_ingest.metrics import ndcg_at_k, recall_at_k, reciprocal_rank


def test_recall_at_k_matches_partial_sources() -> None:
    ranked = ["/docs/acmp.pdf", "/docs/pmi.pdf"]
    assert recall_at_k(ranked, ["acmp.pdf"], 1) == 1.0
    assert recall_at_k(ranked, ["acmp.pdf", "ccmp.pdf"], 2) == 0.5


def test_reciprocal_rank_uses_first_relevant_result() -> None:
    ranked = ["irrelevante.pdf", "acmp.pdf", "pmi.pdf"]
    assert reciprocal_rank(ranked, ["acmp.pdf"]) == 0.5


def test_ndcg_at_k_rewards_early_relevance() -> None:
    early = ndcg_at_k(["acmp.pdf", "x.pdf"], ["acmp.pdf"], 2)
    late = ndcg_at_k(["x.pdf", "acmp.pdf"], ["acmp.pdf"], 2)
    assert early > late


def test_ndcg_at_k_does_not_count_duplicate_matches_twice() -> None:
    score = ndcg_at_k(
        ["acmp.pdf", "acmp-copia.pdf", "x.pdf"],
        ["acmp"],
        3,
    )
    assert score == 1.0


def test_ndcg_at_k_never_exceeds_one() -> None:
    score = ndcg_at_k(
        ["acmp.pdf", "acmp-copia.pdf", "ccmp.pdf"],
        ["acmp", "ccmp"],
        3,
    )
    assert 0.0 <= score <= 1.0
