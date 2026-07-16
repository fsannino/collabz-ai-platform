from rag_ingest.retrieval_policy import candidate_pool_size


def test_candidate_pool_uses_multiplier() -> None:
    assert candidate_pool_size(
        top_k=8,
        per_collection=5,
        multiplier=4,
        minimum=20,
        maximum=100,
    ) == 32


def test_candidate_pool_respects_minimum() -> None:
    assert candidate_pool_size(
        top_k=2,
        per_collection=5,
        multiplier=4,
        minimum=20,
        maximum=100,
    ) == 20


def test_candidate_pool_preserves_explicit_larger_request() -> None:
    assert candidate_pool_size(
        top_k=5,
        per_collection=40,
        multiplier=4,
        minimum=20,
        maximum=100,
    ) == 40


def test_candidate_pool_respects_maximum() -> None:
    assert candidate_pool_size(
        top_k=50,
        per_collection=5,
        multiplier=4,
        minimum=20,
        maximum=100,
    ) == 100


def test_candidate_pool_sanitizes_invalid_values() -> None:
    assert candidate_pool_size(
        top_k=0,
        per_collection=0,
        multiplier=0,
        minimum=0,
        maximum=0,
    ) == 1
