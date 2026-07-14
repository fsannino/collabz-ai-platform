"""Métricas simples para avaliação de recuperação."""

from __future__ import annotations

import math


def recall_at_k(ranked_sources: list[str], expected_sources: list[str], k: int) -> float:
    if not expected_sources:
        return 0.0
    expected = {item.lower() for item in expected_sources}
    retrieved = [item.lower() for item in ranked_sources[:k]]
    hits = sum(any(target in source for source in retrieved) for target in expected)
    return hits / len(expected)


def reciprocal_rank(ranked_sources: list[str], expected_sources: list[str]) -> float:
    expected = [item.lower() for item in expected_sources]
    for rank, source in enumerate(ranked_sources, start=1):
        lowered = source.lower()
        if any(target in lowered for target in expected):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_sources: list[str], expected_sources: list[str], k: int) -> float:
    expected = [item.lower() for item in expected_sources]
    gains = []
    for source in ranked_sources[:k]:
        lowered = source.lower()
        gains.append(1.0 if any(target in lowered for target in expected) else 0.0)
    dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
    ideal_hits = min(len(expected), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(index + 2) for index in range(ideal_hits))
    return dcg / idcg
