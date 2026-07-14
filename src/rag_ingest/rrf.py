"""Reciprocal Rank Fusion para combinar rankings heterogêneos."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from rag_ingest.models import RetrievedChunk


def chunk_key(chunk: RetrievedChunk) -> tuple[str, str, int | None]:
    return (
        chunk.collection_key,
        chunk.normalized_source,
        chunk.metadata.get("chunk_index"),
    )


def reciprocal_rank_fusion(
    rankings: Iterable[list[RetrievedChunk]],
    *,
    k: int = 60,
    limit: int | None = None,
) -> list[tuple[RetrievedChunk, float]]:
    """Combina rankings; maior score RRF é melhor."""
    offset = max(1, int(k))
    scores: dict[tuple[str, str, int | None], float] = defaultdict(float)
    chunks_by_key: dict[tuple[str, str, int | None], RetrievedChunk] = {}

    for ranking in rankings:
        for position, chunk in enumerate(ranking, start=1):
            key = chunk_key(chunk)
            chunks_by_key.setdefault(key, chunk)
            scores[key] += 1.0 / (offset + position)

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if limit is not None:
        ordered = ordered[: max(0, limit)]
    return [(chunks_by_key[key], score) for key, score in ordered]
