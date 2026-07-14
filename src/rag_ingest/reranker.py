"""Filtragem, deduplicação e diversidade dos resultados recuperados."""

from __future__ import annotations

import hashlib
from collections import defaultdict

from rag_ingest.models import RetrievedChunk


class DiversityReranker:
    def __init__(
        self,
        max_chunks_per_source: int = 1,
        max_distance: float | None = None,
    ) -> None:
        self.max_chunks_per_source = max(1, max_chunks_per_source)
        self.max_distance = max_distance

    def rank(
        self,
        chunks: list[RetrievedChunk],
        limit: int,
    ) -> list[RetrievedChunk]:
        selected: list[RetrievedChunk] = []
        source_counts: dict[str, int] = defaultdict(int)
        seen_content: set[str] = set()

        for chunk in sorted(chunks, key=lambda item: item.distance):
            if self.max_distance is not None and chunk.distance > self.max_distance:
                continue

            fingerprint = self._fingerprint(chunk.document)
            if fingerprint in seen_content:
                continue

            source_key = chunk.normalized_source
            if source_counts[source_key] >= self.max_chunks_per_source:
                continue

            seen_content.add(fingerprint)
            source_counts[source_key] += 1
            selected.append(chunk)

            if len(selected) >= limit:
                break

        return selected

    @staticmethod
    def _fingerprint(text: str) -> str:
        normalized = " ".join(text.lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
