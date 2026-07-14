"""Filtragem, deduplicação, diversidade e relevância lexical."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict

from rag_ingest.models import RetrievedChunk


_STOPWORDS = {
    "a", "as", "o", "os", "de", "da", "das", "do", "dos", "e", "em",
    "um", "uma", "para", "por", "com", "que", "quais", "qual", "liste",
    "cite", "tres", "três", "mencionada", "mencionadas", "mencionado",
    "mencionados", "explicitamente", "nao", "não",
}


class DiversityReranker:
    def __init__(
        self,
        max_chunks_per_source: int = 1,
        max_distance: float | None = None,
        lexical_weight: float = 250.0,
    ) -> None:
        self.max_chunks_per_source = max(1, max_chunks_per_source)
        self.max_distance = max_distance
        self.lexical_weight = max(0.0, lexical_weight)

    def rank(
        self,
        chunks: list[RetrievedChunk],
        limit: int,
        question: str = "",
    ) -> list[RetrievedChunk]:
        selected: list[RetrievedChunk] = []
        source_counts: dict[str, int] = defaultdict(int)
        seen_content: set[str] = set()
        query_terms = self._terms(question)

        ranked = sorted(
            chunks,
            key=lambda item: self._score(item, query_terms),
        )

        for chunk in ranked:
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

    def _score(self, chunk: RetrievedChunk, query_terms: set[str]) -> float:
        """Menor é melhor: distância vetorial menos bônus lexical."""
        if not query_terms:
            return chunk.distance

        searchable = " ".join(
            [
                chunk.document,
                chunk.source,
                " ".join(str(value) for value in chunk.metadata.values()),
            ]
        )
        document_terms = self._terms(searchable)
        overlap = len(query_terms & document_terms)
        coverage = overlap / max(1, len(query_terms))
        return chunk.distance - (coverage * self.lexical_weight)

    @staticmethod
    def _terms(text: str) -> set[str]:
        normalized = unicodedata.normalize("NFKD", text.lower())
        ascii_text = "".join(
            character for character in normalized
            if not unicodedata.combining(character)
        )
        tokens = set(re.findall(r"[a-z0-9_]{3,}", ascii_text))
        return {token for token in tokens if token not in _STOPWORDS}

    @staticmethod
    def _fingerprint(text: str) -> str:
        normalized = " ".join(text.lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
