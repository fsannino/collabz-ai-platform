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
    "mencionados", "explicitamente", "nao", "não", "nome", "apareca",
    "apareça", "literalmente", "trecho", "trechos", "cujo", "cujos",
}

_ASSOCIATION_ANCHORS = {
    "association", "associacao", "associacoes", "confederacao",
    "confederation", "federacao", "federation", "society", "sociedade",
    "institute", "instituto", "council", "conselho", "chapter", "capitulo",
    "foundation", "fundacao", "guild", "ordem", "pmi", "acmp", "ccmp",
}

_CANONICAL_TERMS = {
    "associacoes": "association",
    "associacao": "association",
    "associations": "association",
    "organizacoes": "organization",
    "organizacao": "organization",
    "organizations": "organization",
    "instituicoes": "institute",
    "instituicao": "institute",
    "institutions": "institute",
}


class DiversityReranker:
    def __init__(
        self,
        max_chunks_per_source: int = 1,
        max_distance: float | None = None,
        lexical_weight: float = 250.0,
        near_duplicate_threshold: float = 0.88,
    ) -> None:
        self.max_chunks_per_source = max(1, max_chunks_per_source)
        self.max_distance = max_distance
        self.lexical_weight = max(0.0, lexical_weight)
        self.near_duplicate_threshold = min(
            1.0,
            max(0.0, near_duplicate_threshold),
        )

    def rank(
        self,
        chunks: list[RetrievedChunk],
        limit: int,
        question: str = "",
    ) -> list[RetrievedChunk]:
        selected: list[RetrievedChunk] = []
        selected_terms: list[set[str]] = []
        source_counts: dict[str, int] = defaultdict(int)
        seen_content: set[str] = set()

        ranked = sorted(
            chunks,
            key=lambda item: self.score(item, question),
        )

        for chunk in ranked:
            if self.max_distance is not None and chunk.distance > self.max_distance:
                continue

            fingerprint = self._fingerprint(chunk.document)
            if fingerprint in seen_content:
                continue

            terms = self._terms(chunk.document)
            if self._is_near_duplicate(terms, selected_terms):
                continue

            source_key = chunk.normalized_source
            if source_counts[source_key] >= self.max_chunks_per_source:
                continue

            seen_content.add(fingerprint)
            selected_terms.append(terms)
            source_counts[source_key] += 1
            selected.append(chunk)

            if len(selected) >= limit:
                break

        return selected

    def score(self, chunk: RetrievedChunk, question: str = "") -> float:
        """Retorna o score final do reranker; quanto menor, melhor."""
        query_terms = self._terms(question)
        association_intent = bool(query_terms & {"association", "organization"})
        return self._score(
            chunk,
            query_terms=query_terms,
            association_intent=association_intent,
        )

    def _score(
        self,
        chunk: RetrievedChunk,
        query_terms: set[str],
        association_intent: bool,
    ) -> float:
        """Menor é melhor: distância vetorial menos bônus lexical e de intenção."""
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
        score = chunk.distance - (coverage * self.lexical_weight)

        if association_intent:
            anchor_overlap = len(document_terms & _ASSOCIATION_ANCHORS)
            if anchor_overlap:
                score -= min(anchor_overlap, 3) * (self.lexical_weight * 0.75)
            else:
                score += self.lexical_weight * 0.5

        return score

    def _is_near_duplicate(
        self,
        terms: set[str],
        selected_terms: list[set[str]],
    ) -> bool:
        if not terms or self.near_duplicate_threshold >= 1.0:
            return False

        for existing in selected_terms:
            union = terms | existing
            if not union:
                continue
            similarity = len(terms & existing) / len(union)
            if similarity >= self.near_duplicate_threshold:
                return True
        return False

    @staticmethod
    def _terms(text: str) -> set[str]:
        normalized = unicodedata.normalize("NFKD", text.lower())
        ascii_text = "".join(
            character for character in normalized
            if not unicodedata.combining(character)
        )
        tokens = set(re.findall(r"[a-z0-9_]{3,}", ascii_text))
        canonical = {
            _CANONICAL_TERMS.get(token, token)
            for token in tokens
            if token not in _STOPWORDS
        }
        return canonical

    @staticmethod
    def _fingerprint(text: str) -> str:
        normalized = " ".join(text.lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
