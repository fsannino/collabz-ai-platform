"""Ranking lexical BM25 aplicado aos candidatos recuperados."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

from rag_ingest.models import RetrievedChunk


_STOPWORDS = {
    "a", "as", "o", "os", "de", "da", "das", "do", "dos", "e", "em",
    "um", "uma", "para", "por", "com", "que", "qual", "quais", "sao",
    "são", "foi", "foram", "ser", "seja", "no", "na", "nos", "nas",
}


def tokenize(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKD", text.lower())
    ascii_text = "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    )
    return [
        token
        for token in re.findall(r"[a-z0-9_]{2,}", ascii_text)
        if token not in _STOPWORDS
    ]


class BM25Ranker:
    """BM25 sem dependências externas, adequado ao conjunto de candidatos."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = max(0.01, float(k1))
        self.b = min(1.0, max(0.0, float(b)))

    def score_all(
        self,
        question: str,
        chunks: list[RetrievedChunk],
    ) -> list[tuple[RetrievedChunk, float]]:
        if not chunks:
            return []

        query_terms = tokenize(question)
        documents = [
            tokenize(
                " ".join(
                    [
                        chunk.document,
                        chunk.source,
                        " ".join(str(value) for value in chunk.metadata.values()),
                    ]
                )
            )
            for chunk in chunks
        ]
        average_length = sum(len(document) for document in documents) / max(1, len(documents))
        document_frequency = Counter(
            term
            for document in documents
            for term in set(document)
        )
        document_count = len(documents)

        scored: list[tuple[RetrievedChunk, float]] = []
        for chunk, document in zip(chunks, documents, strict=False):
            frequencies = Counter(document)
            document_length = len(document)
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if frequency == 0:
                    continue
                df = document_frequency.get(term, 0)
                idf = math.log(1.0 + (document_count - df + 0.5) / (df + 0.5))
                denominator = frequency + self.k1 * (
                    1.0 - self.b
                    + self.b * document_length / max(1.0, average_length)
                )
                score += idf * (frequency * (self.k1 + 1.0)) / denominator
            scored.append((chunk, score))

        return sorted(scored, key=lambda item: item[1], reverse=True)

    def rank(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        limit: int | None = None,
    ) -> list[RetrievedChunk]:
        ranked = [chunk for chunk, _ in self.score_all(question, chunks)]
        return ranked if limit is None else ranked[: max(0, limit)]
