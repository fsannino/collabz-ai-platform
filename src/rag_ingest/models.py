"""Modelos compartilhados do núcleo RAG."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RetrievedChunk:
    collection_key: str
    distance: float
    document: str
    source: str
    metadata: dict[str, Any]

    @property
    def normalized_source(self) -> str:
        return self.source.strip().lower()


@dataclass(frozen=True)
class RagAnswer:
    answer: str
    model: str
    chunks: tuple[RetrievedChunk, ...]
