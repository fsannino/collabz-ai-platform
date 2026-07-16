"""Recuperação híbrida para o fluxo principal do assistente."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from rag_ingest.config import CollectionConfig, Settings
from rag_ingest.lexical_index import LexicalHit, SQLiteLexicalIndex
from rag_ingest.metadata_filter import MetadataFilter
from rag_ingest.models import RetrievedChunk
from rag_ingest.retriever import VectorRetriever
from rag_ingest.rrf import reciprocal_rank_fusion


class _VectorSearch(Protocol):
    def search(
        self,
        question: str,
        collections: list[CollectionConfig],
        candidates_per_collection: int,
        metadata_filter: MetadataFilter | None = None,
    ) -> list[RetrievedChunk]: ...


class _LexicalSearch(Protocol):
    def search(
        self,
        query: str,
        collection_keys: list[str],
        limit: int = 20,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> list[LexicalHit]: ...


def lexical_hit_to_chunk(hit: LexicalHit) -> RetrievedChunk:
    metadata = dict(hit.metadata)
    metadata.setdefault("collection_key", hit.collection_key)
    metadata.setdefault("source", hit.source)
    metadata["retrieval_origin"] = "lexical"
    source = str(metadata.get("source_absolute") or hit.source)
    return RetrievedChunk(
        collection_key=hit.collection_key,
        distance=0.0,
        document=hit.text,
        source=source,
        metadata=metadata,
    )


class HybridRetriever:
    """Combina busca vetorial e lexical com fallback vetorial seguro."""

    def __init__(
        self,
        settings: Settings,
        *,
        lexical_index_path: str | Path = ".indexes/lexical.sqlite3",
        enabled: bool = True,
        rrf_k: int = 60,
        vector_retriever: _VectorSearch | None = None,
        lexical_index: _LexicalSearch | None = None,
    ) -> None:
        self.vector_retriever = vector_retriever or VectorRetriever(settings)
        self.lexical_index_path = Path(lexical_index_path)
        self.enabled = enabled
        self.rrf_k = max(1, int(rrf_k))
        self._lexical_index = lexical_index

    def search(
        self,
        question: str,
        collections: list[CollectionConfig],
        candidates_per_collection: int,
        metadata_filter: MetadataFilter | None = None,
    ) -> list[RetrievedChunk]:
        vector_chunks = sorted(
            self.vector_retriever.search(
                question=question,
                collections=collections,
                candidates_per_collection=candidates_per_collection,
                metadata_filter=metadata_filter,
            ),
            key=lambda chunk: chunk.distance,
        )

        lexical_index = self._get_lexical_index()
        if lexical_index is None:
            return vector_chunks

        limit = max(1, candidates_per_collection) * max(1, len(collections))
        lexical_hits = lexical_index.search(
            query=question,
            collection_keys=[collection.key for collection in collections],
            limit=limit,
        )
        lexical_chunks = [
            lexical_hit_to_chunk(hit)
            for hit in lexical_hits
            if metadata_filter is None
            or metadata_filter.matches(
                str(hit.metadata.get("source_absolute") or hit.source),
                dict(hit.metadata),
            )
        ]

        if not lexical_chunks:
            return vector_chunks

        fused = reciprocal_rank_fusion(
            [vector_chunks, lexical_chunks],
            k=self.rrf_k,
        )
        return [chunk for chunk, _score in fused]

    def _get_lexical_index(self) -> _LexicalSearch | None:
        if not self.enabled:
            return None
        if self._lexical_index is not None:
            return self._lexical_index
        if not self.lexical_index_path.exists():
            return None
        self._lexical_index = SQLiteLexicalIndex(self.lexical_index_path)
        return self._lexical_index
