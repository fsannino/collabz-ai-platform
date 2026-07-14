"""Recuperação vetorial multicoleção no ChromaDB."""

from __future__ import annotations

from collections.abc import Iterable

import chromadb

from rag_ingest.config import CollectionConfig, Settings
from rag_ingest.ingest import OllamaEmbedder
from rag_ingest.metadata_filter import MetadataFilter
from rag_ingest.models import RetrievedChunk


class VectorRetriever:
    def __init__(self, settings: Settings) -> None:
        self._embedder = OllamaEmbedder(
            base_url=settings.ollama_url,
            model=settings.embed_model,
            timeout_seconds=settings.ollama_timeout_seconds,
        )
        self._client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
        )

    def search(
        self,
        question: str,
        collections: Iterable[CollectionConfig],
        candidates_per_collection: int,
        metadata_filter: MetadataFilter | None = None,
    ) -> list[RetrievedChunk]:
        query_embedding = self._embedder.embed(question)
        hits: list[RetrievedChunk] = []
        where = metadata_filter.to_chroma_where() if metadata_filter else None

        for config in collections:
            try:
                collection = self._client.get_collection(config.collection_name)
                count = collection.count()
            except Exception:
                continue

            if count == 0:
                continue

            query_args = {
                "query_embeddings": [query_embedding],
                "n_results": min(candidates_per_collection, count),
                "include": ["documents", "metadatas", "distances"],
            }
            if where:
                query_args["where"] = where

            result = collection.query(**query_args)

            documents = (result.get("documents") or [[]])[0]
            metadatas = (result.get("metadatas") or [[]])[0]
            distances = (result.get("distances") or [[]])[0]

            for document, metadata, distance in zip(
                documents,
                metadatas,
                distances,
                strict=False,
            ):
                if not document or not document.strip():
                    continue
                metadata = metadata or {}
                source = (
                    metadata.get("source_absolute")
                    or metadata.get("source")
                    or "fonte desconhecida"
                )
                source = str(source)

                if metadata_filter and not metadata_filter.matches(source, metadata):
                    continue

                hits.append(
                    RetrievedChunk(
                        collection_key=config.key,
                        distance=float(distance),
                        document=document.strip(),
                        source=source,
                        metadata=metadata,
                    )
                )

        return sorted(hits, key=lambda item: item.distance)
