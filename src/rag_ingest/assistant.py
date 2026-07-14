"""Orquestrador central do RAG CollabZ AI Core v2."""

from __future__ import annotations

import os
from pathlib import Path

import requests

from rag_ingest.config import CollectionConfig, Settings, load_collections
from rag_ingest.context_builder import ContextBuilder
from rag_ingest.metadata_filter import MetadataFilter
from rag_ingest.models import RagAnswer
from rag_ingest.prompt_manager import PromptManager
from rag_ingest.reranker import DiversityReranker
from rag_ingest.retriever import VectorRetriever


class RagAssistant:
    def __init__(
        self,
        settings: Settings,
        collections_path: Path = Path("collections.yaml"),
        llm_model: str | None = None,
    ) -> None:
        self.settings = settings
        self.collections = load_collections(collections_path)
        self.llm_model = llm_model or os.getenv("LLM_MODEL", "llama3.2:1b")
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))
        self.timeout = int(os.getenv("LLM_TIMEOUT_SECONDS", "900"))
        self.max_chunks_per_source = int(
            os.getenv("RAG_MAX_CHUNKS_PER_SOURCE", "1")
        )
        self.max_context_characters = int(
            os.getenv("RAG_MAX_CONTEXT_CHARACTERS", "12000")
        )
        raw_max_distance = os.getenv("RAG_MAX_DISTANCE", "").strip()
        self.max_distance = (
            float(raw_max_distance) if raw_max_distance else None
        )
        lexical_weight = float(os.getenv("RAG_LEXICAL_WEIGHT", "250"))

        self._retriever = VectorRetriever(settings)
        self._reranker = DiversityReranker(
            max_chunks_per_source=self.max_chunks_per_source,
            max_distance=self.max_distance,
            lexical_weight=lexical_weight,
        )
        self._context_builder = ContextBuilder(
            max_characters=self.max_context_characters
        )
        self._prompt_manager = PromptManager()

    def answer(
        self,
        question: str,
        collection_keys: list[str] | None = None,
        use_all: bool = False,
        top_k: int = 8,
        per_collection: int = 5,
        style: str = "normal",
        metadata_filter: MetadataFilter | None = None,
    ) -> RagAnswer:
        selected = self._select_collections(collection_keys, use_all)
        candidates = self._retriever.search(
            question=question,
            collections=selected,
            candidates_per_collection=max(per_collection, top_k),
            metadata_filter=metadata_filter,
        )
        chunks = self._reranker.rank(
            candidates,
            limit=top_k,
            question=question,
        )

        if not chunks:
            return RagAnswer(
                answer=(
                    "Não encontrei evidência documental suficiente para "
                    "responder com segurança."
                ),
                model=self.llm_model,
                chunks=(),
            )

        context = self._context_builder.build(chunks)
        prompt = self._prompt_manager.build(
            question=question,
            context=context,
            style=style,
        )
        answer = self._generate(prompt)

        return RagAnswer(
            answer=answer.strip(),
            model=self.llm_model,
            chunks=tuple(chunks),
        )

    def _select_collections(
        self,
        requested: list[str] | None,
        use_all: bool,
    ) -> list[CollectionConfig]:
        enabled = {
            key: config
            for key, config in self.collections.items()
            if config.enabled
        }

        if use_all:
            return list(enabled.values())

        if not requested:
            raise RuntimeError("Informe uma coleção ou use --all.")

        missing = [key for key in requested if key not in enabled]
        if missing:
            raise RuntimeError(
                "Coleções inexistentes ou desabilitadas: "
                + ", ".join(missing)
            )

        return [enabled[key] for key in requested]

    def _generate(self, prompt: str) -> str:
        response = requests.post(
            f"{self.settings.ollama_url.rstrip('/')}/api/generate",
            json={
                "model": self.llm_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": self.temperature},
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        answer = response.json().get("response")
        if not answer:
            raise RuntimeError(
                f"Ollama retornou resposta vazia para {self.llm_model}."
            )
        return str(answer)
