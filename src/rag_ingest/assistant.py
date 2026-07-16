"""Orquestrador central do RAG CollabZ AI Core v2."""

from __future__ import annotations

import os
from pathlib import Path

import requests
from rag_ingest.config import CollectionConfig, Settings, load_collections
from rag_ingest.context_builder import ContextBuilder
from rag_ingest.grounding import validate_listed_entities
from rag_ingest.metadata_filter import MetadataFilter
from rag_ingest.models import RagAnswer
from rag_ingest.prompt_manager import PromptManager
from rag_ingest.query_rewriter import QueryRewriter
from rag_ingest.reranker import DiversityReranker
from rag_ingest.retrieval_policy import candidate_pool_size
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
        self.num_predict = int(os.getenv("LLM_NUM_PREDICT", "192"))
        self.num_ctx = int(os.getenv("LLM_NUM_CTX", "2048"))
        self.llm_ollama_urls = self._load_llm_ollama_urls()
        self.max_chunks_per_source = int(
            os.getenv("RAG_MAX_CHUNKS_PER_SOURCE", "1")
        )
        self.max_context_characters = int(
            os.getenv("RAG_MAX_CONTEXT_CHARACTERS", "12000")
        )
        self.candidate_multiplier = int(
            os.getenv("RAG_CANDIDATE_MULTIPLIER", "4")
        )
        self.min_candidates_per_collection = int(
            os.getenv("RAG_MIN_CANDIDATES_PER_COLLECTION", "20")
        )
        self.max_candidates_per_collection = int(
            os.getenv("RAG_MAX_CANDIDATES_PER_COLLECTION", "100")
        )
        raw_max_distance = os.getenv("RAG_MAX_DISTANCE", "").strip()
        self.max_distance = (
            float(raw_max_distance) if raw_max_distance else None
        )
        lexical_weight = float(os.getenv("RAG_LEXICAL_WEIGHT", "250"))
        source_quality_weight = float(
            os.getenv("RAG_SOURCE_QUALITY_WEIGHT", "100")
        )
        self.strict_grounding = os.getenv(
            "RAG_STRICT_GROUNDING", "true"
        ).strip().lower() in {"1", "true", "yes", "on"}

        self._retriever = VectorRetriever(settings)
        self._query_rewriter = QueryRewriter()
        self._reranker = DiversityReranker(
            max_chunks_per_source=self.max_chunks_per_source,
            max_distance=self.max_distance,
            lexical_weight=lexical_weight,
            source_quality_weight=source_quality_weight,
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
        retrieval_question = self._query_rewriter.rewrite(question)
        candidates_per_collection = candidate_pool_size(
            top_k=top_k,
            per_collection=per_collection,
            multiplier=self.candidate_multiplier,
            minimum=self.min_candidates_per_collection,
            maximum=self.max_candidates_per_collection,
        )
        candidates = self._retriever.search(
            question=retrieval_question,
            collections=selected,
            candidates_per_collection=candidates_per_collection,
            metadata_filter=metadata_filter,
        )
        chunks = self._reranker.rank(
            candidates,
            limit=top_k,
            question=retrieval_question,
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
        answer = self._generate(prompt).strip()

        if self.strict_grounding:
            valid, _unsupported = validate_listed_entities(answer, context)
            if not valid:
                answer = (
                    "Não encontrei evidência documental suficiente para "
                    "responder com segurança. A resposta gerada foi descartada "
                    "porque continha entidades não confirmadas literalmente "
                    "nas fontes recuperadas."
                )

        return RagAnswer(
            answer=answer,
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

    def _load_llm_ollama_urls(self) -> tuple[str, ...]:
        """Carrega backends de chat em ordem de preferência.

        ``LLM_OLLAMA_URLS`` aceita URLs separadas por vírgula. Quando ausente,
        mantém compatibilidade usando ``OLLAMA_URL`` do Settings.
        """
        raw_urls = os.getenv("LLM_OLLAMA_URLS", "")
        urls = [item.strip().rstrip("/") for item in raw_urls.split(",")]
        urls = [item for item in urls if item]
        if not urls:
            urls = [self.settings.ollama_url.rstrip("/")]
        return tuple(dict.fromkeys(urls))

    def _generate(self, prompt: str) -> str:
        payload = {
            "model": self.llm_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
                "num_ctx": self.num_ctx,
            },
        }
        failures: list[str] = []

        for base_url in self.llm_ollama_urls:
            try:
                response = requests.post(
                    f"{base_url}/api/generate",
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                answer = response.json().get("response")
                if answer:
                    return str(answer)
                failures.append(f"{base_url}: resposta vazia")
            except (requests.RequestException, ValueError) as error:
                failures.append(f"{base_url}: {error}")

        details = "; ".join(failures) or "nenhum backend configurado"
        raise RuntimeError(
            f"Nenhum backend Ollama respondeu para {self.llm_model}. {details}"
        )
