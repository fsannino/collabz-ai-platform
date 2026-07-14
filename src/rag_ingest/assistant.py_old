"""Serviço central de RAG: busca, contexto, resposta e fontes."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import chromadb
import requests

from rag_ingest.config import CollectionConfig, Settings, load_collections
from rag_ingest.ingest import OllamaEmbedder


@dataclass(frozen=True)
class RetrievedChunk:
    collection_key: str
    distance: float
    document: str
    source: str
    metadata: dict


@dataclass(frozen=True)
class RagAnswer:
    answer: str
    model: str
    chunks: tuple[RetrievedChunk, ...]


class RagAssistant:
    def __init__(
        self,
        settings: Settings,
        collections_path: Path = Path("collections.yaml"),
        llm_model: str | None = None,
    ) -> None:
        self.settings = settings
        self.collections = load_collections(collections_path)
        self.llm_model = llm_model or os.getenv("LLM_MODEL", "qwen3:4b")
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))
        self.timeout = int(os.getenv("LLM_TIMEOUT_SECONDS", "900"))

        self._embedder = OllamaEmbedder(
            base_url=settings.ollama_url,
            model=settings.embed_model,
            timeout_seconds=settings.ollama_timeout_seconds,
        )
        self._chroma = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
        )

    def answer(
        self,
        question: str,
        collection_keys: list[str] | None = None,
        use_all: bool = False,
        top_k: int = 8,
        per_collection: int = 5,
        style: str = "normal",
    ) -> RagAnswer:
        selected = self._select_collections(collection_keys, use_all)
        chunks = self.retrieve(
            question,
            selected,
            top_k,
            per_collection,
        )

        if not chunks:
            return RagAnswer(
                answer=(
                    "Não encontrei conteúdo suficiente nas coleções "
                    "selecionadas para responder com segurança."
                ),
                model=self.llm_model,
                chunks=(),
            )

        prompt = self._build_prompt(question, chunks, style)
        answer = self._generate(prompt)
        return RagAnswer(
            answer=answer.strip(),
            model=self.llm_model,
            chunks=tuple(chunks),
        )

    def retrieve(
        self,
        question: str,
        selected: Iterable[CollectionConfig],
        top_k: int,
        per_collection: int,
    ) -> list[RetrievedChunk]:
        query_embedding = self._embedder.embed(question)
        hits: list[RetrievedChunk] = []

        for config in selected:
            try:
                collection = self._chroma.get_collection(
                    config.collection_name
                )
            except Exception:
                continue

            count = collection.count()
            if count == 0:
                continue

            result = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(per_collection, count),
                include=["documents", "metadatas", "distances"],
            )

            documents = (result.get("documents") or [[]])[0]
            metadatas = (result.get("metadatas") or [[]])[0]
            distances = (result.get("distances") or [[]])[0]

            for document, metadata, distance in zip(
                documents,
                metadatas,
                distances,
                strict=False,
            ):
                metadata = metadata or {}
                source = (
                    metadata.get("source_absolute")
                    or metadata.get("source")
                    or "fonte desconhecida"
                )
                hits.append(
                    RetrievedChunk(
                        collection_key=config.key,
                        distance=float(distance),
                        document=document,
                        source=str(source),
                        metadata=metadata,
                    )
                )

        hits.sort(key=lambda item: item.distance)
        return hits[:top_k]

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

    def _build_prompt(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        style: str,
    ) -> str:
        styles = {
            "normal": "Responda de forma clara, direta e estruturada.",
            "executiva": "Responda de forma executiva e objetiva.",
            "tecnica": "Responda tecnicamente, com detalhes relevantes.",
            "academica": (
                "Responda em estilo acadêmico, distinguindo evidência "
                "de inferência."
            ),
            "resumo": "Produza um resumo curto com os pontos essenciais.",
        }
        instruction = styles.get(style, styles["normal"])

        context_parts = []
        for index, chunk in enumerate(chunks, start=1):
            context_parts.append(
                f"[FONTE {index}]\n"
                f"Coleção: {chunk.collection_key}\n"
                f"Arquivo: {chunk.source}\n"
                f"Trecho:\n{chunk.document.strip()}"
            )

        context = "\n\n".join(context_parts)

        return f"""Você é o assistente privado da plataforma CollabZ AI.

REGRAS:
1. Responda somente com base no contexto fornecido.
2. Não invente fatos, nomes, datas ou conclusões.
3. Se o contexto for insuficiente, declare isso.
4. Cite no corpo da resposta usando [Fonte 1], [Fonte 2] etc.
5. Ao final, inclua uma seção "Fontes utilizadas".
6. {instruction}

PERGUNTA:
{question}

CONTEXTO:
{context}

RESPOSTA:
"""

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
