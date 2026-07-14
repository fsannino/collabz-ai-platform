from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_ingest.assistant import RagAssistant
from rag_ingest.config import Settings
from rag_ingest.metadata_filter import MetadataFilter


def load_environment() -> None:
    load_dotenv(".env.local", override=False)
    load_dotenv(".env", override=False)


load_environment()

app = FastAPI(title="CollabZ AI Core", version="0.3.0")


class MetadataFilterRequest(BaseModel):
    exact: dict[str, Any] = Field(default_factory=dict)
    source_contains: str | None = None
    folder_contains: str | None = None
    file_extension: str | None = None

    def to_domain(self) -> MetadataFilter:
        return MetadataFilter(
            exact=self.exact,
            source_contains=self.source_contains,
            folder_contains=self.folder_contains,
            file_extension=self.file_extension,
        )


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    collections: list[str] | None = None
    use_all: bool = False
    top_k: int = Field(default=8, ge=1, le=30)
    per_collection: int = Field(default=5, ge=1, le=50)
    style: str = "normal"
    model: str | None = None
    show_sources: bool = True
    filters: MetadataFilterRequest | None = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "collabz-rag"
    messages: list[ChatMessage]
    stream: bool = False


def build_assistant(model: str | None = None) -> RagAssistant:
    return RagAssistant(
        settings=Settings.from_env(),
        collections_path=Path(
            os.getenv("COLLECTIONS_CONFIG", "collections.yaml")
        ),
        llm_model=model,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "collabz-ai-core"}


@app.post("/v1/ask")
def ask(payload: AskRequest) -> dict[str, Any]:
    if not payload.use_all and not payload.collections:
        raise HTTPException(
            status_code=400,
            detail="Informe collections ou use_all=true.",
        )

    try:
        result = build_assistant(payload.model).answer(
            question=payload.question,
            collection_keys=payload.collections,
            use_all=payload.use_all,
            top_k=payload.top_k,
            per_collection=payload.per_collection,
            style=payload.style,
            metadata_filter=(
                payload.filters.to_domain() if payload.filters else None
            ),
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    sources = []
    if payload.show_sources:
        for index, chunk in enumerate(result.chunks, start=1):
            sources.append(
                {
                    "index": index,
                    "collection": chunk.collection_key,
                    "source": chunk.source,
                    "distance": chunk.distance,
                    "metadata": chunk.metadata,
                }
            )

    return {
        "answer": result.answer,
        "model": result.model,
        "sources": sources,
    }


@app.get("/v1/models")
def models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": "collabz-rag",
                "object": "model",
                "owned_by": "collabz",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat(payload: ChatCompletionRequest) -> dict[str, Any]:
    if payload.stream:
        raise HTTPException(
            status_code=400,
            detail="Streaming ainda não está habilitado.",
        )

    question = next(
        (
            message.content.strip()
            for message in reversed(payload.messages)
            if message.role == "user" and message.content.strip()
        ),
        None,
    )
    if not question:
        raise HTTPException(
            status_code=400,
            detail="Nenhuma mensagem do usuário foi informada.",
        )

    use_all = os.getenv(
        "OPENAI_COMPAT_USE_ALL",
        "false",
    ).lower() == "true"

    collections = [
        item.strip()
        for item in os.getenv(
            "OPENAI_COMPAT_COLLECTIONS",
            "associacoes",
        ).split(",")
        if item.strip()
    ]

    try:
        result = build_assistant().answer(
            question=question,
            collection_keys=None if use_all else collections,
            use_all=use_all,
            top_k=int(os.getenv("OPENAI_COMPAT_TOP_K", "6")),
            per_collection=int(
                os.getenv("OPENAI_COMPAT_PER_COLLECTION", "4")
            ),
            style=os.getenv("OPENAI_COMPAT_STYLE", "normal"),
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    return {
        "id": "chatcmpl-collabz",
        "object": "chat.completion",
        "model": "collabz-rag",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result.answer,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }
