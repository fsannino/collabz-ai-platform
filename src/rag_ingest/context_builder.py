"""Montagem controlada do contexto enviado ao modelo de linguagem."""

from __future__ import annotations

from rag_ingest.models import RetrievedChunk


class ContextBuilder:
    def __init__(self, max_characters: int = 12000) -> None:
        self.max_characters = max(1000, max_characters)

    def build(self, chunks: list[RetrievedChunk]) -> str:
        parts: list[str] = []
        used = 0

        for index, chunk in enumerate(chunks, start=1):
            block = (
                f"[FONTE {index}]\n"
                f"Coleção: {chunk.collection_key}\n"
                f"Arquivo: {chunk.source}\n"
                f"Trecho:\n{chunk.document.strip()}"
            )

            if parts and used + len(block) > self.max_characters:
                break

            remaining = self.max_characters - used
            if len(block) > remaining:
                block = block[:remaining].rstrip()

            parts.append(block)
            used += len(block)

            if used >= self.max_characters:
                break

        return "\n\n".join(parts)
