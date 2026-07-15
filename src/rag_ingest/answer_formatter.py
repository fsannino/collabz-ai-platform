from __future__ import annotations

from rag_ingest.models import RetrievedChunk


def _unique_sources(
    chunks: tuple[RetrievedChunk, ...],
) -> tuple[RetrievedChunk, ...]:
    unique: list[RetrievedChunk] = []
    seen: set[str] = set()

    for chunk in chunks:
        normalized = chunk.source.strip().replace("\\", "/").lower()

        if normalized in seen:
            continue

        seen.add(normalized)
        unique.append(chunk)

    return tuple(unique)


def add_inline_citations(
    answer: str,
    chunks: tuple[RetrievedChunk, ...],
) -> str:
    clean_answer = answer.rstrip()
    unique_chunks = _unique_sources(chunks)

    if not clean_answer or not unique_chunks:
        return clean_answer

    citations = "".join(
        f"[{index}]"
        for index in range(1, len(unique_chunks) + 1)
    )

    return f"{clean_answer} {citations}"
