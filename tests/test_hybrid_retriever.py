from pathlib import Path

from rag_ingest.config import CollectionConfig
from rag_ingest.hybrid_retriever import HybridRetriever, lexical_hit_to_chunk
from rag_ingest.lexical_index import LexicalHit
from rag_ingest.metadata_filter import MetadataFilter
from rag_ingest.models import RetrievedChunk


class _FakeVectorRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks

    def search(self, **_kwargs: object) -> list[RetrievedChunk]:
        return list(self.chunks)


class _FakeLexicalIndex:
    def __init__(self, hits: list[LexicalHit]) -> None:
        self.hits = hits

    def search(self, **_kwargs: object) -> list[LexicalHit]:
        return list(self.hits)


def _collection() -> CollectionConfig:
    return CollectionConfig(
        key="associacoes",
        collection_name="associacoes_kb",
        roots=(),
    )


def _chunk(source: str, distance: float = 0.2) -> RetrievedChunk:
    return RetrievedChunk(
        collection_key="associacoes",
        distance=distance,
        document=f"Conteúdo de {source}",
        source=source,
        metadata={"chunk_index": 0},
    )


def _hit(source: str, score: float = 3.0) -> LexicalHit:
    return LexicalHit(
        document_id=source,
        collection_key="associacoes",
        source=source,
        text=f"Conteúdo lexical de {source}",
        metadata={"chunk_index": 0},
        score=score,
    )


def test_disabled_hybrid_returns_sorted_vector_results(tmp_path: Path) -> None:
    retriever = HybridRetriever(
        object(),  # type: ignore[arg-type]
        enabled=False,
        lexical_index_path=tmp_path / "missing.sqlite3",
        vector_retriever=_FakeVectorRetriever(
            [_chunk("b.pdf", 0.4), _chunk("a.pdf", 0.1)]
        ),
    )

    result = retriever.search(
        question="ACMP",
        collections=[_collection()],
        candidates_per_collection=20,
    )

    assert [chunk.source for chunk in result] == ["a.pdf", "b.pdf"]


def test_missing_lexical_index_falls_back_to_vector(tmp_path: Path) -> None:
    retriever = HybridRetriever(
        object(),  # type: ignore[arg-type]
        lexical_index_path=tmp_path / "missing.sqlite3",
        vector_retriever=_FakeVectorRetriever([_chunk("vector.pdf")]),
    )

    result = retriever.search(
        question="CCMP",
        collections=[_collection()],
        candidates_per_collection=20,
    )

    assert [chunk.source for chunk in result] == ["vector.pdf"]
    assert not (tmp_path / "missing.sqlite3").exists()


def test_hybrid_search_adds_lexical_only_document() -> None:
    retriever = HybridRetriever(
        object(),  # type: ignore[arg-type]
        vector_retriever=_FakeVectorRetriever([_chunk("vector.pdf")]),
        lexical_index=_FakeLexicalIndex([_hit("exact-ccmp.pdf")]),
    )

    result = retriever.search(
        question="CCMP",
        collections=[_collection()],
        candidates_per_collection=20,
    )

    assert {chunk.source for chunk in result} == {
        "vector.pdf",
        "exact-ccmp.pdf",
    }


def test_document_present_in_both_rankings_is_prioritized() -> None:
    common = _chunk("common.pdf", 0.3)
    retriever = HybridRetriever(
        object(),  # type: ignore[arg-type]
        vector_retriever=_FakeVectorRetriever(
            [_chunk("vector-only.pdf", 0.1), common]
        ),
        lexical_index=_FakeLexicalIndex(
            [_hit("common.pdf"), _hit("lexical-only.pdf")]
        ),
    )

    result = retriever.search(
        question="ACMP",
        collections=[_collection()],
        candidates_per_collection=20,
    )

    assert result[0].source == "common.pdf"


def test_lexical_hit_uses_absolute_source_when_available() -> None:
    hit = LexicalHit(
        document_id="doc-1",
        collection_key="associacoes",
        source="relativo.pdf",
        text="Conteúdo",
        metadata={"source_absolute": "//nas/documentos/absoluto.pdf"},
        score=2.0,
    )

    chunk = lexical_hit_to_chunk(hit)

    assert chunk.source == "//nas/documentos/absoluto.pdf"
    assert chunk.metadata["retrieval_origin"] == "lexical"


def test_metadata_filter_is_applied_to_lexical_results() -> None:
    retriever = HybridRetriever(
        object(),  # type: ignore[arg-type]
        vector_retriever=_FakeVectorRetriever([]),
        lexical_index=_FakeLexicalIndex(
            [_hit("permitido.pdf"), _hit("bloqueado.txt")]
        ),
    )

    result = retriever.search(
        question="ACMP",
        collections=[_collection()],
        candidates_per_collection=20,
        metadata_filter=MetadataFilter(file_extension=".pdf"),
    )

    assert [chunk.source for chunk in result] == ["permitido.pdf"]
