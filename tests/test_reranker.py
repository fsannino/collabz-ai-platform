from rag_ingest.models import RetrievedChunk
from rag_ingest.reranker import DiversityReranker


def chunk(source: str, distance: float, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        collection_key="teste",
        distance=distance,
        document=text,
        source=source,
        metadata={},
    )


def test_reranker_deduplicates_content_and_sources() -> None:
    reranker = DiversityReranker(max_chunks_per_source=1)
    result = reranker.rank(
        [
            chunk("a.pdf", 1.0, "mesmo texto"),
            chunk("a.pdf", 1.1, "outro texto"),
            chunk("b.pdf", 1.2, "mesmo texto"),
            chunk("c.pdf", 1.3, "terceiro texto"),
        ],
        limit=3,
    )

    assert [item.source for item in result] == ["a.pdf", "c.pdf"]


def test_reranker_applies_distance_limit() -> None:
    reranker = DiversityReranker(max_distance=5.0)
    result = reranker.rank(
        [
            chunk("a.pdf", 4.0, "aceito"),
            chunk("b.pdf", 6.0, "rejeitado"),
        ],
        limit=5,
    )

    assert [item.source for item in result] == ["a.pdf"]


def test_reranker_prioritizes_lexical_overlap() -> None:
    reranker = DiversityReranker(lexical_weight=250.0)
    result = reranker.rank(
        [
            chunk("generico.pdf", 100.0, "empresas de tecnologia e varejo"),
            chunk(
                "associacao.pdf",
                140.0,
                "Association of Change Management Professionals e confederação",
            ),
        ],
        limit=1,
        question="Quais associações ou organizações são mencionadas?",
    )

    assert result[0].source == "associacao.pdf"
