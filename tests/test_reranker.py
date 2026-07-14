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


def test_reranker_exposes_final_score() -> None:
    reranker = DiversityReranker(lexical_weight=100.0)
    relevant = chunk("acmp.pdf", 200.0, "ACMP professional association")
    generic = chunk("generic.pdf", 180.0, "commercial companies")

    assert reranker.score(
        relevant,
        "Quais associações profissionais são mencionadas?",
    ) < reranker.score(
        generic,
        "Quais associações profissionais são mencionadas?",
    )


def test_reranker_removes_near_duplicates_across_sources() -> None:
    reranker = DiversityReranker(
        max_chunks_per_source=2,
        near_duplicate_threshold=0.75,
    )
    result = reranker.rank(
        [
            chunk(
                "copia-1.pdf",
                1.0,
                "ACMP oferece acesso ao centro de recursos e webinars aos associados",
            ),
            chunk(
                "copia-2.pdf",
                1.1,
                "ACMP oferece acesso ao centro de recursos, webinars e benefícios aos associados",
            ),
            chunk(
                "pmi.pdf",
                1.2,
                "PMI é uma associação profissional de gerenciamento de projetos",
            ),
        ],
        limit=3,
        question="associações profissionais",
    )

    assert [item.source for item in result] == ["copia-1.pdf", "pmi.pdf"]
