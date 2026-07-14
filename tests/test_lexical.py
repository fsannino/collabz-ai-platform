from rag_ingest.lexical import BM25Ranker, tokenize
from rag_ingest.models import RetrievedChunk


def chunk(source: str, text: str, distance: float = 1.0) -> RetrievedChunk:
    return RetrievedChunk(
        collection_key="teste",
        distance=distance,
        document=text,
        source=source,
        metadata={},
    )


def test_tokenize_normalizes_accents_and_stopwords() -> None:
    assert tokenize("Quais associações profissionais são mencionadas?") == [
        "associacoes",
        "profissionais",
        "mencionadas",
    ]


def test_bm25_prioritizes_exact_lexical_evidence() -> None:
    ranker = BM25Ranker()
    results = ranker.rank(
        "certificação CCMP da ACMP",
        [
            chunk("generico.pdf", "gestão de projetos e tecnologia"),
            chunk("acmp.pdf", "A ACMP mantém o programa de certificação CCMP"),
        ],
    )

    assert results[0].source == "acmp.pdf"
