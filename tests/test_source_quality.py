from rag_ingest.models import RetrievedChunk
from rag_ingest.reranker import DiversityReranker
from rag_ingest.source_quality import SourceQualityPolicy


def _chunk(source: str, text: str = "ACMP gestão da mudança") -> RetrievedChunk:
    return RetrievedChunk(
        collection_key="associacoes",
        distance=0.1,
        document=text,
        source=source,
        metadata={},
    )


def test_regular_pdf_has_no_quality_penalty() -> None:
    chunk = _chunk("associacoes/ACMP/Certificacao CCMP.pdf")

    assert SourceQualityPolicy().penalty(chunk) == 0.0


def test_external_email_text_receives_high_penalty() -> None:
    chunk = _chunk(
        "associacoes/ACMP/99-MODELOS EXTERNOS/Modelo de Apresentacao/99-email.txt"
    )

    assert SourceQualityPolicy().penalty(chunk) >= 3.0


def test_message_file_receives_penalty() -> None:
    chunk = _chunk("associacoes/ACMP/comunicacao/reuniao.msg")

    assert SourceQualityPolicy().penalty(chunk) == 0.75


def test_reranker_prioritizes_regular_document_over_external_email() -> None:
    regular = _chunk("associacoes/ACMP/Certificacao CCMP.pdf")
    email = _chunk(
        "associacoes/ACMP/99-MODELOS EXTERNOS/Modelo de Apresentacao/99-email.txt"
    )
    reranker = DiversityReranker(
        lexical_weight=0.0,
        source_quality_weight=100.0,
        near_duplicate_threshold=1.0,
    )

    result = reranker.rank(
        [email, regular],
        limit=2,
        question="ACMP gestão da mudança",
    )

    assert result[0].source.endswith("Certificacao CCMP.pdf")


def test_source_quality_can_be_disabled() -> None:
    regular = _chunk("associacoes/ACMP/Certificacao CCMP.pdf")
    email = _chunk(
        "associacoes/ACMP/99-MODELOS EXTERNOS/Modelo de Apresentacao/99-email.txt"
    )
    reranker = DiversityReranker(
        lexical_weight=0.0,
        source_quality_weight=0.0,
        near_duplicate_threshold=1.0,
    )

    assert reranker.score(email) == reranker.score(regular)
