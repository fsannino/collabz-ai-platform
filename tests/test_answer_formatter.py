from rag_ingest.answer_formatter import add_inline_citations
from rag_ingest.models import RetrievedChunk


def _chunk(source: str) -> RetrievedChunk:
    return RetrievedChunk(
        collection_key="associacoes",
        distance=0.1,
        document="Trecho documental.",
        source=source,
        metadata={},
    )


def test_answer_without_sources_is_unchanged() -> None:
    assert add_inline_citations("Resposta.", ()) == "Resposta."


def test_answer_with_one_source_receives_citation() -> None:
    result = add_inline_citations(
        "Resposta.",
        (_chunk("associacoes/documento.pdf"),),
    )

    assert result == "Resposta. [1]"


def test_answer_with_two_sources_receives_two_citations() -> None:
    result = add_inline_citations(
        "Resposta.",
        (
            _chunk("associacoes/documento-a.pdf"),
            _chunk("associacoes/documento-b.pdf"),
        ),
    )

    assert result == "Resposta. [1][2]"


def test_duplicate_sources_receive_one_citation() -> None:
    chunk = _chunk("associacoes/documento.pdf")

    result = add_inline_citations(
        "Resposta.",
        (chunk, chunk),
    )

    assert result == "Resposta. [1]"


def test_duplicate_sources_ignore_case_and_slashes() -> None:
    first = _chunk("Associacoes\\ACMP\\Documento.pdf")
    second = _chunk("associacoes/ACMP/documento.pdf")

    result = add_inline_citations(
        "Resposta.",
        (first, second),
    )

    assert result == "Resposta. [1]"


def test_empty_answer_remains_empty() -> None:
    result = add_inline_citations(
        "",
        (_chunk("associacoes/documento.pdf"),),
    )

    assert result == ""
