from rag_ingest.query_rewriter import QueryRewriter


def test_plain_question_is_preserved() -> None:
    result = QueryRewriter().rewrite("Explique a estrutura da associação")

    assert result == "Explique a estrutura da associação"


def test_whitespace_is_normalized() -> None:
    result = QueryRewriter().rewrite("  Explique   a ACMP  ")

    assert result.startswith("Explique a ACMP")
    assert "Association of Change Management Professionals" in result


def test_known_acronyms_are_expanded_for_retrieval() -> None:
    result = QueryRewriter().rewrite("Compare PMO, PPM e BPM")

    assert "Project Management Office" in result
    assert "Project Portfolio Management" in result
    assert "Business Process Management" in result


def test_gm_is_expanded_in_change_management_context() -> None:
    result = QueryRewriter().rewrite(
        "Na ACMP, detalhe a estrutura e explique o que a GM faz"
    )

    assert "GM gestão da mudança change management" in result


def test_gm_is_not_expanded_without_change_context() -> None:
    result = QueryRewriter().rewrite("Quais veículos a GM produz?")

    assert result == "Quais veículos a GM produz?"


def test_empty_question_remains_empty() -> None:
    assert QueryRewriter().rewrite("   ") == ""
