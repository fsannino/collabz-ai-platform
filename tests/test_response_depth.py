from rag_ingest.prompt_manager import PromptManager
from rag_ingest.response_depth import ResponseDepthPolicy


def test_broad_question_requests_deeper_structure() -> None:
    instruction = ResponseDepthPolicy().instruction(
        "Detalhe a estrutura da ACMP, os países de atuação e explique o que a Gestão da Mudança faz."
    )

    assert "pergunta é ampla" in instruction
    assert "cada aspecto solicitado" in instruction
    assert "subtítulos" in instruction


def test_short_question_stays_concise() -> None:
    instruction = ResponseDepthPolicy().instruction("O que é a ACMP?")

    assert "pergunta é objetiva" in instruction
    assert "parágrafo curto" in instruction


def test_summary_style_overrides_broad_depth() -> None:
    instruction = ResponseDepthPolicy().instruction(
        "Detalhe a estrutura, os objetivos e os países de atuação da ACMP.",
        style="resumo",
    )

    assert "forma breve" in instruction
    assert "todos os pontos pedidos" in instruction


def test_intermediate_question_gets_balanced_depth() -> None:
    instruction = ResponseDepthPolicy().instruction(
        "Apresente informações sobre a certificação CCMP."
    )

    assert "profundidade intermediária" in instruction


def test_prompt_includes_depth_instruction() -> None:
    prompt = PromptManager().build(
        question="Detalhe a estrutura e os objetivos da ACMP.",
        context="[1] A ACMP possui objetivos documentados.",
        style="normal",
    )

    assert "A pergunta é ampla" in prompt
    assert "Não omita um aspecto solicitado" in prompt
    assert "CONTEXTO:" in prompt
