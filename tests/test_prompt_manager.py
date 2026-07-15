from rag_ingest.prompt_manager import PromptManager


def test_prompt_prioritizes_direct_readable_answers() -> None:
    prompt = PromptManager().build(
        question="O que é a ACMP?",
        context="[FONTE 1]\nTrecho documental.",
        style="normal",
    )

    assert "Comece pela resposta direta" in prompt
    assert "Evite repetir a mesma ideia" in prompt
    assert "parágrafos de no máximo três frases" in prompt
    assert "um ou dois parágrafos curtos" in prompt


def test_prompt_preserves_temporal_fidelity() -> None:
    prompt = PromptManager().build(
        question="Quem presidiu a associação?",
        context="[FONTE 1]\nPresidente em 2015.",
        style="normal",
    )

    assert "Preserve a temporalidade da fonte" in prompt
    assert "não transforme um cargo passado em cargo atual" in prompt


def test_prompt_avoids_unrequested_side_context() -> None:
    prompt = PromptManager().build(
        question="Explique a estrutura da associação.",
        context="[FONTE 1]\nEstrutura regional.",
        style="normal",
    )

    assert "Não acrescente biografias" in prompt
    assert "que não tenham sido pedidos" in prompt


def test_prompt_does_not_request_duplicate_source_section() -> None:
    prompt = PromptManager().build(
        question="Pergunta",
        context="Contexto",
        style="normal",
    )

    assert "Não crie uma seção própria de fontes" in prompt
    assert "a aplicação acrescentará as fontes" in prompt


def test_unknown_style_falls_back_to_normal() -> None:
    prompt = PromptManager().build(
        question="Pergunta",
        context="Contexto",
        style="inexistente",
    )

    assert "Use linguagem profissional, natural e fácil de ler" in prompt
