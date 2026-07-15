from rag_ingest.grounding import (
    acronym_expansions,
    listed_entities,
    validate_listed_entities,
)


def test_extracts_numbered_and_bulleted_entities() -> None:
    answer = """1. PMI São Paulo
- ACMP
* CCMP [Fonte 2]
"""
    assert listed_entities(answer) == ["PMI São Paulo", "ACMP", "CCMP"]


def test_extracts_acronym_expansions_from_running_text() -> None:
    assert acronym_expansions(
        "A CCMP refere-se à Comissão Central de Controle, Cooperação e Monitoramento."
    ) == ["Comissão Central de Controle, Cooperação e Monitoramento"]


def test_accepts_entities_present_in_context() -> None:
    valid, unsupported = validate_listed_entities(
        "1. PMI São Paulo\n2. ACMP",
        "O PMI São Paulo e a ACMP aparecem neste documento.",
    )
    assert valid is True
    assert unsupported == []


def test_rejects_hallucinated_entities() -> None:
    valid, unsupported = validate_listed_entities(
        "1. Associação de Comunicação e Gestão da Mudança (AGCM)",
        "O trecho menciona apenas PMI São Paulo.",
    )
    assert valid is False
    assert unsupported == [
        "Associação de Comunicação e Gestão da Mudança (AGCM)"
    ]


def test_rejects_unsupported_acronym_expansion_in_sentence() -> None:
    valid, unsupported = validate_listed_entities(
        "A CCMP refere-se à Comissão Central de Controle, Cooperação e Monitoramento.",
        "A CCMP é um rigoroso teste de habilidades em gestão de mudanças.",
    )
    assert valid is False
    assert unsupported == [
        "Comissão Central de Controle, Cooperação e Monitoramento"
    ]


def test_accepts_supported_acronym_expansion_in_sentence() -> None:
    valid, unsupported = validate_listed_entities(
        "CCMP significa Certified Change Management Professional.",
        "O documento apresenta Certified Change Management Professional (CCMP).",
    )
    assert valid is True
    assert unsupported == []
