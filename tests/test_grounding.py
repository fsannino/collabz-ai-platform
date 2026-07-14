from rag_ingest.grounding import listed_entities, validate_listed_entities


def test_extracts_numbered_and_bulleted_entities() -> None:
    answer = """1. PMI São Paulo
- ACMP
* CCMP [Fonte 2]
"""
    assert listed_entities(answer) == ["PMI São Paulo", "ACMP", "CCMP"]


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
