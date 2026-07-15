"""Validação conservadora de entidades e expansões contra o contexto."""

from __future__ import annotations

import re
import unicodedata


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    normalized = "".join(
        ch for ch in normalized if not unicodedata.combining(ch)
    )
    return re.sub(r"\s+", " ", normalized).strip()


def listed_entities(answer: str) -> list[str]:
    """Extrai itens numerados ou com marcadores da resposta."""
    entities: list[str] = []
    pattern = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*(.+?)\s*$")
    for line in answer.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        value = match.group(1)
        value = re.sub(r"\s*\[[^\]]+\]\s*$", "", value)
        value = value.strip(" *.;:")
        if value:
            entities.append(value)
    return entities


def acronym_expansions(answer: str) -> list[str]:
    """Extrai expansões declaradas para siglas em frases corridas.

    Exemplos reconhecidos:
    - ``CCMP significa Certified Change Management Professional``
    - ``CCMP refere-se à Comissão Central de Controle``
    - ``CCMP é a sigla para ...``
    - ``CCMP (Conformidade, Confiabilidade e Compatibilidade de Produção)``

    A sigla precisa estar efetivamente em maiúsculas. Isso evita interpretar
    palavras comuns, como ``Mudança (AGCM)``, como se ``Mudança`` fosse sigla.
    """
    acronym = r"[A-ZÁÉÍÓÚÇ][A-Z0-9ÁÉÍÓÚÇ-]{1,15}"
    patterns = (
        re.compile(
            rf"\b({acronym})\b\s+"
            r"(?i:significa|refere-se\s+(?:a|à)|é\s+a\s+sigla\s+para)\s+"
            r"([^.;:\n]+)",
        ),
        re.compile(
            rf"\b({acronym})\b\s*\(([^)\n]+)\)",
        ),
    )

    expansions: list[str] = []
    for pattern in patterns:
        for match in pattern.finditer(answer):
            value = match.group(2).strip(" *.;:")
            if value and value not in expansions:
                expansions.append(value)
    return expansions


def validate_listed_entities(answer: str, context: str) -> tuple[bool, list[str]]:
    """Confirma entidades listadas e expansões de siglas no contexto.

    A função preserva a interface histórica, mas também rejeita expansões
    declarativas ou parentéticas de siglas que não estejam literalmente
    apoiadas pelo contexto recuperado.
    """
    claims = listed_entities(answer) + acronym_expansions(answer)
    if not claims:
        return True, []

    normalized_context = _normalize(context)
    unsupported = [
        claim
        for claim in claims
        if _normalize(claim) not in normalized_context
    ]
    return not unsupported, unsupported
