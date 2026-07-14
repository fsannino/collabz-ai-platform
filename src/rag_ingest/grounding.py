"""Validação simples de entidades listadas contra o contexto recuperado."""

from __future__ import annotations

import re
import unicodedata


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


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


def validate_listed_entities(answer: str, context: str) -> tuple[bool, list[str]]:
    """Confirma se cada entidade listada aparece literalmente no contexto."""
    entities = listed_entities(answer)
    if not entities:
        return True, []

    normalized_context = _normalize(context)
    unsupported = [
        entity
        for entity in entities
        if _normalize(entity) not in normalized_context
    ]
    return not unsupported, unsupported
