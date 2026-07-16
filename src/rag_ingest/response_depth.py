"""Classificação simples da profundidade esperada para cada resposta."""

from __future__ import annotations

import re


class ResponseDepthPolicy:
    """Define instruções de profundidade sem ampliar a base factual."""

    _BROAD_MARKERS = (
        "detalhe",
        "explique",
        "descreva",
        "compare",
        "analise",
        "analise",
        "estrutura",
        "objetivos",
        "principais",
        "como funciona",
        "quais sao",
        "quais são",
    )

    _SHORT_MARKERS = (
        "o que e",
        "o que é",
        "quem e",
        "quem é",
        "quando",
        "onde",
        "qual e",
        "qual é",
    )

    def instruction(self, question: str, style: str = "normal") -> str:
        normalized = self._normalize(question)

        if style == "resumo":
            return (
                "Responda de forma breve, mas cubra todos os pontos pedidos que "
                "estejam sustentados pelas fontes."
            )

        if self._is_broad(normalized):
            return (
                "A pergunta é ampla. Desenvolva a resposta em profundidade moderada, "
                "cobrindo separadamente cada aspecto solicitado. Use de dois a quatro "
                "subtítulos curtos quando houver múltiplos temas. Em cada seção, apresente "
                "o fato principal, a explicação documental e a limitação aplicável. Não "
                "resuma excessivamente e não introduza conteúdo externo às fontes."
            )

        if self._is_short(normalized):
            return (
                "A pergunta é objetiva. Responda diretamente e acrescente um parágrafo "
                "curto de contexto somente quando houver evidência documental útil."
            )

        return (
            "Produza uma resposta de profundidade intermediária: resposta direta, "
            "explicação suficiente para compreensão e limitações documentais relevantes."
        )

    def _is_broad(self, normalized: str) -> bool:
        marker_count = sum(marker in normalized for marker in self._BROAD_MARKERS)
        conjunctions = len(re.findall(r"\b(e|alem disso|além disso|tambem|também)\b", normalized))
        return marker_count >= 1 or conjunctions >= 2 or len(normalized.split()) >= 18

    def _is_short(self, normalized: str) -> bool:
        return len(normalized.split()) <= 10 and any(
            normalized.startswith(marker) for marker in self._SHORT_MARKERS
        )

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.casefold().split())
