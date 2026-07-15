"""Expansão conservadora de consultas para melhorar a recuperação documental."""

from __future__ import annotations

import re


class QueryRewriter:
    """Acrescenta formas expandidas de siglas sem alterar a pergunta original."""

    _EXPANSIONS = {
        "ACMP": "Association of Change Management Professionals associação de profissionais de gestão da mudança",
        "CCMP": "Certified Change Management Professional certificação em gestão da mudança",
        "OCM": "Organizational Change Management gestão da mudança organizacional",
        "PMO": "Project Management Office escritório de projetos",
        "PPM": "Project Portfolio Management gestão de portfólio de projetos",
        "BPM": "Business Process Management gestão de processos de negócio",
        "HVAC": "Heating Ventilation and Air Conditioning climatização ventilação e ar condicionado",
        "GAMP": "Good Automated Manufacturing Practice boas práticas de manufatura automatizada",
        "FDA": "Food and Drug Administration",
        "EMA": "European Medicines Agency",
        "ANVISA": "Agência Nacional de Vigilância Sanitária",
    }

    _CHANGE_CONTEXT = {
        "acmp",
        "ccmp",
        "change",
        "mudanca",
        "mudança",
        "mudancas",
        "mudanças",
        "gestao",
        "gestão",
        "organizacional",
    }

    def rewrite(self, question: str) -> str:
        clean_question = " ".join(question.split())
        if not clean_question:
            return ""

        expansions: list[str] = []

        for acronym, expansion in self._EXPANSIONS.items():
            if self._contains_token(clean_question, acronym):
                expansions.append(expansion)

        if self._should_expand_gm(clean_question):
            expansions.append("GM gestão da mudança change management")

        unique_expansions = tuple(dict.fromkeys(expansions))
        if not unique_expansions:
            return clean_question

        return (
            f"{clean_question}\n\n"
            "Termos de busca relacionados: "
            + "; ".join(unique_expansions)
        )

    @staticmethod
    def _contains_token(text: str, token: str) -> bool:
        return re.search(rf"(?<!\w){re.escape(token)}(?!\w)", text, re.IGNORECASE) is not None

    def _should_expand_gm(self, question: str) -> bool:
        if not self._contains_token(question, "GM"):
            return False

        normalized = question.lower()
        return any(term in normalized for term in self._CHANGE_CONTEXT)
