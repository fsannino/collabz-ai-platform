"""Heurísticas conservadoras para priorizar fontes documentais mais úteis."""

from __future__ import annotations

from pathlib import PurePosixPath

from rag_ingest.models import RetrievedChunk


class SourceQualityPolicy:
    """Calcula penalidades suaves sem excluir documentos da recuperação."""

    _LOW_VALUE_PATH_MARKERS = {
        "modelos externos": 1.25,
        "modelo de apresentacao": 0.75,
        "modelo de apresentação": 0.75,
        "rascunho": 0.60,
        "temporario": 0.60,
        "temporário": 0.60,
        "backup": 0.40,
        "copia": 0.25,
        "cópia": 0.25,
    }

    _LOW_VALUE_FILENAMES = {
        "email": 1.00,
        "e-mail": 1.00,
        "convite": 0.20,
        "agenda": 0.15,
    }

    _EXTENSION_PENALTIES = {
        ".txt": 0.35,
        ".eml": 0.75,
        ".msg": 0.75,
    }

    def penalty(self, chunk: RetrievedChunk) -> float:
        """Retorna penalidade aditiva; zero representa uma fonte neutra."""
        normalized = chunk.source.replace("\\", "/").casefold()
        filename = PurePosixPath(normalized).name
        suffix = PurePosixPath(filename).suffix

        penalty = sum(
            value
            for marker, value in self._LOW_VALUE_PATH_MARKERS.items()
            if marker in normalized
        )
        penalty += sum(
            value
            for marker, value in self._LOW_VALUE_FILENAMES.items()
            if marker in filename
        )
        penalty += self._EXTENSION_PENALTIES.get(suffix, 0.0)
        return penalty
