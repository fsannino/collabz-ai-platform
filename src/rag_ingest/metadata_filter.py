"""Filtros de metadados para consultas no ChromaDB e pós-filtragem local."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MetadataFilter:
    """Define filtros exatos e textuais aplicados à recuperação.

    `exact` é convertido para o operador `where` do ChromaDB.
    Os demais campos são aplicados após a consulta, pois exigem
    comparação parcial ou normalização de caminho.
    """

    exact: dict[str, Any] = field(default_factory=dict)
    source_contains: str | None = None
    folder_contains: str | None = None
    file_extension: str | None = None

    def to_chroma_where(self) -> dict[str, Any] | None:
        pairs = [
            {str(key): value}
            for key, value in self.exact.items()
            if value is not None
        ]
        if not pairs:
            return None
        if len(pairs) == 1:
            return pairs[0]
        return {"$and": pairs}

    def matches(self, source: str, metadata: dict[str, Any]) -> bool:
        normalized_source = source.replace("\\", "/").lower()

        if self.source_contains:
            needle = self.source_contains.strip().lower()
            if needle and needle not in normalized_source:
                return False

        if self.folder_contains:
            needle = self.folder_contains.replace("\\", "/").strip("/").lower()
            if needle and needle not in normalized_source:
                return False

        if self.file_extension:
            extension = self.file_extension.strip().lower()
            if extension and not extension.startswith("."):
                extension = f".{extension}"
            if extension and not normalized_source.endswith(extension):
                return False

        for key, expected in self.exact.items():
            if expected is None:
                continue
            if metadata.get(key) != expected:
                return False

        return True
