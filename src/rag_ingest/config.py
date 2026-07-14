"""Configuração do pipeline RAG por variáveis de ambiente e collections.yaml."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_OLLAMA_TIMEOUT_SECONDS = 120
DEFAULT_DEBOUNCE_SECONDS = 2.0


@dataclass(frozen=True)
class Settings:
    ollama_url: str
    embed_model: str
    chroma_host: str
    chroma_port: int
    chunk_size: int
    chunk_overlap: int
    ollama_timeout_seconds: int
    debounce_seconds: float

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            ollama_url=_required("OLLAMA_URL"),
            embed_model=_required("EMBED_MODEL"),
            chroma_host=_required("CHROMA_HOST"),
            chroma_port=int(os.environ.get("CHROMA_PORT", "8000")),
            chunk_size=int(os.environ.get("CHUNK_SIZE", DEFAULT_CHUNK_SIZE)),
            chunk_overlap=int(
                os.environ.get("CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP)
            ),
            ollama_timeout_seconds=int(
                os.environ.get(
                    "OLLAMA_TIMEOUT_SECONDS",
                    DEFAULT_OLLAMA_TIMEOUT_SECONDS,
                )
            ),
            debounce_seconds=float(
                os.environ.get(
                    "DEBOUNCE_SECONDS",
                    DEFAULT_DEBOUNCE_SECONDS,
                )
            ),
        )


@dataclass(frozen=True)
class CollectionRoot:
    alias: str
    path: Path


@dataclass(frozen=True)
class CollectionConfig:
    key: str
    collection_name: str
    roots: tuple[CollectionRoot, ...]
    enabled: bool = True


def load_collections(config_path: Path) -> dict[str, CollectionConfig]:
    """Lê e valida o arquivo collections.yaml."""
    if not config_path.is_file():
        raise RuntimeError(f"Arquivo de coleções não encontrado: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    raw_collections = raw.get("collections")

    if not isinstance(raw_collections, dict) or not raw_collections:
        raise RuntimeError(
            "O arquivo collections.yaml não contém uma seção "
            "'collections' válida."
        )

    result: dict[str, CollectionConfig] = {}

    for key, item in raw_collections.items():
        if not isinstance(item, dict):
            raise RuntimeError(f"Configuração inválida para a coleção: {key}")

        enabled = bool(item.get("enabled", True))
        collection_name = str(item.get("collection_name") or f"{key}_kb")
        raw_roots = item.get("roots") or []

        roots: list[CollectionRoot] = []

        for index, root_item in enumerate(raw_roots):
            if isinstance(root_item, str):
                path = Path(root_item)
                alias = _safe_alias(path.name or f"root_{index + 1}")

            elif isinstance(root_item, dict):
                raw_path = root_item.get("path")
                if not raw_path:
                    raise RuntimeError(
                        f"Campo 'path' ausente na coleção '{key}'."
                    )

                path = Path(str(raw_path))
                alias = _safe_alias(
                    str(root_item.get("alias") or path.name)
                )

            else:
                raise RuntimeError(
                    f"Raiz inválida na coleção '{key}': {root_item!r}"
                )

            if not path.is_dir():
                raise RuntimeError(
                    f"Pasta da coleção '{key}' não existe "
                    f"ou não é diretório: {path}"
                )

            roots.append(CollectionRoot(alias=alias, path=path))

        if enabled and not roots:
            raise RuntimeError(
                f"A coleção habilitada '{key}' não possui nenhuma pasta."
            )

        result[key] = CollectionConfig(
            key=key,
            collection_name=collection_name,
            roots=tuple(roots),
            enabled=enabled,
        )

    return result


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Variável de ambiente obrigatória ausente: {name}"
        )
    return value


def _safe_alias(value: str) -> str:
    normalized = "".join(
        character.lower() if character.isalnum() else "_"
        for character in value.strip()
    )
    normalized = "_".join(
        part for part in normalized.split("_") if part
    )
    return normalized or "root"
