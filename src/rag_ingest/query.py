"""Consulta uma, várias ou todas as coleções do ChromaDB."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import chromadb
from dotenv import load_dotenv

from rag_ingest.config import CollectionConfig, Settings, load_collections
from rag_ingest.ingest import OllamaEmbedder


@dataclass(frozen=True)
class SearchHit:
    collection_key: str
    distance: float
    document: str
    metadata: dict


def load_environment() -> None:
    load_dotenv(".env.local", override=False)
    load_dotenv(".env", override=False)


def select_collections(
    all_collections: dict[str, CollectionConfig],
    requested: list[str] | None,
    use_all: bool,
) -> list[CollectionConfig]:
    enabled = {
        key: config
        for key, config in all_collections.items()
        if config.enabled
    }

    if use_all:
        return list(enabled.values())

    if not requested:
        raise RuntimeError("Informe --collection NOME ou --all.")

    missing = [key for key in requested if key not in enabled]
    if missing:
        raise RuntimeError(
            "Coleções inexistentes ou desabilitadas: "
            + ", ".join(missing)
        )

    return [enabled[key] for key in requested]


def main() -> None:
    load_environment()

    parser = argparse.ArgumentParser(
        description="Consulta RAG multicoleção"
    )

    selection_group = parser.add_mutually_exclusive_group(required=True)
    selection_group.add_argument(
        "--collection",
        action="append",
        help="nome da coleção; pode ser repetido",
    )
    selection_group.add_argument(
        "--all",
        action="store_true",
        help="pesquisa em todas as coleções habilitadas",
    )

    parser.add_argument("--query", required=True, help="texto da pergunta")
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="quantidade final de resultados",
    )
    parser.add_argument(
        "--per-collection",
        type=int,
        default=5,
        help="resultados buscados em cada coleção",
    )
    parser.add_argument(
        "--config",
        default="collections.yaml",
        help="arquivo de configuração das coleções",
    )

    args = parser.parse_args()

    settings = Settings.from_env()
    collections = load_collections(Path(args.config))
    selected = select_collections(
        all_collections=collections,
        requested=args.collection,
        use_all=args.all,
    )

    embedder = OllamaEmbedder(
        base_url=settings.ollama_url,
        model=settings.embed_model,
        timeout_seconds=settings.ollama_timeout_seconds,
    )
    query_embedding = embedder.embed(args.query)

    client = chromadb.HttpClient(
        host=settings.chroma_host,
        port=settings.chroma_port,
    )

    hits: list[SearchHit] = []

    for collection_config in selected:
        try:
            collection = client.get_collection(
                collection_config.collection_name
            )
        except Exception:
            continue

        count = collection.count()
        if count == 0:
            continue

        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(args.per_collection, count),
            include=["documents", "metadatas", "distances"],
        )

        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances,
            strict=False,
        ):
            hits.append(
                SearchHit(
                    collection_key=collection_config.key,
                    distance=float(distance),
                    document=document,
                    metadata=metadata or {},
                )
            )

    hits.sort(key=lambda item: item.distance)
    selected_hits = hits[: args.top_k]

    if not selected_hits:
        print("Nenhum resultado encontrado.")
        return

    for position, hit in enumerate(selected_hits, start=1):
        source = (
            hit.metadata.get("source_absolute")
            or hit.metadata.get("source")
            or "desconhecida"
        )
        print("=" * 90)
        print(
            f"{position}. coleção={hit.collection_key} "
            f"distância={hit.distance:.6f}"
        )
        print(f"fonte={source}")
        print("-" * 90)
        print(hit.document.strip())
        print()


if __name__ == "__main__":
    main()
