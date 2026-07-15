from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv

from rag_ingest.config import CollectionConfig, Settings, load_collections
from rag_ingest.lexical_index import LexicalDocument, SQLiteLexicalIndex


def load_environment() -> None:
    load_dotenv(".env.local", override=False)
    load_dotenv(".env", override=False)


def select_collections(
    path: Path,
    requested: list[str] | None,
    use_all: bool,
) -> list[CollectionConfig]:
    collections = load_collections(path)
    enabled = {key: item for key, item in collections.items() if item.enabled}
    if use_all:
        return list(enabled.values())
    if not requested:
        raise RuntimeError("Informe --collection ou use --all.")
    missing = [key for key in requested if key not in enabled]
    if missing:
        raise RuntimeError(
            "Coleções inexistentes ou desabilitadas: " + ", ".join(missing)
        )
    return [enabled[key] for key in requested]


def build_collection(
    client: chromadb.HttpClient,
    index: SQLiteLexicalIndex,
    collection_config: CollectionConfig,
    batch_size: int,
) -> int:
    collection = client.get_collection(collection_config.collection_name)
    total = collection.count()
    index.clear_collection(collection_config.key)

    inserted = 0
    offset = 0
    while offset < total:
        batch = collection.get(
            limit=batch_size,
            offset=offset,
            include=["documents", "metadatas"],
        )
        ids = batch.get("ids") or []
        documents = batch.get("documents") or []
        metadatas = batch.get("metadatas") or []
        lexical_documents: list[LexicalDocument] = []

        for document_id, text, metadata in zip(ids, documents, metadatas):
            metadata_dict = dict(metadata or {})
            lexical_documents.append(
                LexicalDocument(
                    document_id=str(document_id),
                    collection_key=collection_config.key,
                    source=str(metadata_dict.get("source", document_id)),
                    text=str(text or ""),
                    metadata=metadata_dict,
                )
            )

        inserted += index.add_documents(lexical_documents)
        offset += len(ids)
        if not ids:
            break

    return inserted


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Constrói o índice lexical SQLite a partir das coleções ChromaDB"
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--collection", action="append")
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--config", default="collections.yaml")
    parser.add_argument(
        "--database",
        default=os.getenv("LEXICAL_INDEX_PATH", ".indexes/lexical.sqlite3"),
    )
    parser.add_argument("--batch-size", type=int, default=500)
    return parser


def main() -> None:
    load_environment()
    args = build_parser().parse_args()
    settings = Settings.from_env()
    selected = select_collections(Path(args.config), args.collection, args.all)

    client = chromadb.HttpClient(
        host=settings.chroma_host,
        port=settings.chroma_port,
    )
    index = SQLiteLexicalIndex(args.database)

    grand_total = 0
    for collection_config in selected:
        inserted = build_collection(
            client=client,
            index=index,
            collection_config=collection_config,
            batch_size=max(1, args.batch_size),
        )
        grand_total += inserted
        print(
            f"[{collection_config.key}] {inserted} chunks indexados em {args.database}"
        )

    print(f"Total: {grand_total} chunks no índice lexical.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERRO: {error}", file=sys.stderr)
        sys.exit(1)
