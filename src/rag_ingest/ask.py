"""Interface de linha de comando para respostas RAG com fontes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from rag_ingest.assistant import RagAssistant
from rag_ingest.config import Settings
from rag_ingest.metadata_filter import MetadataFilter


def load_environment() -> None:
    load_dotenv(".env.local", override=False)
    load_dotenv(".env", override=False)


def main() -> None:
    load_environment()

    parser = argparse.ArgumentParser(
        description="CollabZ RAG: resposta consolidada com fontes"
    )

    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--collection",
        action="append",
        help="coleção a consultar; pode ser repetida",
    )
    selection.add_argument(
        "--all",
        action="store_true",
        help="consulta todas as coleções habilitadas",
    )

    parser.add_argument("--question", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--style",
        choices=["normal", "executiva", "tecnica", "academica", "resumo"],
        default="normal",
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--per-collection", type=int, default=5)
    parser.add_argument("--config", default="collections.yaml")
    parser.add_argument("--show-context", action="store_true")
    parser.add_argument("--source-contains")
    parser.add_argument("--folder-contains")
    parser.add_argument("--file-extension")
    parser.add_argument(
        "--metadata",
        action="append",
        default=[],
        metavar="CHAVE=VALOR",
        help="filtro exato de metadado; pode ser repetido",
    )

    args = parser.parse_args()

    exact: dict[str, str] = {}
    for item in args.metadata:
        if "=" not in item:
            parser.error(f"Filtro inválido: {item}. Use CHAVE=VALOR.")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            parser.error("A chave do filtro de metadado não pode ser vazia.")
        exact[key] = value.strip()

    metadata_filter = None
    if exact or args.source_contains or args.folder_contains or args.file_extension:
        metadata_filter = MetadataFilter(
            exact=exact,
            source_contains=args.source_contains,
            folder_contains=args.folder_contains,
            file_extension=args.file_extension,
        )

    assistant = RagAssistant(
        settings=Settings.from_env(),
        collections_path=Path(args.config),
        llm_model=args.model,
    )

    result = assistant.answer(
        question=args.question,
        collection_keys=args.collection,
        use_all=args.all,
        top_k=args.top_k,
        per_collection=args.per_collection,
        style=args.style,
        metadata_filter=metadata_filter,
    )

    print("=" * 90)
    print(f"MODELO: {result.model}")
    print("=" * 90)
    print(result.answer)
    print()

    if args.show_context and result.chunks:
        print("=" * 90)
        print("TRECHOS RECUPERADOS")
        print("=" * 90)

        for index, chunk in enumerate(result.chunks, start=1):
            print(
                f"[{index}] coleção={chunk.collection_key} "
                f"distância={chunk.distance:.6f}"
            )
            print(f"fonte={chunk.source}")
            print("-" * 90)
            print(chunk.document.strip())
            print()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERRO: {error}", file=sys.stderr)
        sys.exit(1)
