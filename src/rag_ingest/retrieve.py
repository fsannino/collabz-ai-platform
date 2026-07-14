"""CLI de diagnóstico da recuperação, sem chamar o modelo de linguagem."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

from dotenv import load_dotenv

from rag_ingest.config import CollectionConfig, Settings, load_collections
from rag_ingest.lexical import BM25Ranker
from rag_ingest.metadata_filter import MetadataFilter
from rag_ingest.reranker import DiversityReranker
from rag_ingest.retriever import VectorRetriever
from rag_ingest.rrf import reciprocal_rank_fusion, chunk_key


def load_environment() -> None:
    load_dotenv(".env.local", override=False)
    load_dotenv(".env", override=False)


def parse_metadata(values: list[str] | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Metadado inválido: {value}. Use chave=valor.")
        key, raw = value.split("=", 1)
        key = key.strip()
        raw = raw.strip()
        if not key:
            raise ValueError(f"Metadado inválido: {value}.")
        result[key] = raw
    return result


def select_collections(path: Path, requested: list[str] | None, use_all: bool) -> list[CollectionConfig]:
    collections = load_collections(path)
    enabled = {key: item for key, item in collections.items() if item.enabled}
    if use_all:
        return list(enabled.values())
    if not requested:
        raise RuntimeError("Informe --collection ou use --all.")
    missing = [key for key in requested if key not in enabled]
    if missing:
        raise RuntimeError("Coleções inexistentes ou desabilitadas: " + ", ".join(missing))
    return [enabled[key] for key in requested]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnóstico vetorial, lexical ou híbrido sem chamar o LLM")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--collection", action="append")
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--question", required=True)
    parser.add_argument("--mode", choices=("vector", "lexical", "hybrid"), default="hybrid")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--per-collection", type=int, default=30)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--config", default="collections.yaml")
    parser.add_argument("--source-contains")
    parser.add_argument("--folder-contains")
    parser.add_argument("--file-extension")
    parser.add_argument("--metadata", action="append")
    parser.add_argument("--no-reranker", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--show-metadata", action="store_true")
    return parser


def main() -> None:
    load_environment()
    args = build_parser().parse_args()
    started = perf_counter()
    settings = Settings.from_env()
    selected = select_collections(Path(args.config), args.collection, args.all)
    metadata_filter = MetadataFilter(
        exact=parse_metadata(args.metadata),
        source_contains=args.source_contains,
        folder_contains=args.folder_contains,
        file_extension=args.file_extension,
    )

    retriever = VectorRetriever(settings)
    retrieve_started = perf_counter()
    candidates = retriever.search(
        question=args.question,
        collections=selected,
        candidates_per_collection=max(args.per_collection, args.top_k),
        metadata_filter=metadata_filter,
    )
    retrieve_seconds = perf_counter() - retrieve_started

    vector_ranking = sorted(candidates, key=lambda item: item.distance)
    lexical_ranker = BM25Ranker()
    lexical_scored = lexical_ranker.score_all(args.question, candidates)
    lexical_ranking = [chunk for chunk, _ in lexical_scored]
    lexical_scores = {chunk_key(chunk): score for chunk, score in lexical_scored}
    rrf_scores: dict[tuple[str, str, int | None], float] = {}

    if args.mode == "vector":
        ordered = vector_ranking
    elif args.mode == "lexical":
        ordered = lexical_ranking
    else:
        fused = reciprocal_rank_fusion([vector_ranking, lexical_ranking], k=args.rrf_k)
        ordered = [chunk for chunk, _ in fused]
        rrf_scores = {chunk_key(chunk): score for chunk, score in fused}

    reranker: DiversityReranker | None = None
    if args.no_reranker:
        chunks = ordered[: args.top_k]
    else:
        reranker = DiversityReranker(
            max_chunks_per_source=int(os.getenv("RAG_MAX_CHUNKS_PER_SOURCE", "1")),
            max_distance=float(os.environ["RAG_MAX_DISTANCE"]) if os.getenv("RAG_MAX_DISTANCE", "").strip() else None,
            lexical_weight=float(os.getenv("RAG_LEXICAL_WEIGHT", "250")),
            near_duplicate_threshold=float(os.getenv("RAG_NEAR_DUPLICATE_THRESHOLD", "0.88")),
        )
        if args.mode == "vector":
            chunks = reranker.rank(ordered, limit=args.top_k, question=args.question)
        else:
            chunks = reranker.select_ordered(ordered, limit=args.top_k)

    total_seconds = perf_counter() - started

    def reranker_score(chunk: Any) -> float:
        return float(chunk.distance if reranker is None else reranker.score(chunk, args.question))

    payload = {
        "question": args.question,
        "mode": args.mode,
        "candidate_count": len(candidates),
        "result_count": len(chunks),
        "retrieve_seconds": round(retrieve_seconds, 3),
        "total_seconds": round(total_seconds, 3),
        "results": [
            {
                "rank": index,
                "collection": chunk.collection_key,
                "distance": chunk.distance,
                "bm25_score": lexical_scores.get(chunk_key(chunk), 0.0),
                "rrf_score": rrf_scores.get(chunk_key(chunk)),
                "reranker_score": reranker_score(chunk),
                "source": chunk.source,
                "metadata": chunk.metadata,
                "document": chunk.document,
            }
            for index, chunk in enumerate(chunks, start=1)
        ],
    }

    if args.as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return

    print("=" * 90)
    print("DIAGNÓSTICO DE RECUPERAÇÃO")
    print("=" * 90)
    print(f"Pergunta............. {args.question}")
    print(f"Modo................. {args.mode}")
    print(f"Candidatos........... {len(candidates)}")
    print(f"Resultados........... {len(chunks)}")
    print(f"Tempo recuperação.... {retrieve_seconds:.3f}s")
    print(f"Tempo total.......... {total_seconds:.3f}s")

    if not chunks:
        print("\nNenhum trecho encontrado.")
        return

    for index, chunk in enumerate(chunks, start=1):
        key = chunk_key(chunk)
        print("\n" + "=" * 90)
        print(f"RANK {index}")
        print("=" * 90)
        print(f"coleção={chunk.collection_key}")
        print(f"distância={chunk.distance:.6f}")
        print(f"bm25_score={lexical_scores.get(key, 0.0):.6f}")
        if key in rrf_scores:
            print(f"rrf_score={rrf_scores[key]:.8f}")
        print(f"score_reranker={reranker_score(chunk):.6f}")
        print(f"fonte={chunk.source}")
        if args.show_metadata:
            print("metadados=" + json.dumps(chunk.metadata, ensure_ascii=False, default=str))
        print("-" * 90)
        print(chunk.document.strip())


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERRO: {error}", file=sys.stderr)
        sys.exit(1)
