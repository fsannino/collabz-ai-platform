"""CLI de avaliação objetiva da recuperação."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from statistics import mean
from time import perf_counter

from rag_ingest.metrics import ndcg_at_k, recall_at_k, reciprocal_rank


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Avalia a qualidade da recuperação RAG")
    parser.add_argument("--collection", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--mode", choices=("vector", "lexical", "hybrid"), default="hybrid")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--per-collection", type=int, default=40)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def run_retrieval(
    collection: str,
    question: str,
    mode: str,
    top_k: int,
    per_collection: int,
) -> dict:
    command = [
        sys.executable,
        "-m",
        "rag_ingest.retrieve",
        "--collection",
        collection,
        "--question",
        question,
        "--mode",
        mode,
        "--top-k",
        str(top_k),
        "--per-collection",
        str(per_collection),
        "--json",
    ]

    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUTF8"] = "1"

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())

    start = completed.stdout.find("{")
    if start < 0:
        raise RuntimeError("Saída JSON não encontrada no rag-retrieve.")
    return json.loads(completed.stdout[start:])


def main() -> None:
    args = build_parser().parse_args()
    dataset_path = Path(args.dataset)
    cases = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise RuntimeError("Dataset deve conter uma lista não vazia de casos.")

    started = perf_counter()
    rows = []
    for case in cases:
        question = str(case["question"])
        expected = [str(item) for item in case.get("expected_sources", [])]
        result = run_retrieval(
            collection=args.collection,
            question=question,
            mode=args.mode,
            top_k=args.top_k,
            per_collection=args.per_collection,
        )
        sources = [str(item["source"]) for item in result.get("results", [])]
        rows.append(
            {
                "question": question,
                "expected_sources": expected,
                "retrieved_sources": sources,
                "recall_at_5": recall_at_k(sources, expected, 5),
                "recall_at_10": recall_at_k(sources, expected, 10),
                "reciprocal_rank": reciprocal_rank(sources, expected),
                "ndcg_at_10": ndcg_at_k(sources, expected, 10),
            }
        )

    payload = {
        "dataset": str(dataset_path),
        "collection": args.collection,
        "mode": args.mode,
        "questions": len(rows),
        "recall_at_5": mean(row["recall_at_5"] for row in rows),
        "recall_at_10": mean(row["recall_at_10"] for row in rows),
        "mrr": mean(row["reciprocal_rank"] for row in rows),
        "ndcg_at_10": mean(row["ndcg_at_10"] for row in rows),
        "elapsed_seconds": round(perf_counter() - started, 3),
        "cases": rows,
    }

    if args.as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print("=" * 70)
    print("RAG EVALUATION")
    print("=" * 70)
    print(f"Dataset............... {payload['dataset']}")
    print(f"Coleção............... {payload['collection']}")
    print(f"Modo.................. {payload['mode']}")
    print(f"Perguntas............. {payload['questions']}")
    print(f"Recall@5.............. {payload['recall_at_5']:.4f}")
    print(f"Recall@10............. {payload['recall_at_10']:.4f}")
    print(f"MRR................... {payload['mrr']:.4f}")
    print(f"NDCG@10............... {payload['ndcg_at_10']:.4f}")
    print(f"Tempo total........... {payload['elapsed_seconds']:.3f}s")


if __name__ == "__main__":
    main()
