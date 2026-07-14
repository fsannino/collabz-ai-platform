"""Pipeline multicoleção: NAS -> chunks -> embeddings (Ollama) -> ChromaDB.

Exemplos:
    python -m rag_ingest.ingest --collection associacoes --scan
    python -m rag_ingest.ingest --collection trabalho --collection projetos_codigo --scan
    python -m rag_ingest.ingest --all --scan
    python -m rag_ingest.ingest --all --watch
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import chromadb
import requests
from dotenv import load_dotenv

from rag_ingest.config import (
    CollectionConfig,
    CollectionRoot,
    Settings,
    load_collections,
)
from rag_ingest.extractors import SUPPORTED_EXTENSIONS, extract_text

logging.basicConfig(
    level=logging.INFO,
    format='{"ts":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
)
log = logging.getLogger("rag_ingest")


@dataclass(frozen=True)
class Chunk:
    text: str
    index: int


def split_into_chunks(text: str, chunk_size: int, overlap: int) -> list[Chunk]:
    """Divide o texto em blocos com sobreposição."""
    if not text.strip():
        return []

    if overlap >= chunk_size:
        raise ValueError("CHUNK_OVERLAP deve ser menor que CHUNK_SIZE.")

    chunks: list[Chunk] = []
    start = 0
    index = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        if end < text_length:
            paragraph_break = text.rfind("\n\n", start, end)
            if paragraph_break > start + chunk_size // 2:
                end = paragraph_break

        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(text=piece, index=index))
            index += 1

        start = end - overlap if end < text_length else text_length

    return chunks


class OllamaEmbedder:
    """Cliente para embeddings no Ollama."""

    def __init__(self, base_url: str, model: str, timeout_seconds: int) -> None:
        self._url = f"{base_url.rstrip('/')}/api/embeddings"
        self._model = model
        self._timeout = timeout_seconds

    def embed(self, text: str) -> list[float]:
        response = requests.post(
            self._url,
            json={"model": self._model, "prompt": text},
            timeout=self._timeout,
        )
        response.raise_for_status()

        embedding = response.json().get("embedding")
        if not embedding:
            raise ValueError(
                f"Ollama retornou embedding vazio para o modelo {self._model}."
            )

        return embedding


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)

    return digest.hexdigest()


class DocumentIndexer:
    """Indexador de uma única coleção do ChromaDB."""

    def __init__(
        self,
        settings: Settings,
        collection_config: CollectionConfig,
        embedder: OllamaEmbedder,
    ) -> None:
        self._settings = settings
        self._config = collection_config
        self._embedder = embedder

        client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
        )

        self._collection = client.get_or_create_collection(
            name=collection_config.collection_name,
            metadata={
                "embed_model": settings.embed_model,
                "collection_key": collection_config.key,
            },
        )

        self._guard_embed_model_consistency()

    def _guard_embed_model_consistency(self) -> None:
        existing = (self._collection.metadata or {}).get("embed_model")

        if existing and existing != self._settings.embed_model:
            raise RuntimeError(
                f"A coleção '{self._config.collection_name}' usa o modelo "
                f"'{existing}', mas EMBED_MODEL está definido como "
                f"'{self._settings.embed_model}'."
            )

    def index_file(self, path: Path, root: CollectionRoot) -> None:
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return

        try:
            content_hash = file_sha256(path)
        except OSError as error:
            log.warning("Arquivo inacessível: %s (%s)", path, error)
            return

        relative_path = path.relative_to(root.path).as_posix()
        document_id = f"{root.alias}/{relative_path}"

        if self._already_indexed(document_id, content_hash):
            log.info(
                "[%s] Inalterado, pulando: %s",
                self._config.key,
                document_id,
            )
            return

        text = extract_text(path)
        if not text:
            log.warning(
                "[%s] Sem texto extraível: %s",
                self._config.key,
                document_id,
            )
            return

        chunks = split_into_chunks(
            text=text,
            chunk_size=self._settings.chunk_size,
            overlap=self._settings.chunk_overlap,
        )

        if not chunks:
            return

        self._delete_existing_chunks(document_id)

        ids: list[str] = []
        documents: list[str] = []
        embeddings: list[list[float]] = []
        metadatas: list[dict] = []

        for chunk in chunks:
            ids.append(f"{document_id}::{chunk.index}")
            documents.append(chunk.text)
            embeddings.append(self._embedder.embed(chunk.text))
            metadatas.append(
                {
                    "source": document_id,
                    "source_absolute": str(path),
                    "collection_key": self._config.key,
                    "root_alias": root.alias,
                    "chunk_index": chunk.index,
                    "content_hash": content_hash,
                    "folder": str(Path(relative_path).parent),
                    "extension": path.suffix.lower(),
                }
            )

        self._collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        log.info(
            "[%s] Indexado: %s (%d chunks)",
            self._config.key,
            document_id,
            len(chunks),
        )

    def remove_file(self, path: Path, root: CollectionRoot) -> None:
        try:
            relative_path = path.relative_to(root.path).as_posix()
        except ValueError:
            return

        document_id = f"{root.alias}/{relative_path}"
        self._delete_existing_chunks(document_id)

        log.info(
            "[%s] Removido do índice: %s",
            self._config.key,
            document_id,
        )

    def _already_indexed(
        self,
        document_id: str,
        content_hash: str,
    ) -> bool:
        existing = self._collection.get(
            where={"source": document_id},
            limit=1,
            include=["metadatas"],
        )

        metadatas = existing.get("metadatas") or []

        return (
            bool(metadatas)
            and metadatas[0].get("content_hash") == content_hash
        )

    def _delete_existing_chunks(self, document_id: str) -> None:
        self._collection.delete(where={"source": document_id})


def full_scan(
    indexer: DocumentIndexer,
    collection_config: CollectionConfig,
) -> None:
    total = 0

    for root in collection_config.roots:
        log.info(
            "[%s] Scan em %s",
            collection_config.key,
            root.path,
        )

        for path in sorted(root.path.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                indexer.index_file(path, root)
                total += 1

    log.info(
        "[%s] Scan concluído: %d arquivos considerados",
        collection_config.key,
        total,
    )


def watch_collections(
    bindings: list[tuple[DocumentIndexer, CollectionConfig]],
    debounce_seconds: float,
) -> None:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    class Handler(FileSystemEventHandler):
        def __init__(
            self,
            indexer: DocumentIndexer,
            root: CollectionRoot,
        ) -> None:
            self._indexer = indexer
            self._root = root
            self._last_seen: dict[str, float] = {}

        def _debounced(self, source_path: str) -> bool:
            now = time.monotonic()
            last = self._last_seen.get(source_path, 0.0)
            self._last_seen[source_path] = now
            return (now - last) < debounce_seconds

        def on_created(self, event) -> None:
            self.on_modified(event)

        def on_modified(self, event) -> None:
            if event.is_directory or self._debounced(event.src_path):
                return

            self._indexer.index_file(
                Path(event.src_path),
                self._root,
            )

        def on_deleted(self, event) -> None:
            if not event.is_directory:
                self._indexer.remove_file(
                    Path(event.src_path),
                    self._root,
                )

        def on_moved(self, event) -> None:
            if event.is_directory:
                return

            self._indexer.remove_file(
                Path(event.src_path),
                self._root,
            )
            self._indexer.index_file(
                Path(event.dest_path),
                self._root,
            )

    observer = Observer()

    for indexer, collection_config in bindings:
        for root in collection_config.roots:
            observer.schedule(
                Handler(indexer, root),
                str(root.path),
                recursive=True,
            )
            log.info(
                "[%s] Monitorando %s",
                collection_config.key,
                root.path,
            )

    observer.start()
    log.info("Monitoramento ativo. Pressione Ctrl+C para sair.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()


def load_environment() -> None:
    """Carrega primeiro .env.local e depois .env."""
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

    missing = [
        key
        for key in requested
        if key not in enabled
    ]

    if missing:
        raise RuntimeError(
            "Coleções inexistentes ou desabilitadas: "
            + ", ".join(missing)
        )

    return [enabled[key] for key in requested]


def main() -> None:
    load_environment()

    parser = argparse.ArgumentParser(
        description="Ingestão RAG multicoleção"
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--scan",
        action="store_true",
        help="executa o scan e encerra",
    )
    mode_group.add_argument(
        "--watch",
        action="store_true",
        help="executa o scan e monitora alterações",
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
        help="todas as coleções habilitadas",
    )

    parser.add_argument(
        "--config",
        default="collections.yaml",
        help="caminho do arquivo de coleções",
    )

    args = parser.parse_args()

    settings = Settings.from_env()
    all_collections = load_collections(Path(args.config))
    selected = select_collections(
        all_collections=all_collections,
        requested=args.collection,
        use_all=args.all,
    )

    embedder = OllamaEmbedder(
        base_url=settings.ollama_url,
        model=settings.embed_model,
        timeout_seconds=settings.ollama_timeout_seconds,
    )

    bindings: list[tuple[DocumentIndexer, CollectionConfig]] = []

    for collection_config in selected:
        indexer = DocumentIndexer(
            settings=settings,
            collection_config=collection_config,
            embedder=embedder,
        )

        bindings.append((indexer, collection_config))
        full_scan(indexer, collection_config)

    if args.watch:
        watch_collections(
            bindings=bindings,
            debounce_seconds=settings.debounce_seconds,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        log.error("Falha fatal: %s", error)
        sys.exit(1)
