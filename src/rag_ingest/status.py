"""Mostra o status das coleções do ChromaDB e dos serviços RAG."""

from __future__ import annotations

import sys
from pathlib import Path

import chromadb
import requests
from dotenv import load_dotenv

from rag_ingest.config import Settings, load_collections


def load_environment() -> None:
    load_dotenv('.env.local', override=False)
    load_dotenv('.env', override=False)


def check_http(url: str, timeout: int = 5) -> bool:
    try:
        response = requests.get(url, timeout=timeout)
        return response.ok
    except requests.RequestException:
        return False


def main() -> None:
    load_environment()
    settings = Settings.from_env()
    collections_config = load_collections(Path('collections.yaml'))

    ollama_online = check_http(f"{settings.ollama_url.rstrip('/')}/api/tags")
    chroma_online = check_http(
        f"http://{settings.chroma_host}:{settings.chroma_port}/api/v2/heartbeat"
    )

    print('=' * 72)
    print('              COLLABZ AI - STATUS DAS COLEÇÕES')
    print('=' * 72)
    print(f"Ollama........: {'ONLINE' if ollama_online else 'OFFLINE'}")
    print(f"ChromaDB......: {'ONLINE' if chroma_online else 'OFFLINE'}")
    print(f"Embedding.....: {settings.embed_model}")
    print('-' * 72)

    if not chroma_online:
        print('Não foi possível consultar o ChromaDB.')
        sys.exit(1)

    client = chromadb.HttpClient(
        host=settings.chroma_host,
        port=settings.chroma_port,
    )

    existing = {
        collection.name: collection
        for collection in client.list_collections()
    }

    total_chunks = 0
    total_collections = 0

    for key, config in collections_config.items():
        status = 'HABILITADA' if config.enabled else 'DESABILITADA'
        collection = existing.get(config.collection_name)

        if collection is None:
            count = 0
            state = 'NÃO CRIADA'
        else:
            count = collection.count()
            state = 'CRIADA'

        total_chunks += count
        total_collections += 1

        print(key)
        print(f'  Chroma.......: {config.collection_name}')
        print(f'  Configuração.: {status}')
        print(f'  Estado.......: {state}')
        print(f"  Chunks.......: {count:,}".replace(',', '.'))
        print()

    print('-' * 72)
    total_chunks_text = f"{total_chunks:,}".replace(',', '.')
    print(
        f'TOTAL DE COLEÇÕES: {total_collections} | '
        f'TOTAL DE CHUNKS: {total_chunks_text}'
    )
    print('=' * 72)


if __name__ == '__main__':
    main()
