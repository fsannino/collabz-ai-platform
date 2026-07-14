# CollabZ RAG v0.1.0

Pipeline RAG multicoleção usando:

- Ollama para embeddings
- ChromaDB para busca vetorial
- pastas originais do NAS, sem cópia de documentos

## Instalação no Windows

```powershell
cd D:\00-Sistemas_Dev\collabz-rag
& "D:\00-Sistemas_Dev\.venv\Scripts\Activate.ps1"
python -m pip install -e .
Copy-Item .env.example .env.local
```

Edite `.env.local` e confirme os endereços do NAS.

## Comandos

```powershell
rag-status
rag-ingest --collection trabalho --scan
rag-query --collection trabalho --query "Quais clientes aparecem?"
```

Também continuam válidos:

```powershell
python -m rag_ingest.status
python -m rag_ingest.ingest --collection trabalho --scan
python -m rag_ingest.query --collection trabalho --query "..."
```

## Observação

O módulo `rag-ask` está reservado para a próxima fase, quando será adicionado um modelo LLM para gerar respostas consolidadas com fontes.
