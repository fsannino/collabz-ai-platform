from pathlib import Path

from rag_ingest.lexical_index import (
    LexicalDocument,
    SQLiteLexicalIndex,
    tokenize,
)


def test_tokenize_normalizes_accents() -> None:
    assert tokenize("Associação Profissional") == [
        "associacao",
        "profissional",
    ]


def test_sqlite_lexical_index_ranks_relevant_document(tmp_path: Path) -> None:
    index = SQLiteLexicalIndex(tmp_path / "lexical.sqlite3")
    index.add_documents(
        [
            LexicalDocument(
                document_id="acmp-1",
                collection_key="associacoes",
                source="acmp.pdf",
                text="A ACMP é uma associação profissional de gestão de mudanças.",
                metadata={"page": 1},
            ),
            LexicalDocument(
                document_id="empresa-1",
                collection_key="associacoes",
                source="empresa.pdf",
                text="Empresa comercial de tecnologia e distribuição.",
                metadata={"page": 2},
            ),
        ]
    )

    results = index.search(
        query="associação profissional",
        collection_keys=["associacoes"],
        limit=2,
    )

    assert results
    assert results[0].source == "acmp.pdf"
    assert results[0].score > 0
    assert index.count("associacoes") == 2


def test_clear_collection_removes_documents(tmp_path: Path) -> None:
    index = SQLiteLexicalIndex(tmp_path / "lexical.sqlite3")
    index.add_documents(
        [
            LexicalDocument(
                document_id="doc-1",
                collection_key="associacoes",
                source="acmp.pdf",
                text="ACMP associação profissional",
                metadata={},
            )
        ]
    )

    index.clear_collection("associacoes")

    assert index.count("associacoes") == 0
    assert index.search(
        query="ACMP",
        collection_keys=["associacoes"],
    ) == []
