from __future__ import annotations

import json
import math
import re
import sqlite3
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class LexicalDocument:
    document_id: str
    collection_key: str
    source: str
    text: str
    metadata: dict[str, object]


@dataclass(frozen=True)
class LexicalHit:
    document_id: str
    collection_key: str
    source: str
    text: str
    metadata: dict[str, object]
    score: float


def tokenize(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKD", text.lower())
    ascii_text = "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    )
    return re.findall(r"[a-z0-9_]{2,}", ascii_text)


class SQLiteLexicalIndex:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;

                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    collection_key TEXT NOT NULL,
                    source TEXT NOT NULL,
                    text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    document_length INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS postings (
                    term TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    term_frequency INTEGER NOT NULL,
                    PRIMARY KEY (term, document_id),
                    FOREIGN KEY (document_id)
                        REFERENCES documents(document_id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_postings_term
                ON postings(term);

                CREATE INDEX IF NOT EXISTS idx_documents_collection
                ON documents(collection_key);
                """
            )

    def clear_collection(self, collection_key: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM documents WHERE collection_key = ?",
                (collection_key,),
            )

    def add_documents(self, documents: Iterable[LexicalDocument]) -> int:
        inserted = 0
        with self._connect() as connection:
            for document in documents:
                terms = tokenize(document.text)
                frequencies = Counter(terms)
                connection.execute(
                    """
                    INSERT OR REPLACE INTO documents (
                        document_id, collection_key, source, text,
                        metadata_json, document_length
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document.document_id,
                        document.collection_key,
                        document.source,
                        document.text,
                        json.dumps(document.metadata, ensure_ascii=False, default=str),
                        len(terms),
                    ),
                )
                connection.execute(
                    "DELETE FROM postings WHERE document_id = ?",
                    (document.document_id,),
                )
                connection.executemany(
                    """
                    INSERT INTO postings (term, document_id, term_frequency)
                    VALUES (?, ?, ?)
                    """,
                    [
                        (term, document.document_id, frequency)
                        for term, frequency in frequencies.items()
                    ],
                )
                inserted += 1
        return inserted

    def count(self, collection_key: str | None = None) -> int:
        with self._connect() as connection:
            if collection_key is None:
                row = connection.execute(
                    "SELECT COUNT(*) AS total FROM documents"
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT COUNT(*) AS total FROM documents WHERE collection_key = ?",
                    (collection_key,),
                ).fetchone()
            return int(row["total"])

    def search(
        self,
        query: str,
        collection_keys: list[str],
        limit: int = 20,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> list[LexicalHit]:
        query_terms = tokenize(query)
        if not query_terms or not collection_keys:
            return []

        placeholders = ",".join("?" for _ in collection_keys)
        with self._connect() as connection:
            stats = connection.execute(
                f"""
                SELECT COUNT(*) AS document_count,
                       COALESCE(AVG(document_length), 0) AS average_length
                FROM documents
                WHERE collection_key IN ({placeholders})
                """,
                collection_keys,
            ).fetchone()
            document_count = int(stats["document_count"])
            average_length = float(stats["average_length"])
            if document_count == 0 or average_length == 0:
                return []

            scores: dict[str, float] = {}
            for term in set(query_terms):
                df_row = connection.execute(
                    f"""
                    SELECT COUNT(*) AS frequency
                    FROM postings p
                    JOIN documents d ON d.document_id = p.document_id
                    WHERE p.term = ?
                      AND d.collection_key IN ({placeholders})
                    """,
                    [term, *collection_keys],
                ).fetchone()
                document_frequency = int(df_row["frequency"])
                if not document_frequency:
                    continue

                idf = math.log(
                    1
                    + (document_count - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                rows = connection.execute(
                    f"""
                    SELECT d.document_id, d.document_length, p.term_frequency
                    FROM postings p
                    JOIN documents d ON d.document_id = p.document_id
                    WHERE p.term = ?
                      AND d.collection_key IN ({placeholders})
                    """,
                    [term, *collection_keys],
                ).fetchall()
                for row in rows:
                    tf = float(row["term_frequency"])
                    dl = float(row["document_length"])
                    denominator = tf + k1 * (1 - b + b * dl / average_length)
                    score = idf * (tf * (k1 + 1) / denominator)
                    document_id = str(row["document_id"])
                    scores[document_id] = scores.get(document_id, 0.0) + score

            if not scores:
                return []

            ranked_ids = sorted(scores, key=scores.get, reverse=True)[:limit]
            ranked_placeholders = ",".join("?" for _ in ranked_ids)
            rows = connection.execute(
                f"""
                SELECT document_id, collection_key, source, text, metadata_json
                FROM documents
                WHERE document_id IN ({ranked_placeholders})
                """,
                ranked_ids,
            ).fetchall()
            by_id = {str(row["document_id"]): row for row in rows}
            return [
                LexicalHit(
                    document_id=document_id,
                    collection_key=str(by_id[document_id]["collection_key"]),
                    source=str(by_id[document_id]["source"]),
                    text=str(by_id[document_id]["text"]),
                    metadata=json.loads(str(by_id[document_id]["metadata_json"])),
                    score=float(scores[document_id]),
                )
                for document_id in ranked_ids
                if document_id in by_id
            ]
