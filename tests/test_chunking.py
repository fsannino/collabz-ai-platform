from rag_ingest.ingest import split_into_chunks


def test_split_into_chunks_returns_content():
    chunks = split_into_chunks("abc " * 1000, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(chunk.text for chunk in chunks)
