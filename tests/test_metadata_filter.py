from rag_ingest.metadata_filter import MetadataFilter


def test_to_chroma_where_none():
    assert MetadataFilter().to_chroma_where() is None


def test_to_chroma_where_single_value():
    result = MetadataFilter(exact={"language": "pt"}).to_chroma_where()
    assert result == {"language": "pt"}


def test_to_chroma_where_multiple_values():
    result = MetadataFilter(
        exact={"language": "pt", "author": "ACMP"}
    ).to_chroma_where()
    assert result == {
        "$and": [
            {"language": "pt"},
            {"author": "ACMP"},
        ]
    }


def test_matches_source_folder_and_extension():
    metadata_filter = MetadataFilter(
        source_contains="ACMP",
        folder_contains="04-01-ACMP",
        file_extension="pdf",
    )
    source = r"\\192.168.0.68\home\004-Associacoes\04-01-ACMP\manual.pdf"
    assert metadata_filter.matches(source, {}) is True


def test_rejects_non_matching_source():
    metadata_filter = MetadataFilter(folder_contains="04-01-ACMP")
    source = r"\\192.168.0.68\home\005-Artigos\texto.pdf"
    assert metadata_filter.matches(source, {}) is False


def test_exact_metadata_match():
    metadata_filter = MetadataFilter(exact={"language": "pt"})
    assert metadata_filter.matches("arquivo.pdf", {"language": "pt"}) is True
    assert metadata_filter.matches("arquivo.pdf", {"language": "en"}) is False
