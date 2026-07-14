from rag_ingest.models import RetrievedChunk
from rag_ingest.rrf import reciprocal_rank_fusion


def chunk(source: str, distance: float) -> RetrievedChunk:
    return RetrievedChunk(
        collection_key="teste",
        distance=distance,
        document=source,
        source=source,
        metadata={"chunk_index": 0},
    )


def test_rrf_rewards_documents_present_in_both_rankings() -> None:
    a = chunk("a.pdf", 1.0)
    b = chunk("b.pdf", 2.0)
    c = chunk("c.pdf", 3.0)

    fused = reciprocal_rank_fusion(
        [
            [a, b, c],
            [c, a, b],
        ],
        k=60,
    )

    assert fused[0][0].source == "a.pdf"
    assert fused[0][1] > fused[-1][1]


def test_rrf_respects_limit() -> None:
    a = chunk("a.pdf", 1.0)
    b = chunk("b.pdf", 2.0)

    assert len(reciprocal_rank_fusion([[a, b]], limit=1)) == 1
