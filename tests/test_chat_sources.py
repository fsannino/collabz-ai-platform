from fastapi.testclient import TestClient

from rag_ingest import api
from rag_ingest.models import RagAnswer, RetrievedChunk


class _FakeAssistantWithSources:
    def answer(self, **_kwargs: object) -> RagAnswer:
        chunk = RetrievedChunk(
            collection_key="associacoes",
            distance=0.15,
            document="Trecho documental.",
            source="associacoes/ACMP/Certificação CCMP.pdf",
            metadata={"folder": "ACMP", "extension": ".pdf"},
        )
        return RagAnswer(
            answer="Resposta fundamentada.",
            model="llama3.2:1b",
            chunks=(chunk,),
        )


def _fake_build_assistant(
    _model: str | None = None,
) -> _FakeAssistantWithSources:
    return _FakeAssistantWithSources()


def test_non_streaming_chat_includes_sources(monkeypatch) -> None:
    monkeypatch.setattr(api, "build_assistant", _fake_build_assistant)
    client = TestClient(api.app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "collabz-rag",
            "messages": [{"role": "user", "content": "Pergunta"}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    content = response.json()["choices"][0]["message"]["content"]

    assert "Resposta fundamentada." in content
    assert "Fontes:" in content
    assert "Certificação CCMP.pdf" in content
    assert "Colecao: associacoes" in content
    assert "associacoes/ACMP/Certificação CCMP.pdf" in content


def test_streaming_chat_includes_sources(monkeypatch) -> None:
    monkeypatch.setattr(api, "build_assistant", _fake_build_assistant)
    client = TestClient(api.app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "collabz-rag",
            "messages": [{"role": "user", "content": "Pergunta"}],
            "stream": True,
        },
    )

    assert response.status_code == 200
    assert "Fontes:" in response.text
    assert "Certificação CCMP.pdf" in response.text
    assert "Colecao: associacoes" in response.text
    assert response.text.endswith("data: [DONE]\n\n")


def test_chat_deduplicates_same_source() -> None:
    chunk = RetrievedChunk(
        collection_key="associacoes",
        distance=0.1,
        document="Trecho.",
        source="associacoes/ACMP/documento.pdf",
        metadata={},
    )
    result = RagAnswer(
        answer="Resposta.",
        model="llama3.2:1b",
        chunks=(chunk, chunk),
    )

    content = api._answer_with_sources(result)

    assert content.count("documento.pdf") == 2
    assert content.count("1. documento.pdf") == 1
    assert "2. documento.pdf" not in content
