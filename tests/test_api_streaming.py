from fastapi.testclient import TestClient

from rag_ingest import api
from rag_ingest.models import RagAnswer


class _FakeAssistant:
    def answer(self, **_kwargs: object) -> RagAnswer:
        return RagAnswer(
            answer="Resposta fundamentada.",
            model="llama3.2:1b",
            chunks=(),
        )


def _fake_build_assistant(_model: str | None = None) -> _FakeAssistant:
    return _FakeAssistant()


def test_chat_completion_without_stream(monkeypatch) -> None:
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
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"]["content"] == (
        "Resposta fundamentada."
    )
    assert payload["choices"][0]["finish_reason"] == "stop"


def test_chat_completion_with_stream(monkeypatch) -> None:
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
    assert response.headers["content-type"].startswith("text/event-stream")
    assert '"object": "chat.completion.chunk"' in response.text
    assert '"role": "assistant"' in response.text
    assert '"content": "Resposta fundamentada."' in response.text
    assert '"finish_reason": "stop"' in response.text
    assert response.text.endswith("data: [DONE]\n\n")


def test_chat_completion_requires_user_message(monkeypatch) -> None:
    monkeypatch.setattr(api, "build_assistant", _fake_build_assistant)
    client = TestClient(api.app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "collabz-rag",
            "messages": [{"role": "system", "content": "Instrução"}],
            "stream": True,
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Nenhuma mensagem do usuário foi informada."
    )
