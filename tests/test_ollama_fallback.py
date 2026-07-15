from __future__ import annotations

from types import SimpleNamespace

import requests

from rag_ingest.assistant import RagAssistant


class FakeResponse:
    def __init__(self, payload: dict[str, str]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, str]:
        return self._payload


def make_assistant(urls: tuple[str, ...]) -> RagAssistant:
    assistant = RagAssistant.__new__(RagAssistant)
    assistant.settings = SimpleNamespace(ollama_url=urls[-1])
    assistant.llm_model = "llama3.2:1b"
    assistant.temperature = 0.1
    assistant.timeout = 30
    assistant.num_predict = 32
    assistant.num_ctx = 1024
    assistant.llm_ollama_urls = urls
    return assistant


def test_generate_uses_first_available_backend(monkeypatch) -> None:
    calls: list[str] = []

    def fake_post(url: str, **_kwargs) -> FakeResponse:
        calls.append(url)
        return FakeResponse({"response": "resposta local"})

    monkeypatch.setattr(requests, "post", fake_post)
    assistant = make_assistant(
        ("http://localhost:11434", "http://192.168.0.68:11434")
    )

    assert assistant._generate("pergunta") == "resposta local"
    assert calls == ["http://localhost:11434/api/generate"]


def test_generate_falls_back_to_nas(monkeypatch) -> None:
    calls: list[str] = []

    def fake_post(url: str, **_kwargs) -> FakeResponse:
        calls.append(url)
        if "localhost" in url:
            raise requests.ConnectionError("local indisponível")
        return FakeResponse({"response": "resposta do NAS"})

    monkeypatch.setattr(requests, "post", fake_post)
    assistant = make_assistant(
        ("http://localhost:11434", "http://192.168.0.68:11434")
    )

    assert assistant._generate("pergunta") == "resposta do NAS"
    assert calls == [
        "http://localhost:11434/api/generate",
        "http://192.168.0.68:11434/api/generate",
    ]


def test_generate_reports_all_backend_failures(monkeypatch) -> None:
    def fake_post(url: str, **_kwargs) -> FakeResponse:
        raise requests.Timeout(f"timeout em {url}")

    monkeypatch.setattr(requests, "post", fake_post)
    assistant = make_assistant(
        ("http://localhost:11434", "http://192.168.0.68:11434")
    )

    try:
        assistant._generate("pergunta")
    except RuntimeError as error:
        message = str(error)
    else:
        raise AssertionError("Era esperado RuntimeError")

    assert "localhost:11434" in message
    assert "192.168.0.68:11434" in message
