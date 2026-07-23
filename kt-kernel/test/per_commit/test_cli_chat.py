from __future__ import annotations

import sys
from io import StringIO
from types import ModuleType, SimpleNamespace

import pytest
import typer
from rich.console import Console

from kt_kernel.cli.commands import chat as chat_module


class FakeSettings:
    def get(self, _key: str, default=None):
        return default


class FakeOpenAI:
    def __init__(self, *, base_url: str, api_key: str):
        assert base_url == "http://127.0.0.1:30000/v1"
        assert api_key == "EMPTY"
        self.models = SimpleNamespace(
            list=lambda: SimpleNamespace(data=[SimpleNamespace(id="Qwen3.5-35B-A3B-GPTQ-Int4")])
        )


def call_chat(monkeypatch: pytest.MonkeyPatch, console: Console) -> None:
    monkeypatch.setattr(chat_module, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(chat_module, "get_settings", lambda: FakeSettings())
    monkeypatch.setattr(chat_module, "console", console)
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "http_proxy",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
    ):
        monkeypatch.delenv(name, raising=False)
    chat_module.chat(
        host="127.0.0.1",
        port=30000,
        model=None,
        temperature=0.7,
        max_tokens=4,
        system_prompt=None,
        save_history=False,
        history_file=None,
        stream=True,
    )


def test_chat_tokenizer_failure_falls_back_to_estimated_counts(monkeypatch):
    fake_transformers = ModuleType("transformers")

    class FailingTokenizer:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            raise RuntimeError("tokenizer unavailable")

    fake_transformers.AutoTokenizer = FailingTokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    output = StringIO()
    test_console = Console(file=output, force_terminal=False)
    monkeypatch.setattr(test_console, "input", lambda *_args, **_kwargs: "/quit")

    call_chat(monkeypatch, test_console)

    assert "token counts will be estimated" in output.getvalue()


def test_chat_escapes_markup_in_connection_errors(monkeypatch):
    class FailingOpenAI(FakeOpenAI):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.models = SimpleNamespace(
                list=lambda: (_ for _ in ()).throw(RuntimeError("closing tag '[/dim]' does not match"))
            )

    monkeypatch.setattr(chat_module, "OpenAI", FailingOpenAI)
    monkeypatch.setattr(chat_module, "get_settings", lambda: FakeSettings())
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "http_proxy",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
    ):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(typer.Exit) as raised:
        chat_module.chat(
            host="127.0.0.1",
            port=30000,
            model=None,
            temperature=0.7,
            max_tokens=4,
            system_prompt=None,
            save_history=False,
            history_file=None,
            stream=True,
        )

    assert raised.value.exit_code == 1
