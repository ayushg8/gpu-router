"""The terminal UI: defaults on enter, colour only on the icon, diffs coloured, prompts on
the chosen stream (stderr under --json)."""

from __future__ import annotations

import io

import pytest

from gpu_router.setup.ui import Mark, TerminalUi


def _ui(monkeypatch: pytest.MonkeyPatch, replies: list[str]) -> tuple[TerminalUi, io.StringIO]:
    stream = io.StringIO()
    answers = iter(replies)
    monkeypatch.setattr("builtins.input", lambda: next(answers))
    return TerminalUi(stream, interactive=True), stream


def test_ask_defaults_and_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    ui, stream = _ui(monkeypatch, ["", "", "y", "no", "YES"])
    assert ui.ask("go?", True) is True
    assert ui.ask("go?", False) is False
    assert ui.ask("go?", False) is True
    assert ui.ask("go?", True) is False
    assert ui.ask("go?", False) is True
    text = stream.getvalue()
    assert "go? [Y/n]" in text
    assert "go? [y/N]" in text


def test_eof_is_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = io.StringIO()

    def eof() -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    ui = TerminalUi(stream, interactive=True)
    assert ui.ask("go?", False) is False


def test_choose_repeats_until_a_known_key(monkeypatch: pytest.MonkeyPatch) -> None:
    ui, _ = _ui(monkeypatch, ["x", "p"])
    assert ui.choose("how?", [("b", "browser"), ("p", "paste")], default="b") == "p"
    ui2, _ = _ui(monkeypatch, [""])
    assert ui2.choose("how?", [("b", "browser"), ("p", "paste")], default="b") == "b"


def test_items_and_commands_render_plain_text() -> None:
    stream = io.StringIO()
    ui = TerminalUi(stream, interactive=False)
    ui.item(Mark.DONE, "kaggle: stored", None)
    ui.item(Mark.WAIT, "hf: missing", "gpu login hf")
    ui.block("--- a\n+++ b\n@@ -1 +1 @@\n-old\n+new\n")
    out = stream.getvalue()
    assert "✓ kaggle: stored" in out
    assert "⏸ hf: missing" in out
    assert "$ gpu login hf" in out
    assert "+new" in out
    assert "-old" in out
    assert not ui.interactive
