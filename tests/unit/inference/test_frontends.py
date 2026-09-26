"""The inference lane end to end through a real in-process daemon: /v1/infer*, `gpu infer`
(single, --dry-run, --file batches, --list), `gpu providers` / `gpu quota` additions,
`gpu login groq|gemini|cloudflare`, the MCP tool gpu_infer and the shell's /infer.

The daemon is the shell suite's InProcDaemon (uvicorn on a thread, tmp home, port 0, fake
GPU providers); its InferenceService is swapped for one whose HTTP goes to mocked
providers, so nothing reaches the network (invariant 20) and keys live in the in-memory
keyring (conftest)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from rich.console import Console

from gpu_router import secrets
from gpu_router.cli.app import main
from gpu_router.client import GpuClient
from gpu_router.clock import FakeClock
from gpu_router.errors import ApiError, InvalidRequest
from gpu_router.inference import remote
from gpu_router.inference.models import InferRequest
from tests.shell.conftest import InProcDaemon
from tests.unit.inference.fakes import KEYS, FakeProviders, groq_headers, login, service

NOW = 1_790_251_200.0


@pytest.fixture
def fake() -> FakeProviders:
    return FakeProviders()


@pytest.fixture
def daemon(
    gpu_home: Path, monkeypatch: pytest.MonkeyPatch, fake: FakeProviders
) -> Iterator[InProcDaemon]:
    monkeypatch.setenv("GPU_ROUTER_NO_AUTOSTART", "1")
    d = InProcDaemon(home=gpu_home)
    d.start()
    built = d.runtime.inference
    assert built is not None  # the real runtime builds the lane
    built.close()
    d.runtime.inference = service(fake, FakeClock(NOW))
    try:
        yield d
    finally:
        d.stop()


def cli(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    code = main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


# --------------------------------------------------------------------------- API


def test_api_routes_calls_and_views(daemon: InProcDaemon, fake: FakeProviders) -> None:
    login("groq")
    with GpuClient.from_env(daemon.paths) as client:
        decision = remote.route(client, InferRequest(model="gpt-oss-20b", prompt="hi"))
        assert decision.outcome == "place"
        assert decision.chosen is not None
        assert fake.calls() == []  # a dry run sends nothing
        result = remote.infer(client, InferRequest(model="gpt-oss-20b", prompt="hi"))
        assert (result.provider, result.text) == ("groq", "4")
        views = {v.name: v for v in remote.providers(client)}
        assert views["groq"].logged_in
        assert not views["gemini"].logged_in
        quota = {q.provider: q for q in remote.quota(client)}
        assert quota["groq"].requests_today == 1
        with pytest.raises(InvalidRequest, match="no free inference provider serves"):
            remote.infer(client, InferRequest(model="nope", prompt="hi"))
        with pytest.raises(ApiError) as info:
            remote.infer(client, InferRequest(model="gemini-3.5-flash", prompt="hi"))
        assert info.value.raw_code == "auth_required"
        assert "gpu login gemini" in (info.value.hint or "")


def test_api_infer_runs_on_the_lanes_own_executor_with_a_deadline(
    daemon: InProcDaemon, fake: FakeProviders
) -> None:
    """Review fix: /v1/infer used asyncio.to_thread (the default executor that submits and
    bundling share) and the daemon kept calling providers after the client gave up."""
    import threading

    from gpu_router.inference.service import INFER_WORKERS

    login("groq")
    threads: list[str] = []

    def answer(request: httpx.Request) -> httpx.Response:
        threads.append(threading.current_thread().name)
        return httpx.Response(200, json={"choices": [{"message": {"content": "4"}}]})

    fake.queue("groq", answer)
    svc = daemon.runtime.inference
    assert svc is not None
    deadlines: list[float | None] = []
    real = svc.infer

    def spy(req: InferRequest) -> Any:
        deadlines.append(req.deadline_s)
        return real(req)

    svc.infer = spy  # type: ignore[method-assign]
    with GpuClient.from_env(daemon.paths) as client:
        remote.infer(client, InferRequest(model="gpt-oss-20b", prompt="hi"))
    assert threads
    assert threads[0].startswith("gpu-infer")
    assert svc.executor._max_workers == INFER_WORKERS
    assert deadlines == [remote.INFER_TIMEOUT_S - remote.DEADLINE_MARGIN_S]


def test_api_validation_is_a_400(daemon: InProcDaemon) -> None:
    with GpuClient.from_env(daemon.paths) as client, pytest.raises(InvalidRequest):
        client.request("POST", "/infer", json={"model": "x"})  # neither prompt nor messages


# --------------------------------------------------------------------------- CLI


def test_gpu_infer_prints_the_reply_on_stdout_and_the_rest_on_stderr(
    daemon: InProcDaemon, fake: FakeProviders, capsys: pytest.CaptureFixture[str]
) -> None:
    login("groq")
    code, out, err = cli(capsys, "infer", "-m", "gpt-oss-20b", "What", "is", "2+2?")
    assert code == 0
    assert out == "4\n"
    assert "groq · gpt-oss-20b (openai/gpt-oss-20b) · 12+3 tokens" in err
    assert "left today" in err
    assert fake.calls("groq")[0].body["messages"] == [{"role": "user", "content": "What is 2+2?"}]


def test_gpu_infer_says_when_the_reply_was_cut(
    daemon: InProcDaemon, fake: FakeProviders, capsys: pytest.CaptureFixture[str]
) -> None:
    # live 2026-09-25: gemini-3.5-flash with --max-tokens 20 thought and answered nothing
    login("gemini")
    doc = {
        "choices": [{"message": {"role": "assistant", "content": ""}, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 6, "completion_tokens": 0},
    }
    fake.queue("gemini", httpx.Response(200, json=doc))
    code, out, err = cli(capsys, "infer", "-m", "gemini-3.5-flash", "--max-tokens", "20", "hi")
    assert code == 0
    assert out == "\n"
    assert "an empty reply: cut at the token limit" in err
    assert "raise --max-tokens" in err
    assert fake.calls("gemini")[-1].body["max_tokens"] == 20


def test_gpu_infer_json_dry_run_and_errors(
    daemon: InProcDaemon, fake: FakeProviders, capsys: pytest.CaptureFixture[str]
) -> None:
    login("groq")
    code, out, _ = cli(capsys, "infer", "-m", "gpt-oss-20b", "-s", "be brief", "--json", "hi")
    doc = json.loads(out)
    assert code == 0
    assert doc["provider"] == "groq"
    assert doc["text"] == "4"
    assert fake.calls("groq")[-1].body["messages"][0] == {"role": "system", "content": "be brief"}
    code, out, _ = cli(capsys, "infer", "-m", "gpt-oss-20b", "--dry-run", "--json", "hi")
    route = json.loads(out)
    assert route["outcome"] == "place"
    assert route["chosen"]["provider"] == "groq"
    code, out, _err = cli(capsys, "infer", "-m", "gpt-oss-20b", "--dry-run", "hi")
    assert "→ groq: gpt-oss-20b" in out
    code, out, _ = cli(capsys, "infer", "--json", "hi")
    assert code == 2
    assert json.loads(out)["error"]["message"] == "--model is required"
    code, out, _ = cli(capsys, "infer", "-m", "gemini-3.5-flash", "--json", "hi")
    assert code == 1
    assert json.loads(out)["error"]["code"] == "auth_required"
    # a dry run that nothing can take exits 12, like `gpu route`
    code, out, _ = cli(capsys, "infer", "-m", "gemini-3.5-flash", "--dry-run", "--json", "hi")
    assert code == 12
    assert json.loads(out)["outcome"] == "no_fit"


def test_gpu_infer_file_batch(
    daemon: InProcDaemon,
    fake: FakeProviders,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    login("groq", "cloudflare")
    evals = tmp_path / "evals.jsonl"
    evals.write_text(
        "\n".join(
            [
                '{"id": "q1", "prompt": "2+2?", "expected": "4"}',
                "# a comment",
                "",
                json.dumps(
                    {
                        "id": "q2",
                        "messages": [{"role": "user", "content": "hi"}],
                        "model": "llama-3.1-8b",
                    }
                ),
                "not json",
                '{"id": "q4", "prompt": "x", "max_tokens": 0}',
            ]
        )
    )
    fake.queue("groq", httpx.Response(429, json={}, headers={"retry-after": "30"}))
    results = tmp_path / "out" / "results.jsonl"
    code, out, _ = cli(capsys, "infer", "-m", "gpt-oss-20b", "-f", str(evals), "-o", str(results))
    assert code == 0
    assert "4 prompts: 2 answered (cloudflare 2), 2 failed; results in" in out
    rows = [json.loads(x) for x in results.read_text().splitlines()]
    assert [r["id"] for r in rows] == ["q1", "q2", "5", "q4"]
    assert rows[0]["ok"]
    assert rows[0]["meta"] == {"expected": "4"}
    assert rows[0]["fallbacks"][0]["provider"] == "groq"  # 429 there, answered elsewhere
    assert rows[1]["model_id"] == "@cf/meta/llama-3.1-8b-instruct-fp8-fast"
    assert rows[2]["error"]["message"].startswith("line 5: not valid JSON")
    assert "max_tokens" in rows[3]["error"]["message"]
    # --json: one record per line on stdout, then the summary
    code, out, _ = cli(capsys, "infer", "-m", "gpt-oss-20b", "-f", str(evals), "-n", "1", "--json")
    lines = [json.loads(x) for x in out.splitlines()]
    assert lines[0]["id"] == "q1"
    assert lines[-1]["summary"]["total"] == 1


def test_a_batch_stops_when_every_provider_is_used_up(
    daemon: InProcDaemon, fake: FakeProviders, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    login("groq")
    fake.queue("groq", httpx.Response(429, json={}, headers=groq_headers(0, reset="4h0m0s")))
    evals = tmp_path / "e.jsonl"
    evals.write_text("\n".join(json.dumps({"prompt": f"q{i}"}) for i in range(6)))
    _code, out, _ = cli(
        capsys, "infer", "-m", "gpt-oss-20b", "-p", "groq", "-f", str(evals), "--json"
    )
    summary = json.loads(out.splitlines()[-1])["summary"]
    assert (summary["failed"], summary["not_sent"]) == (3, 3)
    assert summary["stopped"]
    assert len(fake.calls("groq")) == 1  # the ledger kept the rest away from groq


def test_gpu_infer_list_providers_and_quota(
    daemon: InProcDaemon, capsys: pytest.CaptureFixture[str]
) -> None:
    login("groq")
    code, out, _ = cli(capsys, "infer", "--list")
    assert code == 0
    assert "gpu login cloudflare" in out
    assert "✓ key" in out
    code, out, _ = cli(capsys, "providers", "--json")
    doc = json.loads(out)
    assert code == 0
    names = {p["name"] for p in doc["providers"]}
    assert not names & {"modal", "paperspace", "saturn", "sagemaker_studio_lab"}
    lanes = doc["not_routed"]
    assert [m["name"] for m in lanes["manual"]] == ["sagemaker_studio_lab"]
    assert {v["name"] for v in lanes["verify_at_signup"]} == {"paperspace", "saturn"}
    modal = {x["name"]: x for x in lanes["excluded"]}["modal"]
    assert modal["quote"].startswith("Note that you must have a payment method on file")
    assert {v["name"] for v in doc["inference"]} >= {"groq", "cloudflare", "gemini", "hf"}
    code, out, _ = cli(capsys, "providers")
    assert "manual" in out
    assert "https://studiolab.sagemaker.aws/" in out
    assert "verify at signup" in out
    assert "modal" in out
    assert "needs a card" in out
    code, out, _ = cli(capsys, "quota", "--json")
    q = json.loads(out)
    assert "quota" in q
    assert {v["provider"] for v in q["inference"]} >= {"groq"}
    assert not any(x["provider"] == "modal" for x in q["quota"])


def test_gpu_login_stores_keys_after_a_check(
    gpu_home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from gpu_router.cli import infer as cli_infer

    fake = FakeProviders()
    monkeypatch.setattr(cli_infer, "verify_transport", fake.transport())
    key = KEYS["INFER_GROQ_API_KEY"]
    monkeypatch.setattr("sys.stdin", _Stdin(key + "\n"))
    code, out, err = cli(capsys, "login", "groq", "--stdin", "--json")
    assert code == 0
    assert json.loads(out) == {
        "provider": "groq",
        "stored": ["INFER_GROQ_API_KEY"],
        "verified": True,
        "note": None,
    }
    assert secrets.get_secret("INFER_GROQ_API_KEY") == key
    assert key not in out + err
    assert fake.calls()[0].url == "https://api.groq.com/openai/v1/models"
    # a rejected key stores nothing
    fake.queue("gemini", httpx.Response(401))
    monkeypatch.setattr("sys.stdin", _Stdin(KEYS["INFER_GEMINI_API_KEY"] + "\n"))
    code, out, _ = cli(capsys, "login", "gemini", "--stdin", "--json")
    assert code == 2
    assert "nothing stored" in json.loads(out)["error"]["message"]
    assert secrets.get_secret("INFER_GEMINI_API_KEY") is None
    # cloudflare needs the account id too; it lands in the URL of the check
    monkeypatch.setattr("sys.stdin", _Stdin(KEYS["INFER_CLOUDFLARE_API_TOKEN"] + "\n"))
    code, out, _ = cli(
        capsys, "login", "cloudflare", "--stdin", "--account-id", "acct0123456789abcdef", "--json"
    )
    assert code == 0, out
    assert secrets.get_secret("INFER_CLOUDFLARE_ACCOUNT_ID") == "acct0123456789abcdef"
    assert "/accounts/acct0123456789abcdef/ai/models/search" in fake.calls("cloudflare")[-1].url
    monkeypatch.setattr("sys.stdin", _Stdin("short\n"))
    code, out, _ = cli(capsys, "login", "groq", "--stdin", "--no-check", "--json")
    assert code == 2
    assert "too short" in json.loads(out)["error"]["message"]


class _Stdin:
    def __init__(self, text: str) -> None:
        self._lines = text.splitlines(keepends=True)

    def isatty(self) -> bool:
        return False

    def readline(self) -> str:
        return self._lines.pop(0) if self._lines else ""

    def read(self) -> str:
        out = "".join(self._lines)
        self._lines = []
        return out


# --------------------------------------------------------------------------- MCP


async def test_mcp_gpu_infer(daemon: InProcDaemon, fake: FakeProviders) -> None:
    from fastmcp import Client

    from gpu_router.mcp.server import build_server
    from tests.mcp.conftest import call, call_error

    login("groq")
    async with Client(build_server()) as mcp:
        doc = await call(mcp, "gpu_infer", model="gpt-oss-20b", prompt="2+2?", system="brief")
        assert (doc["provider"], doc["model_id"], doc["text"]) == (
            "groq",
            "openai/gpt-oss-20b",
            "4",
        )
        assert "untrusted" in doc["untrusted"]
        assert doc["route"].startswith("groq:")
        dry = await call(mcp, "gpu_infer", model="gpt-oss-20b", prompt="x", dry_run=True)
        assert dry["route"]["outcome"] == "place"
        assert len(fake.calls()) == 1
        msgs = await call(
            mcp, "gpu_infer", model="gpt-oss-20b", messages=[{"role": "user", "content": "hi"}]
        )
        assert msgs["provider"] == "groq"
        missing = await call_error(mcp, "gpu_infer", model="gemini-3.5-flash", prompt="x")
        assert missing["code"] == "auth_required"
        assert "gpu login gemini" in missing["hint"]
        bad = await call_error(mcp, "gpu_infer", model="gpt-oss-20b")
        assert bad["code"] == "invalid_request"


# --------------------------------------------------------------------------- shell


class _Host:
    def __init__(self, paths: Any, cwd: Path) -> None:
        self.paths = paths
        self.cwd = cwd

    def connect(self, note: Any) -> GpuClient:
        return GpuClient.from_env(self.paths)

    def output_width(self) -> int:
        return 120


def _plain(items: list[Any]) -> str:
    console = Console(width=120, record=True, color_system=None)
    for item in items:
        console.print(item)
    return console.export_text()


def test_shell_infer_and_login_hint(
    daemon: InProcDaemon, fake: FakeProviders, tmp_path: Path
) -> None:
    from gpu_router.shell.commands import HANDLERS, Ctx, lookup

    login("groq")
    assert lookup("/infer") is not None
    shown: list[Any] = []
    ctx = Ctx(host=_Host(daemon.paths, tmp_path), sink=shown.extend)  # type: ignore[arg-type]
    fake.queue(
        "groq",
        httpx.Response(200, json={"choices": [{"message": {"content": "four\x1b[31m"}}]}),
    )
    HANDLERS["infer"](ctx, ["-m", "gpt-oss-20b", "what", "is", "2+2?"])
    text = _plain(shown)
    assert "four" in text
    assert "\x1b" not in text
    assert "groq · openai/gpt-oss-20b" in text
    shown.clear()
    (tmp_path / "e.jsonl").write_text('{"id": "a", "prompt": "hi"}\n')
    HANDLERS["infer"](ctx, ["-m", "gpt-oss-20b", "--file", "e.jsonl", "-o", "r.jsonl"])
    assert "1 prompts: 1 answered (groq 1)" in _plain(shown)
    assert json.loads((tmp_path / "r.jsonl").read_text())["ok"]
    shown.clear()
    HANDLERS["login"](ctx, ["groq"])
    assert "gpu login groq" in _plain(shown)
    shown.clear()
    HANDLERS["providers"](ctx, [])
    out = _plain(shown)
    assert "sagemaker_studio_lab" in out
    assert "manual" in out
    assert "groq" in out
    ctx.close()
