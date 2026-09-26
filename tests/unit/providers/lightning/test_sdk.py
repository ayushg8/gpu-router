"""sdk.py: the bounded runner (process-group kill), result parsing, bridge failure modes,
interpreter resolution and the kind -> taxonomy table."""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from gpu_router.errors import (
    AuthRequired,
    InvalidJob,
    NotFound,
    Permanent,
    QuotaExhausted,
    RateLimited,
    Unavailable,
)
from gpu_router.providers.lightning import sdk
from gpu_router.providers.lightning.sdk import (
    MARKER,
    CallTimeout,
    ProcResult,
    SdkBridge,
    SubprocessRunner,
    parse_result,
    resolve_interpreter,
    to_error,
)


def _line(doc: dict[str, Any]) -> str:
    return MARKER + base64.b64encode(json.dumps(doc).encode()).decode()


def test_parse_result_takes_the_last_marker_line() -> None:
    out = "\n".join(["noise", _line({"ok": True, "result": {"a": 1}}), _line({"ok": False})])
    assert parse_result(out) == {"ok": False}
    assert parse_result("no marker here") is None
    assert parse_result(MARKER + "!!!not base64") is None


def test_a_timeout_kills_the_whole_process_group(tmp_path: Path) -> None:
    pidfile = tmp_path / "child.pid"
    script = (
        "import subprocess, sys, time\n"
        f"p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open({str(pidfile)!r}, 'w').write(str(p.pid))\n"
        "time.sleep(60)\n"
    )
    start = time.monotonic()
    with pytest.raises(CallTimeout):
        SubprocessRunner()([sys.executable, "-c", script], stdin="", timeout=1.5, env=os.environ)
    assert time.monotonic() - start < 10
    pid = int(pidfile.read_text())
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        pytest.fail("the grandchild survived the timeout")


def test_the_runner_passes_stdin_and_env_and_never_raises_for_exit_codes() -> None:
    code = "import os, sys; print(sys.stdin.read() + os.environ['X_T']); sys.exit(3)"
    res = SubprocessRunner()(
        [sys.executable, "-c", code], stdin="in-", timeout=20, env={"X_T": "env"}
    )
    assert (res.returncode, res.stdout.strip()) == (3, "in-env")


def test_a_missing_interpreter_is_auth_required() -> None:
    with pytest.raises(sdk.SdkMissing):
        SubprocessRunner()(["/no/such/python"], stdin="", timeout=5, env={})


class _Scripted:
    def __init__(self, result: ProcResult | Exception) -> None:
        self.result = result
        self.argv: list[str] = []
        self.env: dict[str, str] = {}
        self.stdin = ""

    def __call__(self, argv: Any, *, stdin: str, timeout: float, env: Any) -> ProcResult:
        self.argv, self.env, self.stdin = list(argv), dict(env), stdin
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _bridge(runner: Any, tmp_path: Path, prefix: list[str] | None = ["py"]) -> SdkBridge:  # noqa: B006
    return SdkBridge(
        "lightning",
        runner=runner,
        interpreter=lambda: prefix,
        env_factory=lambda: {"LIGHTNING_USER_ID": "u-123456", "LIGHTNING_API_KEY": "k-123456"},
        home=lambda: tmp_path / "home",
    )


def test_the_bridge_sends_the_request_on_stdin_and_creds_in_env_only(tmp_path: Path) -> None:
    runner = _Scripted(ProcResult((), 0, _line({"ok": True, "result": {"x": 1}}), ""))
    res = _bridge(runner, tmp_path).call("whoami", {"teamspace": "a/b"}, timeout=5)
    assert res.ok
    assert res.result == {"x": 1}
    assert runner.argv[:3] == ["py", "-s", "-u"]
    assert runner.argv[3].endswith("driver.py")
    assert json.loads(runner.stdin) == {"op": "whoami", "params": {"teamspace": "a/b"}}
    assert "k-123456" not in " ".join(runner.argv)
    assert runner.env["LIGHTNING_API_KEY"] == "k-123456"


@pytest.mark.parametrize(
    ("result", "error", "text"),
    [
        (CallTimeout("t"), Unavailable, "did not answer"),
        (sdk.SdkMissing("py"), AuthRequired, "could not run"),
        (
            ProcResult((), 1, "", "ModuleNotFoundError: No module named 'lightning_sdk'"),
            AuthRequired,
            "not installed",
        ),
        (ProcResult((), 1, "", "Traceback ... boom"), Unavailable, "without a result"),
    ],
)
def test_bridge_failures(
    tmp_path: Path, result: ProcResult | Exception, error: type[Exception], text: str
) -> None:
    with pytest.raises(error, match=text):
        _bridge(_Scripted(result), tmp_path).call("whoami", {}, timeout=5)


def test_no_interpreter_at_all_is_auth_required_with_the_install_hint(tmp_path: Path) -> None:
    with pytest.raises(AuthRequired) as info:
        _bridge(_Scripted(ProcResult((), 0, "", "")), tmp_path, prefix=None).call(
            "x", {}, timeout=1
        )
    assert "uv tool install lightning-sdk" in (info.value.hint or "")


def test_a_classified_failure_comes_back_redacted(tmp_path: Path) -> None:
    from gpu_router import secrets

    secrets.register_for_redaction("k-123456-leaky")
    doc = {
        "ok": False,
        "kind": "rate",
        "error": "429 for k-123456-leaky",
        "stage": "run",
        "status": 429,
    }
    res = _bridge(_Scripted(ProcResult((), 1, _line(doc), "")), tmp_path).call(
        "submit", {}, timeout=5
    )
    assert (res.ok, res.kind, res.stage, res.status) == (False, "rate", "run", 429)
    assert "k-123456-leaky" not in (res.error or "")


def test_interpreter_resolution_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert resolve_interpreter(python="/x/python") == ["/x/python"]
    tool = tmp_path / "tools" / "lightning-sdk" / "bin" / "python"
    tool.parent.mkdir(parents=True)
    tool.write_text("")
    assert resolve_interpreter(tool_python=tool) == [str(tool)]
    missing = tmp_path / "nope"
    cmd = resolve_interpreter(tool_python=missing, uv="/bin/uv")
    assert cmd == [
        "/bin/uv",
        "run",
        "--no-project",
        "--quiet",
        "--python",
        "3.12",
        "--with",
        f"lightning-sdk=={sdk.DEFAULT_SDK_VERSION}",
        "python",
    ]
    assert resolve_interpreter(tool_python=missing, uv="/bin/uv", sdk_version="latest")[-2] == (
        "lightning-sdk"
    )
    monkeypatch.setattr(sdk, "find_uv", lambda configured=None: None)
    assert resolve_interpreter(tool_python=missing) is None


def test_uv_tool_dir_is_respected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UV_TOOL_DIR", str(tmp_path))
    assert sdk._uv_tool_python() == tmp_path / "lightning-sdk" / "bin" / "python"


@pytest.mark.parametrize(
    ("kind", "cls"),
    [
        ("auth", AuthRequired),
        ("verify", AuthRequired),
        ("config", AuthRequired),
        ("not_found", NotFound),
        ("rate", RateLimited),
        ("quota", QuotaExhausted),
        ("invalid", InvalidJob),
        ("permanent", Permanent),
        ("unavailable", Unavailable),
        ("sdk", Unavailable),
        (None, Unavailable),
    ],
)
def test_kinds_map_to_the_taxonomy(kind: str | None, cls: type[Exception]) -> None:
    err = to_error("lightning", "op", kind, "message", resets_at=123.0)
    assert type(err) is cls
    assert err.provider == "lightning"
    if cls is QuotaExhausted:
        assert err.resets_at == 123.0  # type: ignore[attr-defined]
    if kind == "verify":
        assert "phone" in (err.hint or "")
