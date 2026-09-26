"""kaggle CLI wrapper: the subprocess runner, KaggleCli, and error classification."""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from gpu_router import secrets
from gpu_router.errors import AuthRequired, NotFound, RateLimited, Unavailable
from gpu_router.providers.kaggle.cli import (
    CliMissing,
    CliResult,
    CliTimeout,
    KaggleCli,
    SubprocessRunner,
    classify,
    find_executable,
    snippet,
)


def _res(stdout: str = "", stderr: str = "", rc: int = 1) -> CliResult:
    return CliResult(("kaggle",), rc, stdout, stderr)


def test_subprocess_runner_captures_output_and_exit_code(tmp_path: Path) -> None:
    r = SubprocessRunner()(
        [
            sys.executable,
            "-c",
            "import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)",
        ],
        timeout=30,
        env={"PATH": "/usr/bin:/bin"},
        cwd=tmp_path,
    )
    assert r.returncode == 3
    assert r.stdout == "out\n"
    assert r.stderr == "err\n"
    assert not r.ok
    assert "out" in r.output
    assert "err" in r.output


def test_subprocess_runner_timeout_and_missing() -> None:
    with pytest.raises(CliTimeout):
        SubprocessRunner()(
            [sys.executable, "-c", "import time; time.sleep(5)"], timeout=0.3, env={}
        )
    with pytest.raises(CliMissing):
        SubprocessRunner()(["/nonexistent/kaggle-binary"], timeout=5, env={})


class Recorder:
    def __init__(self, result: CliResult | None = None, exc: Exception | None = None) -> None:
        self.result = result or _res("ok\n", rc=0)
        self.exc = exc
        self.calls: list[tuple[list[str], float, dict[str, str]]] = []

    def __call__(
        self,
        argv: Sequence[str],
        *,
        timeout: float,
        env: Mapping[str, str],
        cwd: Path | None = None,
    ) -> CliResult:
        self.calls.append((list(argv), timeout, dict(env)))
        if self.exc is not None:
            raise self.exc
        return self.result


def test_kaggle_cli_builds_argv_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KAGGLE_KEY", "inherited-should-be-dropped")
    rec = Recorder()
    cli = KaggleCli(
        "kaggle",
        runner=rec,
        executable=lambda: "/opt/kaggle",
        env_factory=lambda: (("KAGGLE_KEY",), {"KAGGLE_USERNAME": "u"}),
    )
    res = cli.run(["quota", "--format", "json"], timeout=12)
    assert res.ok
    argv, timeout, env = rec.calls[0]
    assert argv == ["/opt/kaggle", "-W", "quota", "--format", "json"]
    assert timeout == 12
    assert "KAGGLE_KEY" not in env
    assert env["KAGGLE_USERNAME"] == "u"
    assert env["PYTHONIOENCODING"] == "utf-8"


def test_kaggle_cli_missing_and_timeout() -> None:
    cli = KaggleCli("kaggle", runner=Recorder(), executable=lambda: None)
    with pytest.raises(AuthRequired, match="not installed") as info:
        cli.run(["--version"], timeout=5)
    assert "uv tool install kaggle" in (info.value.hint or "")
    cli = KaggleCli("kaggle", runner=Recorder(exc=CliMissing("x")), executable=lambda: "/x")
    with pytest.raises(AuthRequired, match="not found"):
        cli.run(["--version"], timeout=5)
    cli = KaggleCli("kaggle", runner=Recorder(exc=CliTimeout("slow")), executable=lambda: "/x")
    with pytest.raises(Unavailable, match="did not answer within 5s"):
        cli.run(["kernels", "status", "a/b"], timeout=5)


@pytest.mark.parametrize(
    ("result", "kind"),
    [
        (_res("Authentication required to call the Kaggle API.\n"), AuthRequired),
        (_res(stderr="401 Client Error: Unauthorized for url"), AuthRequired),
        (_res(stderr="429 Client Error: Too Many Requests for url"), RateLimited),
        (
            _res(stderr="Cannot access kernel 'a/b' (Permission 'kernels.get' was denied)."),
            NotFound,
        ),
        (_res(stderr="404 Client Error: Not Found for url: https://x"), NotFound),
        (_res(stderr="503 Server Error: Service Unavailable for url"), Unavailable),
        (_res(stderr="requests.exceptions.ConnectionError: Max retries exceeded"), Unavailable),
        (_res(stderr="something nobody expected"), Unavailable),
        (_res(rc=1), Unavailable),
    ],
)
def test_classify(result: CliResult, kind: type[Exception]) -> None:
    err = classify("kaggle", "kernels status", result)
    assert type(err) is kind
    assert err.provider == "kaggle"
    assert err.message


def test_classify_rate_limit_has_retry_after() -> None:
    err = classify("kaggle", "push", _res(stderr="429 Client Error"))
    assert isinstance(err, RateLimited)
    assert err.retry_after == 60


def test_snippet_redacts_and_truncates() -> None:
    secrets.register_for_redaction("s3cr3t-value-123")
    text = "boom s3cr3t-value-123 " + "x" * 1000
    out = snippet(text)
    assert "s3cr3t-value-123" not in out
    assert len(out) <= 300
    assert out.endswith("...")
    assert snippet("  a \n b  ") == "a b"


def test_find_executable_prefers_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert find_executable("/custom/kaggle") == "/custom/kaggle"
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    assert find_executable() is None
    local = tmp_path / ".local" / "bin" / "kaggle"
    local.parent.mkdir(parents=True)
    local.write_text("#!/bin/sh\n")
    assert find_executable() == str(local)
