"""Colab adapter pieces without a VM: remote script builder/parser, log splitting, CLI
failure classification, redaction and the bounded CLI runner."""

from __future__ import annotations

import base64
import json
import sys
import time
from pathlib import Path

import pytest

from gpu_router.errors import AuthRequired, QuotaExhausted, RateLimited, Unavailable
from gpu_router.providers.colab import remote
from gpu_router.providers.colab.cli import (
    CliResult,
    ColabCli,
    classify,
    redact,
    resolve_cli,
    session_gone,
)


def _res(stdout: str = "", stderr: str = "", code: int | None = 1) -> CliResult:
    return CliResult(("x",), code, stdout, stderr)


# --------------------------------------------------------------------------- remote scripts


@pytest.mark.parametrize("name", sorted(remote.SCRIPTS))
def test_every_remote_script_compiles_for_python38(name: str) -> None:
    source = remote.build(name, {"run_dir": "/content/gr/x", "path": "/x", "outputs": True})
    import ast

    ast.parse(source, filename=f"<{name}>", feature_version=(3, 8))
    compile(source, f"<{name}>", "exec", dont_inherit=True)
    assert source.rstrip().endswith("))")
    assert "_gr_run(" in source


def test_parse_result_takes_the_last_marker_and_ignores_noise() -> None:
    def marker(obj: object) -> str:
        return remote.MARKER + base64.b64encode(json.dumps(obj).encode()).decode()

    out = "\n".join(
        [
            "A new version of Colab CLI is available",
            marker({"ok": True, "result": {"n": 1}}),
            "noise " + marker({"ok": True, "result": {"n": 2}}),
        ]
    )
    assert remote.parse_result(out) == {"n": 2}
    with pytest.raises(remote.RemoteScriptError, match="boom"):
        remote.parse_result(marker({"ok": False, "error": "RuntimeError: boom"}))
    with pytest.raises(ValueError, match="no result marker"):
        remote.parse_result("no marker here")
    with pytest.raises(ValueError, match="unreadable"):
        remote.parse_result(remote.MARKER + "!!notbase64!!")


def test_scripts_run_locally_and_report(tmp_path: Path) -> None:
    import subprocess

    run = tmp_path / "run"
    source = remote.build("poll", {"run_dir": str(run), "pack": True})
    script = tmp_path / "poll.py"
    script.write_text(source)
    out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True).stdout
    assert remote.parse_result(out) == {"run_dir": False}
    run.mkdir()
    (run / "EXIT").write_text("0\n")
    (run / "job.log").write_text("a\nb\n")
    (run / "work" / "outputs").mkdir(parents=True)
    (run / "work" / "outputs" / "m.txt").write_text("model")
    (run / "launched.json").write_text(json.dumps({"pid": 999999, "started_at": 1.0}))
    out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True).stdout
    res = remote.parse_result(out)
    assert res["exit"] == 0
    assert res["alive"] is False
    assert res["packed"]["outputs_files"] == 1
    assert (run / "outputs.tar.gz").is_file()
    assert (run / "job.log.gz").is_file()


# --------------------------------------------------------------------------- log splitting


def _lines(data: bytes, final: bool, max_line: int = remote.MAX_LINE_BYTES) -> list[str]:
    return [line for line, _ in remote.split_log_bytes(data, final=final, max_line=max_line)]


def test_split_holds_back_a_growing_fragment_until_final() -> None:
    assert _lines(b"a\nb\ncc", final=False) == ["a", "b"]
    assert _lines(b"a\nb\ncc", final=True) == ["a", "b", "cc"]
    assert _lines(b"a\r\n\n", final=False) == ["a", ""]
    assert remote.split_log_bytes(b"a\nb\ncc", final=False)[-1][1] == 4
    assert _lines(b"", final=True) == []
    assert _lines(b"\xff\xfeok\n", final=False) == ["��ok"]


def test_long_lines_are_cut_at_the_same_offsets_however_they_are_read() -> None:
    data = b"short\n" + b"x" * 25 + b"\n" + b"y" * 7 + b"\nend"
    whole = _lines(data, final=True, max_line=10)
    assert whole == ["short", "x" * 10, "x" * 10, "x" * 5, "y" * 7, "end"]
    # read in small windows, resuming at the consumed offset each time
    got: list[str] = []
    off = 0
    while off < len(data):
        window = data[off : off + 12]
        at_end = off + len(window) >= len(data)
        pairs = remote.split_log_bytes(window, final=at_end, max_line=10)
        got.extend(line for line, _ in pairs)
        if not pairs:
            break
        off += pairs[-1][1]
    assert got == whole


# --------------------------------------------------------------------------- classification


def test_gpu_refusal_is_quota_exhausted_with_a_reset_time() -> None:
    err = classify(
        _res(stderr="[colab] Backend rejected accelerator 'T4'. You may not have quota"),
        provider="colab",
        op="new",
        gpu="T4",
        now=1000.0,
    )
    assert isinstance(err, QuotaExhausted)
    assert err.resets_at == 1000.0 + 24 * 3600
    assert err.provider == "colab"


@pytest.mark.parametrize(
    ("text", "cls"),
    [
        ("[colab] Allocation refused (precondition failed). too many", Unavailable),
        ("Keep-alive pre-flight failed: your credentials are missing an OAuth scope", AuthRequired),
        ("No valid default credentials found. To authenticate, run:", AuthRequired),
        ("google.auth.exceptions.RefreshError: ('invalid_grant: Bad Request')", AuthRequired),
        ("requests.exceptions.ConnectionError: Max retries exceeded", Unavailable),
        ("429 Client Error: Too Many Requests", RateLimited),
        ("something nobody expected", Unavailable),
    ],
)
def test_classify_table(text: str, cls: type) -> None:
    err = classify(_res(stderr=text), provider="colab", op="new")
    assert type(err) is cls
    assert err.message


def test_timeout_and_missing_cli() -> None:
    assert isinstance(classify(_res(code=None), provider="colab", op="exec"), Unavailable)
    missing = classify(_res(code=127), provider="colab", op="new")
    assert isinstance(missing, AuthRequired)
    assert missing.hint is not None
    assert "uv tool install" in missing.hint


def test_session_gone_detection() -> None:
    assert session_gone(_res("[colab] Session 'gr-x' appears to be lost (404/401). Cleaning up."))
    assert session_gone(_res("[colab] Session 'gr-x' not found."))
    assert not session_gone(_res("[colab] Upload failed: disk full"))


def test_cli_output_is_redacted_before_it_can_leak() -> None:
    text = (
        "[colab] Upload failed: 404 Client Error: Not Found for url: https://h/api/contents/x"
        "?authuser=0&colab-runtime-proxy-token=SECRETTOKEN123 Authorization: Bearer ya29.abcDEF"
        ' {"token": "tok-999"}'
    )
    out = redact(text)
    for secret in ("SECRETTOKEN123", "ya29.abcDEF", "tok-999"):
        assert secret not in out
    err = classify(_res(stdout=text), provider="colab", op="upload")
    assert "SECRETTOKEN123" not in err.message


def test_update_noise_is_dropped() -> None:
    res = _res(stdout="A new version of Colab CLI is available\nreal line\n", code=0)
    assert res.text == "real line"


# --------------------------------------------------------------------------- runner


def test_runner_kills_a_hung_cli_at_the_timeout(tmp_path: Path) -> None:
    cli = ColabCli([sys.executable, "-c", "import time; time.sleep(60)", "--"], tmp_path / "s.json")
    t0 = time.monotonic()
    res = cli.run(["new"], timeout=1.0)
    assert res.timed_out
    assert time.monotonic() - t0 < 10


def test_runner_reports_a_missing_binary(tmp_path: Path) -> None:
    res = ColabCli([str(tmp_path / "nope")], tmp_path / "s.json").run(["sessions"], timeout=5)
    assert res.returncode == 127


def test_resolve_cli_prefers_the_setting() -> None:
    assert resolve_cli("/opt/colab") == ["/opt/colab"]
    assert resolve_cli(["python", "sim.py"]) == ["python", "sim.py"]
