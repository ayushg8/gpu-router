"""Parsers for kaggle CLI 2.2.4 output (strings copied from the CLI source / live calls)."""

from __future__ import annotations

import json

import pytest

from gpu_router.providers.kaggle import parse
from gpu_router.providers.kaggle.parse import KernelStatus


@pytest.mark.parametrize(
    ("text", "status", "failure"),
    [
        ('me/k has status "KernelWorkerStatus.RUNNING"\n', KernelStatus.RUNNING, None),
        ('me/k has status "KernelWorkerStatus.COMPLETE"\n', KernelStatus.COMPLETE, None),
        ('me/k has status "QUEUED"\n', KernelStatus.QUEUED, None),
        (
            'me/k has status "KernelWorkerStatus.ERROR"\nFailure message: "Out of time"\n',
            KernelStatus.ERROR,
            "Out of time",
        ),
        (
            'me/k has status "KernelWorkerStatus.CANCEL_ACKNOWLEDGED"\n',
            KernelStatus.CANCEL_ACKNOWLEDGED,
            None,
        ),
    ],
)
def test_parse_status(text: str, status: KernelStatus, failure: str | None) -> None:
    out = parse.parse_status(text)
    assert out is not None
    assert out.status is status
    assert out.failure_message == failure


def test_parse_status_rejects_other_text() -> None:
    assert parse.parse_status("Cannot access kernel 'me/k'") is None
    assert parse.parse_status('me/k has status "KernelWorkerStatus.EXPLODED"') is None


def test_status_flags() -> None:
    assert KernelStatus.COMPLETE.finished
    assert KernelStatus.ERROR.finished
    assert not KernelStatus.RUNNING.finished
    assert KernelStatus.CANCEL_REQUESTED.cancelled
    assert not KernelStatus.ERROR.cancelled


def test_parse_push_success_and_warnings() -> None:
    text = (
        "Your kernel title does not resolve to the specified id. This may result in ...\n"
        "Kernel version 3 successfully pushed.  Please check progress at "
        "https://www.kaggle.com/code/me/gpu-router-0123456789ab-1\n"
    )
    out = parse.parse_push(text)
    assert out is not None
    assert out.ok
    assert out.version == 3
    assert out.url == "https://www.kaggle.com/code/me/gpu-router-0123456789ab-1"
    assert out.warnings
    assert out.warnings[0].startswith("Your kernel title")


def test_parse_push_without_version_number() -> None:
    out = parse.parse_push(
        "Kernel version successfully pushed.  Please check progress at https://x/y\n"
    )
    assert out is not None
    assert out.ok
    assert out.version is None
    assert out.url == "https://x/y"


def test_parse_push_error_and_unknown() -> None:
    out = parse.parse_push("Kernel push error: Maximum weekly GPU quota reached\n")
    assert out is not None
    assert not out.ok
    assert out.error == "Maximum weekly GPU quota reached"
    assert parse.parse_push("Traceback (most recent call last):\n  boom\n") is None


def test_parse_quota_live_shape() -> None:
    text = json.dumps(
        [
            {
                "resource": "GPU",
                "used": "1.25h",
                "remaining": "28.75h",
                "total": "30.00h",
                "refreshAt": "2026-09-26T00:00:00",
            },
            {
                "resource": "TPU",
                "used": "0.00h",
                "remaining": "20.00h",
                "total": "20.00h",
                "refreshAt": "2026-09-26T00:00:00",
            },
        ],
        indent=2,
    )
    rows = parse.parse_quota(text)
    gpu = rows["GPU"]
    assert gpu.used_h == pytest.approx(1.25)
    assert gpu.total_h == pytest.approx(30)
    assert gpu.remaining_h == pytest.approx(28.75)
    assert gpu.refresh_at == pytest.approx(1790380800.0)  # 2026-09-26T00:00:00Z
    assert rows["TPU"].total_h == pytest.approx(20)


def test_parse_quota_computes_remaining_and_tolerates_prefix() -> None:
    rows = parse.parse_quota(
        'Warning: something\n[{"resource":"GPU","used":"2h","total":"30h","refreshAt":""}]'
    )
    assert rows["GPU"].remaining_h == pytest.approx(28)
    assert rows["GPU"].refresh_at is None


@pytest.mark.parametrize("bad", ["", "No quota information available", '{"a": 1}'])
def test_parse_quota_rejects_garbage(bad: str) -> None:
    with pytest.raises(ValueError, match=r"kaggle quota|JSON|Expecting"):
        parse.parse_quota(bad)


def test_parse_log_json_events() -> None:
    events = [
        {"stream_name": "stdout", "time": 0.7, "data": "first line\n"},
        {"stream_name": "stderr", "time": 0.8, "data": "partial "},
        {"stream_name": "stderr", "time": 0.9, "data": "joined\nsecond\n"},
        {"stream_name": "stdout", "time": 1.0, "data": "10%\r50%\r100%\n"},
    ]
    assert parse.parse_log(json.dumps(events) + "\n") == [
        "first line",
        "partial joined",
        "second",
        "100%",
    ]


def test_trim_post_run_drops_kaggle_nbconvert_tail() -> None:
    lines = [
        "gpu-router: running gpucheck.py",
        "wrote gpucheck.json",
        '::gpu:: {"t":"exit","code":0}',
        "/usr/local/lib/python3.12/dist-packages/mistune.py:435: SyntaxWarning: invalid escape",
        "[NbConvertApp] Writing 315398 bytes to __results__.html",
    ]
    assert parse.trim_post_run(lines) == lines[:3]
    # the last exit line wins; a log without one is kept whole
    two = ['::gpu:: {"t":"exit","code":1}', "retry", '::gpu:: {"t":"exit","code":0}', "tail"]
    assert parse.trim_post_run(two) == two[:3]
    assert parse.trim_post_run(lines[:2]) == lines[:2]
    assert parse.trim_post_run([]) == []


def test_parse_log_plain_text_and_empty() -> None:
    assert parse.parse_log("a\r\nb\n") == ["a", "b"]
    assert parse.parse_log("\n") == []
    assert parse.parse_log("") == []


def test_last_exit_code_takes_the_last_exit_line() -> None:
    lines = [
        '::gpu:: {"t":"hello","v":1,"runner":"bootstrap/0.2"}',
        '::gpu:: {"t":"exit","code":0}',
        "gpu-router: runner ended without an exit file (exit 137)",
        '::gpu:: {"t":"exit","code":137}',
    ]
    assert parse.last_exit_code(lines) == 137
    assert parse.last_exit_code(lines[:2]) == 0
    assert parse.last_exit_code(["::gpu:: not json", "plain"]) is None


def test_parse_config_username_and_version() -> None:
    text = "Configuration values from /x/.kaggle\n- username: someone\n- auth_method: LEGACY\n"
    assert parse.parse_config_username(text) == "someone"
    assert parse.parse_config_username("- username: None\n") is None
    assert parse.parse_config_username("nothing") is None
    assert parse.parse_version("Kaggle CLI 2.2.4\n") == "2.2.4"
    assert parse.parse_version("python: command not found") is None
