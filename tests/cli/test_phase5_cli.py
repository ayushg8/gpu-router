"""Phase 5 through the CLI against a real daemon: `gpu policy`, `gpu route --smoke`,
gpu.yaml `smoke:`, `gpu quota --refresh` and the ledger's note column."""

from __future__ import annotations

import json
from typing import Any

import yaml

from gpu_router.cli import exitcodes
from tests.cli.conftest import Cli


def sub_json(cli: Cli, *args: str) -> tuple[Any, int]:
    """`gpu policy <sub> --json ...` (Cli.json puts --json right after the first word)."""
    res = cli(*args[:2], "--json", *args[2:])
    lines = [ln for ln in res.stdout.splitlines() if ln.strip()]
    assert lines, res.stderr
    return json.loads(lines[-1]), res.exit_code


def test_policy_show(cli: Cli) -> None:
    human = cli("policy")
    assert human.exit_code == 0, human.stderr
    assert "agent jobs" in human.stdout
    assert "your jobs" in human.stdout
    assert "runs automatically  up to 1h\n" in human.stdout  # 7b: modal dropped, no "except"
    data, res = cli.json("policy")
    assert res.exit_code == 0
    assert data["editable"] is True
    assert data["policy"]["agent"]["auto_max_hours"] == 1.0
    assert data["policy"]["user"]["auto_max_hours"] is None
    show, _ = sub_json(cli, "policy", "show")
    assert show == data


def test_policy_set_and_reset(cli: Cli) -> None:
    try:
        doc, code = sub_json(cli, "policy", "set", "agent.auto_max_hours", "2")
        assert code == 0
        assert doc["policy"]["agent"]["auto_max_hours"] == 2.0
        saved = yaml.safe_load((cli.home / "config.yaml").read_text())
        assert saved["policy"]["agent"]["auto_max_hours"] == 2.0

        human = cli("policy", "set", "max_quota_share", "75%")
        assert human.exit_code == 0, human.stderr
        assert "max_quota_share = 75%" in human.stdout
        assert "over 75% of a provider's quota left" in human.stdout

        bad = cli("policy", "set", "agent.nope", "1")
        assert bad.exit_code == exitcodes.USAGE
        assert "unknown policy rule" in bad.stderr
        err, code = sub_json(cli, "policy", "set", "agent.auto_max_hours", "lots")
        assert code == exitcodes.USAGE
        assert err["error"]["code"] == "invalid_request"
    finally:
        reset, code = sub_json(cli, "policy", "reset")
        assert code == 0
        assert reset["policy"] == reset["defaults"]


def test_route_smoke_flag_and_gpu_yaml(cli: Cli) -> None:
    data, res = cli.json("route", "--smoke")
    assert res.exit_code == 0
    assert data["spec"]["smoke"] is True
    assert data["route"]["smoke"] is True
    assert data["route"]["router"] == "scoring"
    human = cli("route", "--smoke")
    assert "smoke test" in human.stdout
    cli.write_yaml("version: 1\nscript: train.py\nsmoke: true\n")
    data, _ = cli.json("route")
    assert data["spec"]["smoke"] is True
    cli.write_yaml("version: 1\nscript: train.py\nsmoke: maybe\n")
    bad = cli("route")
    assert bad.exit_code == exitcodes.USAGE
    assert "smoke" in bad.stderr


def test_route_shows_the_runtime_assumption(cli: Cli) -> None:
    human = cli("route", "--hours", "6")
    assert human.exit_code == 0
    assert "6h runtime" in human.stdout
    assert "candidates, best first" in human.stdout
    estimated = cli("route")
    assert "assuming ~1h (estimated" in estimated.stdout


def test_quota_refresh_and_note(cli: Cli) -> None:
    data, res = cli.json("quota", "--refresh")
    assert res.exit_code == 0
    rows = {q["provider"]: q for q in data["quota"]}
    assert rows["fake"]["detail"]["basis"] in {"live", "live+history"}
    human = cli("quota")
    assert "how" in human.stdout
    assert "live from fake" in human.stdout


def test_route_takes_data_and_include_like_run(cli: Cli) -> None:
    """2026-10-04 field test: `gpu route` had no --data / --include, so the dry run could
    not show how a job with a dataset would be routed."""
    (cli.project / "rows").mkdir()
    (cli.project / "rows" / "a.csv").write_text("a\n")
    data, res = cli.json("route", "--data", "rows=rows", "--include", "extra/", "train.py")
    # the test daemon's fakes are remote and there is no HF storage: the data reaches none
    assert res.exit_code == exitcodes.NO_FIT
    assert "cannot receive data= without Hugging Face storage" in data["route"]["reason"]
    (ref,) = data["spec"]["data"]
    assert (ref["mount"], ref["path"]) == ("rows", str((cli.project / "rows").resolve()))
    assert data["spec"]["include"] == ["extra/"]
