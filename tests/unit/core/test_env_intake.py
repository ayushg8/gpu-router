"""D39 (phase-3 review): new specs are held to a stricter secret-env rule than stored ones.

JobSpec.env is published verbatim into Kaggle kernel source and Colab exec code, so
`secret_env_problem` refuses `*_KEY` / `*_PASS` / webhook / DSN / credential-URL names and
token-shaped values at every intake (daemon submit + dry route, CLI). The JobSpec
validator itself keeps the original rule so specs stored before D39 still load.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gpu_router.errors import InvalidSpec
from gpu_router.jobspec import Flags, GpuYamlError, build_spec
from gpu_router.models import JobSpec, secret_env_problem

REJECTED_NAMES = [
    "KAGGLE_KEY",
    "WANDB_KEY",
    "OPENAI_KEY",
    "HF_KEY",
    "SSH_KEY",
    "DB_PASS",
    "DATABASE_URL",
    "REDIS_URL",
    "SLACK_WEBHOOK_URL",
    "SENTRY_DSN",
    "SESSION_COOKIE",
    "API_TOKEN",  # the original rule still applies
]
ACCEPTED = {
    "WANDB_MODE": "offline",
    "PWD": "/x",
    "MONKEY": "1",
    "HF_ENDPOINT": "https://huggingface.co",
    "WANDB_BASE_URL": "https://api.wandb.ai",
    "KEYBOARD": "us",
    "PASSES": "3",
}


@pytest.mark.parametrize("name", REJECTED_NAMES)
def test_secret_looking_names_are_refused(name: str) -> None:
    problem = secret_env_problem({name: "x"})
    assert problem is not None
    assert repr(name) in problem
    assert "gpu secrets set" in problem


@pytest.mark.parametrize(
    "value",
    [
        "hf_" + "a" * 34,
        "sk-proj-" + "b" * 30,
        "ghp_" + "c" * 36,
        "--token=abc123",
        "AKIAIOSFODNN7EXAMPLE",
    ],
)
def test_token_shaped_values_are_refused_without_echoing_them(value: str) -> None:
    problem = secret_env_problem({"HF": value})
    assert problem is not None
    assert value not in problem


def test_ordinary_env_is_accepted() -> None:
    assert secret_env_problem(ACCEPTED) is None


def test_stored_specs_with_newly_refused_names_still_load() -> None:
    spec = JobSpec(project_dir="/p", script="t.py", env={"WANDB_KEY": "x"})
    again = JobSpec.model_validate_json(spec.model_dump_json())  # what Store does on load
    assert again.env == {"WANDB_KEY": "x"}


# --------------------------------------------------------------------------- CLI


@pytest.fixture
def proj(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    p = tmp_path / "proj"
    p.mkdir()
    (p / "train.py").write_text("print(1)\n")
    monkeypatch.chdir(p)
    return p


def test_gpu_yaml_env_is_refused_with_its_line(proj: Path) -> None:
    (proj / "gpu.yaml").write_text("script: train.py\nenv:\n  KAGGLE_KEY: abc\n")
    with pytest.raises(GpuYamlError) as info:
        build_spec(Flags(), cwd=proj, project_root=proj)
    assert "looks like a secret" in info.value.message
    assert "gpu.yaml:2" in info.value.message
    assert "abc" not in info.value.message


def test_env_flag_with_a_token_value_is_refused(proj: Path) -> None:
    token = "hf_" + "z" * 34
    with pytest.raises(InvalidSpec) as info:
        build_spec(Flags(script="train.py", env={"HF": token}), cwd=proj, project_root=proj)
    assert info.value.message.startswith("--env:")
    assert token not in info.value.message
