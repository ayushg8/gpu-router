"""`gpu run` argv split (app.split_run_argv, D23): gpu options before the script, a short
list of gpu-only long options after it, everything else to the script."""

from __future__ import annotations

from pathlib import Path

import pytest

from gpu_router.cli.app import split_run_argv


@pytest.mark.parametrize(
    ("argv", "click_args", "script_args", "clashes"),
    [
        # gpu options before the script, script args after
        (
            ["--vram", "8", "-p", "fake", "train.py", "--lr", "3"],
            ["--vram", "8", "-p", "fake", "train.py"],
            ["--lr", "3"],
            [],
        ),
        # review finding: -wd after the script was read as --wait --detach
        (["train.py", "-wd", "0.01", "--lr", "3"], ["train.py"], ["-wd", "0.01", "--lr", "3"], []),
        # -e 10 (epochs) after the script is the script's, noted as a clash
        (["train.py", "-e", "10"], ["train.py"], ["-e", "10"], ["-e"]),
        # common training flags that gpu also has go to the script
        (
            ["train.py", "--name", "exp1", "--gpu", "0"],
            ["train.py"],
            ["--name", "exp1", "--gpu", "0"],
            ["--name", "--gpu"],
        ),
        # -p stays gpu's (a non-provider value fails loudly); -d/-w would silently change
        # what runs, so they go to the script
        (["train.py", "--json", "-p", "fake"], ["train.py", "--json", "-p", "fake"], [], []),
        (["train.py", "-d", "cifar10"], ["train.py"], ["-d", "cifar10"], ["-d"]),
        # the spec's agent example: `gpu run train.py --json`
        (["train.py", "--json"], ["train.py", "--json"], [], []),
        # tail gpu options, including --opt=value
        (
            ["train.py", "--epochs", "3", "--vram=24", "--hours", "2", "--provider", "fake", "-x"],
            ["train.py", "--vram=24", "--hours", "2", "--provider", "fake"],
            ["--epochs", "3", "-x"],
            [],
        ),
        (
            ["train.py", "--detach", "--dry-run", "--wait"],
            ["train.py", "--detach", "--dry-run", "--wait"],
            [],
            [],
        ),
        # `--` ends gpu parsing: verbatim, even gpu-looking options
        (["train.py", "--", "--vram", "1", "--json"], ["train.py"], ["--vram", "1", "--json"], []),
        (["train.py", "--a", "--", "--json"], ["train.py"], ["--a", "--json"], []),
        # no script: an unknown option starts gpu.yaml's script args
        (["--epochs", "3", "--json"], ["--json"], ["--epochs", "3"], []),
        (["--vram", "8", "--epochs", "3"], ["--vram", "8"], ["--epochs", "3"], []),
        # `--` before any script: first token is the script unless it is an option
        (["--", "train.py", "--vram", "2"], ["train.py"], ["--vram", "2"], []),
        (["-d", "--", "--epochs", "2"], ["-d"], ["--epochs", "2"], []),
        # short clusters before the script are gpu's; attached and separate values
        (["-wd", "-pfake", "train.py"], ["-wd", "-pfake", "train.py"], [], []),
        (["-dp", "fake", "train.py"], ["-dp", "fake", "train.py"], [], []),
        (["-C", "/x", "-e", "A=1", "train.py"], ["-C", "/x", "-e", "A=1", "train.py"], [], []),
        # unknown short option before the script: gpu.yaml script args
        (["-x", "1"], [], ["-x", "1"], []),
        # non-.py commands
        (["bash", "run.sh", "--fast"], ["bash"], ["run.sh", "--fast"], []),
        # help after the script is still gpu's help
        (["train.py", "--help"], ["train.py", "--help"], [], []),
        ([], [], [], []),
    ],
)
def test_split_run_argv(
    argv: list[str], click_args: list[str], script_args: list[str], clashes: list[str]
) -> None:
    assert split_run_argv(argv) == (click_args, script_args, clashes)


@pytest.mark.parametrize(
    ("argv", "click_args", "script_args", "clashes"),
    [
        # the live bug (job e940): `--gpu L4` after the script went to the script, so the
        # job ran on the default T4 while the user asked for an L4
        (
            ["gpucheck.py", "--provider", "lightning", "--gpu", "L4", "--hours", "0.1", "--wait"],
            ["gpucheck.py", "--provider", "lightning", "--gpu", "L4", "--hours", "0.1", "--wait"],
            [],
            [],
        ),
        # case-insensitive, =value form, catalog names with dashes
        (["train.py", "--gpu=l4"], ["train.py", "--gpu=l4"], [], []),
        (["train.py", "--gpu", "A100-40GB"], ["train.py", "--gpu", "A100-40GB"], [], []),
        # values that are no GPU type stay the script's (D23: `--gpu 0` is a training arg)
        (["train.py", "--gpu", "0"], ["train.py"], ["--gpu", "0"], ["--gpu"]),
        (["train.py", "--gpu=cuda:0"], ["train.py"], ["--gpu=cuda:0"], ["--gpu"]),
        (["train.py", "--gpu"], ["train.py"], ["--gpu"], ["--gpu"]),
        # a GPU model no provider offers is still gpu's: the router then refuses it
        # plainly ("lightning: no L4 (offers T4)") instead of running on a T4 (D56)
        (["train.py", "--gpu", "H100"], ["train.py", "--gpu", "H100"], [], []),
        (["train.py", "--gpu", "a100-80gb"], ["train.py", "--gpu", "a100-80gb"], [], []),
        (["train.py", "--gpu", "gpu0"], ["train.py"], ["--gpu", "gpu0"], ["--gpu"]),
        (["train.py", "--gpu", "all"], ["train.py"], ["--gpu", "all"], ["--gpu"]),
        # after `--` everything is the script's, even a GPU type
        (["train.py", "--", "--gpu", "L4"], ["train.py"], ["--gpu", "L4"], []),
    ],
)
def test_gpu_type_after_the_script_is_gpus(
    argv: list[str], click_args: list[str], script_args: list[str], clashes: list[str]
) -> None:
    """D56: `--gpu <a GPU type the catalog offers>` after the script is gpu's."""
    types = frozenset({"T4", "L4", "P100", "MPS", "A100-40GB"})
    assert split_run_argv(argv, types) == (click_args, script_args, clashes)


def test_gpu_types_default_to_the_catalog() -> None:
    from gpu_router.cli.app import catalog_gpu_types

    types = catalog_gpu_types()
    assert {"T4", "P100"} <= types
    assert split_run_argv(["x.py", "--gpu", "L4"]) == (["x.py", "--gpu", "L4"], [], [])
    assert split_run_argv(["x.py", "--gpu", "1"]) == (["x.py"], ["--gpu", "1"], ["--gpu"])


def test_a_broken_user_catalog_falls_back_to_the_packaged_types(
    gpu_home: Path,
) -> None:
    from gpu_router.cli.app import catalog_gpu_types
    from gpu_router.paths import Paths

    Paths.from_env().user_providers.parent.mkdir(parents=True, exist_ok=True)
    Paths.from_env().user_providers.write_text("providers: [not, a, mapping\n")
    assert {"T4", "L4"} <= catalog_gpu_types()


def test_missing_value_stays_with_its_option() -> None:
    """`train.py --vram` must be "requires an argument", not --vram=train.py."""
    click_args, _, _ = split_run_argv(["train.py", "--vram"])
    assert click_args == ["train.py", "--vram"]
