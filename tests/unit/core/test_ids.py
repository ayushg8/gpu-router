from __future__ import annotations

import pytest

from gpu_router import ids
from gpu_router.errors import InvalidRequest


def test_new_job_id_shape() -> None:
    job_id = ids.new_job_id(lambda _p: False)
    assert ids.JOB_ID_RE.match(job_id)


def test_new_job_id_retries_taken_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    seq = iter(["aaaa11111111", "aaaa22222222", "bbbb33333333"])
    monkeypatch.setattr(ids.secrets, "token_hex", lambda _n: next(seq))
    assert ids.new_job_id(lambda p: p == "aaaa") == "bbbb33333333"


def test_new_job_id_gives_up_after_max_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def fake(_n: int) -> str:
        calls.append(1)
        return f"aaaa{len(calls):08x}"

    monkeypatch.setattr(ids.secrets, "token_hex", fake)
    job_id = ids.new_job_id(lambda _p: True)
    assert len(calls) == ids.MAX_PREFIX_RETRIES
    assert job_id == f"aaaa{ids.MAX_PREFIX_RETRIES:08x}"


@pytest.mark.parametrize(
    ("raw", "want"), [("a7f2", "a7f2"), (" A7F2 ", "a7f2"), ("0", "0"), ("abcdef012345",) * 2]
)
def test_normalize_ref_ok(raw: str, want: str) -> None:
    assert ids.normalize_ref(raw) == want


@pytest.mark.parametrize("raw", ["", "  ", "xyz", "a7f2-", "abcdef0123456", "g1"])
def test_normalize_ref_rejects(raw: str) -> None:
    with pytest.raises(InvalidRequest, match="is not a job id"):
        ids.normalize_ref(raw)


@pytest.mark.parametrize(
    ("job_id", "neighbours", "want"),
    [
        ("a7f2c19e0b3d", (None, None), "a7f2"),
        ("a7f2c19e0b3d", ("a7f1ffffffff", "a7f300000000"), "a7f2"),
        ("a7f2c19e0b3d", ("a7f2c0000000", None), "a7f2c1"),
        ("a7f2c19e0b3d", (None, "a7f2c19e0b3e"), "a7f2c19e0b3d"),
        ("a7f2c19e0b3d", ("a7f2a0000000", "a7f2c19f0000"), "a7f2c19e"),
    ],
)
def test_shortest_unique_prefix(
    job_id: str, neighbours: tuple[str | None, str | None], want: str
) -> None:
    assert ids.shortest_unique_prefix(job_id, neighbours) == want


def test_attempt_and_checkpoint_ids() -> None:
    job = "a7f2c19e0b3d"
    assert ids.attempt_id(job, 1) == "a7f2c19e0b3d.1"
    assert ids.attempt_key(job, 12) == "gpu-a7f2c19e0b3d-12"
    assert ids.ATTEMPT_KEY_RE.match(ids.attempt_key(job, 999))
    assert len(ids.attempt_key(job, 999)) <= 24
    assert ids.checkpoint_id(job, 3) == "a7f2c19e0b3d.c3"
    for fn in (ids.attempt_id, ids.attempt_key, ids.checkpoint_id):
        with pytest.raises(ValueError, match=">= 1"):
            fn(job, 0)
