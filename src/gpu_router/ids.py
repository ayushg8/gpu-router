"""Job, attempt and checkpoint identifiers (phase 1; owner: group A).

- Job id: 12 lowercase hex chars (48 random bits from `secrets.token_hex`). Stored in full.
- Display id (`short_id`): the shortest prefix of the id, at least 4 chars, that is unique
  among all jobs. Creation retries (up to `MAX_PREFIX_RETRIES`) until the 4-char prefix is
  unused, so in practice every job displays as 4 chars (`a7f2`).
- Refs typed by users/agents: any prefix (1..12 hex chars, case-insensitive). Resolution is
  the store's job (`Store.resolve_ref`): exactly one match or JobNotFound/AmbiguousJobRef.
- Attempt id: "<job_id>.<n>"; attempt key (handed to providers, used to name/tag remote
  runs, must be a valid Kaggle slug / Modal app name / Colab session label):
  "gpu-<job_id>-<n>" -> [a-z0-9-], at most 24 chars for n < 1000.
- Checkpoint id: "<job_id>.c<seq>".
"""

from __future__ import annotations

import re
import secrets
from collections.abc import Callable

JOB_ID_LEN = 12
SHORT_ID_LEN = 4
MAX_PREFIX_RETRIES = 20
JOB_ID_RE = re.compile(r"^[0-9a-f]{12}$")
JOB_REF_RE = re.compile(r"^[0-9a-f]{1,12}$")
ATTEMPT_KEY_RE = re.compile(r"^gpu-[0-9a-f]{12}-[1-9][0-9]{0,2}$")


def new_job_id(prefix_taken: Callable[[str], bool]) -> str:
    """Return a fresh job id whose first SHORT_ID_LEN chars are not `prefix_taken`.

    Tries MAX_PREFIX_RETRIES random ids; if all 4-char prefixes collide, returns the last
    candidate anyway (its short_id will simply be longer). Never returns an id that equals
    an existing one: the caller's INSERT enforces that via the primary key.
    """
    candidate = secrets.token_hex(JOB_ID_LEN // 2)
    for _ in range(MAX_PREFIX_RETRIES - 1):
        if not prefix_taken(candidate[:SHORT_ID_LEN]):
            return candidate
        candidate = secrets.token_hex(JOB_ID_LEN // 2)
    return candidate


def normalize_ref(ref: str) -> str:
    """Strip whitespace, lowercase, validate against JOB_REF_RE.

    Raises gpu_router.errors.InvalidRequest("'xyz' is not a job id") if it is not hex.
    """
    from gpu_router.errors import InvalidRequest

    norm = ref.strip().lower()
    if not JOB_REF_RE.match(norm):
        raise InvalidRequest(
            f"{ref.strip()!r} is not a job id",
            hint="job ids are hex, e.g. a7f2 (see `gpu status`)",
        )
    return norm


def shortest_unique_prefix(job_id: str, neighbours: tuple[str | None, str | None]) -> str:
    """Shortest prefix (>= SHORT_ID_LEN) of `job_id` that differs from both lexicographic
    neighbours (the ids immediately before and after it; None at either end)."""
    length = SHORT_ID_LEN
    for other in neighbours:
        if other is None or other == job_id:
            continue
        common = 0
        for a, b in zip(job_id, other, strict=False):
            if a != b:
                break
            common += 1
        length = max(length, common + 1)
    return job_id[: min(length, len(job_id))]


def attempt_id(job_id: str, n: int) -> str:
    """ "<job_id>.<n>" (n >= 1)."""
    if n < 1:
        raise ValueError(f"attempt number must be >= 1, got {n}")
    return f"{job_id}.{n}"


def attempt_key(job_id: str, n: int) -> str:
    """ "gpu-<job_id>-<n>" (n >= 1). Must match ATTEMPT_KEY_RE."""
    if n < 1:
        raise ValueError(f"attempt number must be >= 1, got {n}")
    return f"gpu-{job_id}-{n}"


def checkpoint_id(job_id: str, seq: int) -> str:
    """ "<job_id>.c<seq>" (seq >= 1)."""
    if seq < 1:
        raise ValueError(f"checkpoint seq must be >= 1, got {seq}")
    return f"{job_id}.c{seq}"
