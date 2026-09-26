"""`<home>/setup.json`: what the wizard asked, did and found (phase 8b). Never a secret.

Shape (version 1, additive only):

    {
      "version": 1,
      "completed_at": 1727200000.0 | null,   # a run reached the end (first-run prompt off)
      "dismissed_at": 1727200000.0 | null,   # "not now" on the first-run prompt
      "run": {"id": "3f2a9c1b", "started_at": ..., "finished_at": ... | null,
              "answers": {"integration.plugin": "no", ...}},
      "items": {"tools.kaggle": {"outcome": "done", "summary": "...", "at": ...}, ...},
      "smoke": {"kaggle": {"ok": true, "at": ..., "job": "a7f2...", "gpu": "Tesla T4",
                           "seconds": 131.0, "summary": "..."}}
    }

Resume rule: a run whose `finished_at` is null was interrupted (Ctrl-C, closed terminal,
crash). The next `gpu setup` continues THAT run: its answers stand (a "no" is not asked
again), and every step re-detects what is done. A run that finished starts a new run on
the next `gpu setup`: done steps are skipped, declined ones are offered again.

Written atomically (tmp + fsync + rename), 0600, after every answer and every outcome, so
a crash loses at most the step in flight.
"""

from __future__ import annotations

import json
import os
import secrets as _random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gpu_router.setup.firstrun import SETUP_FILE, read_raw

__all__ = ["STATE_VERSION", "SetupState"]

STATE_VERSION = 1


@dataclass
class SetupState:
    path: Path
    completed_at: float | None = None
    dismissed_at: float | None = None
    run: dict[str, Any] = field(default_factory=dict)
    items: dict[str, dict[str, Any]] = field(default_factory=dict)
    smoke: dict[str, dict[str, Any]] = field(default_factory=dict)
    resumed: bool = False  # this process continues an interrupted run
    subset: dict[str, str] | None = None  # `--only` runs: answers kept here, not in `run`

    # ------------------------------------------------------------------ load / save

    @classmethod
    def load(cls, home: Path) -> SetupState:
        raw = read_raw(str(home)) or {}

        def num(v: Any) -> float | None:
            return float(v) if isinstance(v, int | float) and not isinstance(v, bool) else None

        def dicts(v: Any) -> dict[str, dict[str, Any]]:
            if not isinstance(v, dict):
                return {}
            return {str(k): dict(x) for k, x in v.items() if isinstance(x, dict)}

        run = raw.get("run")
        return cls(
            path=home / SETUP_FILE,
            completed_at=num(raw.get("completed_at")),
            dismissed_at=num(raw.get("dismissed_at")),
            run=dict(run) if isinstance(run, dict) else {},
            items=dicts(raw.get("items")),
            smoke=dicts(raw.get("smoke")),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "version": STATE_VERSION,
            "completed_at": self.completed_at,
            "dismissed_at": self.dismissed_at,
            "run": self.run,
            "items": self.items,
            "smoke": self.smoke,
        }

    def save(self) -> None:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        data = (json.dumps(self.to_json(), indent=2, sort_keys=True) + "\n").encode("utf-8")
        tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, self.path)

    # ------------------------------------------------------------------ runs

    @property
    def interrupted(self) -> bool:
        return bool(self.run.get("started_at")) and not self.run.get("finished_at")

    def begin(self, now: float, *, fresh: bool = False) -> None:
        """Continue an interrupted run, or start a new one (`fresh` always starts anew)."""
        if self.interrupted and not fresh:
            self.resumed = True
            self.run.setdefault("answers", {})
        else:
            self.resumed = False
            self.run = {
                "id": _random.token_hex(4),
                "started_at": now,
                "finished_at": None,
                "answers": {},
            }
        self.save()

    def begin_subset(self) -> None:
        """An `--only` run: it neither resumes nor ends the recorded run (an interrupted
        full run stays resumable), and its answers are not remembered."""
        self.resumed = False
        self.subset = {}

    def finish(self, now: float, *, complete: bool) -> None:
        """End this run. `complete` = every step ran (not an `--only` subset): the
        first-run prompt stops appearing."""
        if self.subset is not None:
            self.save()
            return
        self.run["finished_at"] = now
        if complete:
            self.completed_at = now
        self.save()

    def dismiss(self, now: float) -> None:
        self.dismissed_at = now
        self.save()

    # ------------------------------------------------------------------ answers / items

    def answer(self, item: str) -> str | None:
        answers = self.subset if self.subset is not None else self.run.get("answers")
        value = answers.get(item) if isinstance(answers, dict) else None
        return str(value) if value in ("yes", "no") else None

    def record_answer(self, item: str, yes: bool) -> None:
        if self.subset is not None:
            self.subset[item] = "yes" if yes else "no"
            return
        self.run.setdefault("answers", {})[item] = "yes" if yes else "no"
        self.save()

    def record_item(self, item: str, outcome: str, summary: str, now: float) -> None:
        self.items[item] = {"outcome": outcome, "summary": summary, "at": now}
        self.save()

    def record_smoke(self, provider: str, result: dict[str, Any]) -> None:
        self.smoke[provider] = result
        self.save()
