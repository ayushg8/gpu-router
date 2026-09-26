"""SimLightning: a simulated Lightning AI behind the SDK driver, for tests.

It is a `Runner` (gpu_router.providers.lightning.sdk): LightningAdapter's SdkBridge hands it
the driver's argv, stdin request and environment exactly as it would to the real SDK
interpreter, and it answers with the driver's `@@GRL:` result line (protocol in
providers/lightning/driver.py). All state is in memory and every fact derives from the
injected FakeClock (like the fake adapter, D9).

Behaviour per job comes from fake-style directives registered by job name (`register`):

    duration (10)      seconds Running before the runner exits
    pending_s (0)      seconds Pending first
    exit_code (0)      runner exit code (non-zero -> Failed + exit line in the log)
    die_after          seconds into Running when the machine is lost (Failed, no exit line)
    steps (100)        metric lines in the log
    checkpoint_every   seconds between ckpt_begin/ckpt_end lines
    rate_limit_n       first N submits of this job answer HTTP 429 from the create call
    unavailable_n      first N submits fail with a connection error before anything exists
    quota_limit        credit-seconds allowed in total (1 credit per second here); a run is
                       Stopped ("insufficient credits") when the budget left at its submit is
                       used, and later submits see a zero balance
    invalid            the create call answers 400 (bad machine)
    permanent          the create call answers 403 "account suspended"
    auth_required      the call answers 401 before anything is created
    ignore_cancel      stop requests are accepted but the job keeps running
    wall               seconds after which the launcher's wall clock ends the run (WALL_MARK)

`fail_next[op] = {"kind": ..., "error": ...}` makes the next call of that op fail once;
`hang_next[op] = True` makes it raise CallTimeout (as if the process group was killed).
`calls` records (op, params) for every call; `envs` the environments they got.
"""

from __future__ import annotations

import base64
import io
import json
import tarfile
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gpu_router import protocol
from gpu_router.clock import FakeClock
from gpu_router.providers.lightning import launch
from gpu_router.providers.lightning.sdk import MARKER, CallTimeout, ProcResult

TERMINAL = ("Completed", "Failed", "Stopped")


@dataclass
class SimJob:
    name: str
    submitted_at: float
    directives: dict[str, Any]
    machine: str
    command: str
    env: dict[str, str]
    files: dict[str, bytes]
    quota_budget_s: float | None = None
    stop_at: float | None = None
    stop_done_at: float | None = None
    interruptible: bool = False


@dataclass
class _Timeline:
    status: str
    run_start: float
    end_at: float
    exit_code: int | None  # None: ended before the runner wrote its exit line
    message: str | None
    wall: bool = False


@dataclass
class SimLightning:
    clock: FakeClock
    user: str = "simuser"
    teamspace: str = "simuser/default"
    teamspaces: list[str] | None = None
    balance_limit: float = 15 * 3600.0  # credit-seconds (1 credit per GPU second here)
    balance_api: bool = True
    jobs: dict[str, SimJob] = field(default_factory=dict)
    directives: dict[str, dict[str, Any]] = field(default_factory=dict)
    counters: dict[str, dict[str, int]] = field(default_factory=dict)
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    envs: list[dict[str, str]] = field(default_factory=list)
    uploads: dict[str, bytes] = field(default_factory=dict)  # drive path -> content
    removed: list[str] = field(default_factory=list)
    fail_next: dict[str, dict[str, Any]] = field(default_factory=dict)
    hang_next: dict[str, bool] = field(default_factory=dict)
    studios: set[str] = field(default_factory=set)
    quota_limit_s: float | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # ------------------------------------------------------------------ test API

    def register(self, name: str, directives: Mapping[str, Any]) -> None:
        with self._lock:
            self.directives[name] = dict(directives)

    def ops(self) -> list[str]:
        return [op for op, _ in self.calls]

    # ------------------------------------------------------------------ runner

    def __call__(
        self, argv: Sequence[str], *, stdin: str, timeout: float, env: Mapping[str, str]
    ) -> ProcResult:
        request = json.loads(stdin)
        op = str(request["op"])
        params = dict(request.get("params") or {})
        with self._lock:
            self.calls.append((op, params))
            self.envs.append(dict(env))
            if self.hang_next.pop(op, False):
                raise CallTimeout(f"timed out after {timeout:g}s")
            injected = self.fail_next.pop(op, None)
            if injected is not None:
                return self._reply(argv, {"ok": False, **injected})
            if not (env.get("LIGHTNING_USER_ID") and env.get("LIGHTNING_API_KEY")):
                return self._reply(
                    argv, {"ok": False, "kind": "auth", "error": "no lightning credentials"}
                )
            try:
                result = getattr(self, f"_op_{op}")(params)
            except _Fail as exc:
                doc = dict(exc.doc)
                if op == "submit" and not doc.get("stage"):
                    doc["stage"] = "pre"  # the driver's rule: before the create call
                return self._reply(argv, {"ok": False, **doc})
        return self._reply(argv, {"ok": True, "result": result})

    @staticmethod
    def _reply(argv: Sequence[str], doc: dict[str, Any]) -> ProcResult:
        data = base64.b64encode(json.dumps(doc).encode()).decode()
        return ProcResult(tuple(argv), 0 if doc.get("ok") else 1, f"{MARKER}{data}\n", "")

    # ------------------------------------------------------------------ model

    def _count(self, name: str, what: str) -> int:
        job_id = name.rsplit("-", 1)[0]
        bucket = self.counters.setdefault(job_id, {})
        bucket[what] = bucket.get(what, 0) + 1
        return bucket[what]

    def _limit_s(self) -> float:
        return self.quota_limit_s if self.quota_limit_s is not None else self.balance_limit

    def _used_s(self, now: float) -> float:
        total = 0.0
        for job in self.jobs.values():
            tl = self._timeline(job)
            total += max(0.0, min(now, tl.end_at) - tl.run_start)
        return total

    def _timeline(self, job: SimJob) -> _Timeline:
        d = job.directives
        now = self.clock.now()
        run_start = job.submitted_at + float(d.get("pending_s", 0))
        duration = float(d.get("duration", 10))
        end_at = run_start + duration
        code: int | None = int(d.get("exit_code", 0))
        message: str | None = None
        status_end = "Completed" if code == 0 else "Failed"
        wall = False
        if d.get("die_after") is not None and float(d["die_after"]) < duration:
            end_at = run_start + float(d["die_after"])
            code, message, status_end = None, "the machine was lost", "Failed"
        if d.get("wall") is not None and run_start + float(d["wall"]) < end_at:
            end_at = run_start + float(d["wall"])
            code, status_end, wall = 143, "Failed", True
        if job.quota_budget_s is not None and run_start + job.quota_budget_s < end_at:
            end_at = run_start + job.quota_budget_s
            code, message, status_end = None, "insufficient credits: job stopped", "Stopped"
        if job.stop_done_at is not None and job.stop_done_at < end_at:
            end_at = max(job.stop_done_at, run_start)
            code, message, status_end = None, None, "Stopped"
        if now < run_start:
            status = "Pending"
        elif now < end_at:
            status = "Stopping" if job.stop_at is not None and job.stop_done_at else "Running"
        else:
            status = status_end
        return _Timeline(status, run_start, end_at, code, message, wall)

    def _lines(self, job: SimJob, tl: _Timeline) -> list[tuple[float, str]]:
        """(time, line) for the whole run as far as the timeline goes."""
        d = job.directives
        steps = int(d.get("steps", 100))
        duration = float(d.get("duration", 10))
        start = tl.run_start
        out: list[tuple[float, str]] = [
            (start, f"gpu-router: lightning attempt starting on Tesla T4, 15360 MiB ({job.name})"),
            (start, protocol.hello("bootstrap/0.2")),
        ]
        if steps:
            out.append((start, protocol.total(steps)))
        for i in range(1, steps + 1):
            t = start + duration * i / (steps + 1)
            loss = round(2.0 * 0.97**i, 4)
            out.append((t, protocol.metric(i, {"loss": loss})))
            out.append((t, f"step {i}/{steps} loss={loss}"))
        every = float(d.get("checkpoint_every", 0))
        if every > 0:
            seq, t = 1, every
            while t <= duration:
                out.append((start + t, protocol.ckpt_begin(seq)))
                out.append(
                    (start + t, protocol.ckpt_end(seq, f"file:///tmp/gpu-router/ckpt-{seq}"))
                )
                seq, t = seq + 1, t + every
        out.sort(key=lambda x: x[0])
        out = [(t, line) for t, line in out if t < tl.end_at or t == start]
        if tl.wall:
            out.append((tl.end_at, f"{launch.WALL_MARK} (60s); stopping the job"))
        if tl.exit_code is not None:
            out.append((tl.end_at, protocol.exit_line(tl.exit_code)))
            out.append((tl.end_at, f"gpu-router: finished with exit code {tl.exit_code}"))
        return out

    def _summary(self, job: SimJob) -> dict[str, Any]:
        tl = self._timeline(job)
        now = self.clock.now()
        running = tl.status not in ("Pending",)
        return {
            "name": job.name,
            "id": f"id-{job.name}",
            "status": tl.status,
            "started_at": tl.run_start if running else None,
            "stopped_at": tl.end_at if tl.status in TERMINAL else None,
            "total_cost": max(0.0, min(now, tl.end_at) - tl.run_start) / 3600,
            "message": tl.message,
            "server_error": None,
            "interrupted": False,
        }

    def _log(self, job: SimJob) -> dict[str, Any]:
        tl = self._timeline(job)
        if tl.status == "Pending":
            return {"lines": [], "note": "Logs are not available while the job is Pending."}
        now = self.clock.now()
        return {"lines": [line for t, line in self._lines(job, tl) if t <= now]}

    def _job(self, params: Mapping[str, Any]) -> SimJob:
        job = self.jobs.get(str(params["name"]))
        if job is None:
            raise _Fail("not_found", f"lightning has no job {params['name']}")
        return job

    def _check_ts(self, params: Mapping[str, Any]) -> str:
        ts = params.get("teamspace")
        if ts and ts not in (self.teamspaces or [self.teamspace]):
            raise _Fail("config", f"lightning teamspace {ts!r}: not found")
        return str(ts or self.teamspace)

    # ------------------------------------------------------------------ ops

    def _op_whoami(self, params: dict[str, Any]) -> dict[str, Any]:
        spaces = self.teamspaces or [self.teamspace]
        chosen = self._check_ts(params) if params.get("teamspace") else None
        if chosen is None and len(spaces) == 1:
            chosen = spaces[0]
        return {"user": self.user, "teamspaces": spaces, "teamspace": chosen, "sdk_version": "sim"}

    def _op_submit(self, params: dict[str, Any]) -> dict[str, Any]:
        ts = self._check_ts(params)
        name = str(params["name"])
        d = self.directives.get(name, {})
        if name in self.jobs:
            return {"existed": True, "teamspace": ts, **self._summary(self.jobs[name])}
        if d.get("auth_required"):
            raise _Fail("auth", "lightning rejected the credentials (401)", status=401)
        if d.get("unavailable_n") and self._count(name, "unavailable") <= int(d["unavailable_n"]):
            raise _Fail("unavailable", "lightning is unreachable: ConnectionError")
        now = self.clock.now()
        left = self._limit_s() - self._used_s(now)
        if params.get("min_balance") is not None and left / 3600 < float(params["min_balance"]):
            raise _Fail("quota", f"lightning credits are used up ({left / 3600:.2f} left)")
        if not params.get("create_studio") and params.get("studio") not in self.studios:
            raise _Fail(
                "config", f"lightning studio: Studio '{params.get('studio')}' does not exist."
            )
        self.studios.add(str(params.get("studio")))
        files: dict[str, bytes] = {}
        for item in params.get("files") or []:
            data = Path(item["local"]).read_bytes()
            self.uploads[item["remote"]] = data
            files[Path(item["remote"]).name] = data
        if d.get("rate_limit_n") and self._count(name, "rate") <= int(d["rate_limit_n"]):
            raise _Fail("rate", "lightning is rate limiting (429)", status=429, stage="run")
        if d.get("invalid"):
            raise _Fail("invalid", "lightning refused the request (400): bad machine", 400, "run")
        if d.get("permanent"):
            raise _Fail("permanent", "lightning refused access: account suspended", 403, "run")
        job = SimJob(
            name=name,
            submitted_at=now,
            directives=dict(d),
            machine=str(params["machine"]),
            command=str(params["command"]),
            env=dict(params.get("env") or {}),
            files=files,
            interruptible=bool(params.get("interruptible")),
        )
        if "quota_limit" in d:
            self.quota_limit_s = float(d["quota_limit"])
        if self.quota_limit_s is not None:
            job.quota_budget_s = max(0.0, self.quota_limit_s - self._used_s(now))
        self.jobs[name] = job
        return {
            "existed": False,
            "teamspace": ts,
            "link": f"https://lightning.ai/{ts}/studios/gpu-router/app?job_name={name}",
            **self._summary(job),
        }

    def _op_status(self, params: dict[str, Any]) -> dict[str, Any]:
        self._check_ts(params)
        job = self._job(params)
        out = self._summary(job)
        if params.get("with_log") and out["status"] in TERMINAL:
            out["log"] = self._log(job)
        return out

    def _op_logs(self, params: dict[str, Any]) -> dict[str, Any]:
        self._check_ts(params)
        job = self._job(params)
        out = self._summary(job)
        out["log"] = self._log(job)
        return out

    def _op_stop(self, params: dict[str, Any]) -> dict[str, Any]:
        self._check_ts(params)
        job = self.jobs.get(str(params["name"]))
        if job is None:
            return {"missing": True}
        tl = self._timeline(job)
        if tl.status in TERMINAL:
            return {"status": tl.status, "already": True}
        if job.stop_at is None:
            job.stop_at = self.clock.now()
            if not job.directives.get("ignore_cancel"):
                job.stop_done_at = self.clock.now() + 2.0
        return {"status": self._timeline(job).status, "confirmed": False}

    def _op_fetch(self, params: dict[str, Any]) -> dict[str, Any]:
        self._check_ts(params)
        job = self._job(params)
        tl = self._timeline(job)
        dest = Path(str(params["dest"]))
        dest.mkdir(parents=True, exist_ok=True)
        if tl.status not in TERMINAL:
            return {"source": None, "archive": None, "notes": ["not finished"]}
        archive = dest / "outputs.tar.gz"
        members: dict[str, bytes] = {}
        if tl.status == "Completed":
            members = {"model.txt": b"trained\n", "metrics/result.json": b'{"loss": 0.1}\n'}
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for rel, data in members.items():
                info = tarfile.TarInfo(rel)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        if tl.status == "Stopped":
            return {"source": None, "archive": None, "notes": ["drive: FileNotFoundError"]}
        archive.write_bytes(buf.getvalue())
        return {"source": "drive", "archive": str(archive), "notes": []}

    def _op_cleanup(self, params: dict[str, Any]) -> dict[str, Any]:
        self._check_ts(params)
        removed = []
        for path in params.get("paths") or []:
            self.uploads.pop(path, None)
            self.removed.append(path)
            removed.append(path)
        return {"removed": removed, "failed": []}

    def _op_quota(self, params: dict[str, Any]) -> dict[str, Any]:
        ts = self._check_ts(params)
        now = self.clock.now()
        used_credits = self._used_s(now) / 3600
        out: dict[str, Any] = {
            "teamspace": ts,
            "jobs_cost": used_credits,
            "jobs_counted": len(self.jobs),
            "rates": {m: {"cost": 0.68, "interruptible_cost": 0.2} for m in params["machines"]},
        }
        if self.balance_api:
            out["balance"] = max(0.0, self._limit_s() / 3600 - used_credits)
        else:
            out["balance_error"] = "ApiException: (404)"
        return out


class _Fail(Exception):
    def __init__(
        self, kind: str, error: str, status: int | None = None, stage: str | None = None
    ) -> None:
        super().__init__(error)
        self.doc = {"kind": kind, "error": error, "status": status, "stage": stage}
