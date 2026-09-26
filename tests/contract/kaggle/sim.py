"""SimKaggle: a simulated Kaggle behind the kaggle CLI, for tests.

It is a `Runner` (gpu_router.providers.kaggle.cli): KaggleAdapter shells out to it exactly
as it would to `kaggle`, and it answers with the stdout/stderr/exit codes kaggle CLI 2.2.4
prints (shapes copied from the CLI source and live calls; see the Kaggle NOTES.md). All
state is in memory and every fact derives from the injected FakeClock, like the fake
adapter (D9).

Behaviour per kernel comes from fake-style directives registered by slug (`register`):

    duration (10)      seconds RUNNING before the runner exits
    pending_s (0)      seconds QUEUED first
    exit_code (0)      runner exit code (non-zero -> ERROR + exit line in the log)
    die_after          seconds into RUNNING when the session is killed (ERROR, time limit
                       failure message, no exit line)
    steps (100)        metric lines in the log
    checkpoint_every   seconds between ckpt_begin/ckpt_end lines
    rate_limit_n       first N pushes of this job answer HTTP 429
    unavailable_n      first N pushes fail with a connection error before anything is saved
    quota_limit        GPU-seconds allowed in total; runs die (ERROR, no message) when the
                       budget left at their push is used; `quota` reports it
    invalid            push answers `Kernel push error: Invalid machine shape ...`
    permanent          push answers `Kernel push error: ... violates ... terms of service`
    auth_required      push answers the CLI's 401 help text

`fail_next[op] = CliResult` makes the next call of that op (e.g. "kernels status") return
the given result once (outage injection).

Datasets (phase 5, the secrets channel): `datasets status|create|version` keep private
datasets in `datasets` (ref -> SimDataset); a new version reports `blobs_received` for
`dataset_ready_after` status calls before `ready`.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gpu_router import protocol
from gpu_router.clock import FakeClock
from gpu_router.providers.kaggle.cli import CliResult

AUTH_HELP = (
    "Authentication required to call the Kaggle API.\n\nFirst, you will need a Kaggle "
    "account. You can sign up at\n  https://www.kaggle.com/account/login\n"
)
CANNOT_ACCESS = (
    "Cannot access kernel '{ref}' (Permission 'kernels.get' was denied). The most likely "
    "cause is a wrong kernel slug."
)
_KEY_IN_RUNPY = re.compile(r"^ATTEMPT_KEY = '([^']+)'", re.M)


@dataclass
class SimKernel:
    slug: str
    attempt_key: str
    pushed_at: float
    metadata: dict[str, Any]
    run_py: str
    timeout_s: int | None
    directives: dict[str, Any]
    version: int = 1
    deleted_at: float | None = None
    quota_budget_s: float | None = None
    gpu: bool = True


@dataclass
class SimDataset:
    ref: str
    files: dict[str, bytes]
    version: int = 1
    public: bool = False
    pending: int = 0  # status calls left before "ready"
    deleted_old: int = 0  # versions created with --delete-old-versions


@dataclass
class _Timeline:
    status: str  # QUEUED RUNNING COMPLETE ERROR
    run_start: float
    end_at: float
    exit_code: int | None  # None: killed before the runner wrote its exit line
    failure: str | None


@dataclass
class SimKaggle:
    clock: FakeClock
    owner: str = "simuser"
    total_h: float = 30.0
    kernels: dict[str, SimKernel] = field(default_factory=dict)
    directives: dict[str, dict[str, Any]] = field(default_factory=dict)  # slug -> directives
    counters: dict[str, dict[str, int]] = field(default_factory=dict)  # job id -> counts
    calls: list[tuple[str, ...]] = field(default_factory=list)
    pushes: list[str] = field(default_factory=list)  # slugs, one entry per push
    fail_next: dict[str, CliResult] = field(default_factory=dict)
    quota_limit_s: float | None = None
    datasets: dict[str, SimDataset] = field(default_factory=dict)
    dataset_ready_after: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # ------------------------------------------------------------------ test API

    def register(self, slug: str, directives: Mapping[str, Any]) -> None:
        with self._lock:
            self.directives[slug] = dict(directives)

    def ref(self, slug: str) -> str:
        return f"{self.owner}/{slug}"

    # ------------------------------------------------------------------ runner

    def __call__(
        self,
        argv: Sequence[str],
        *,
        timeout: float,
        env: Mapping[str, str],
        cwd: Path | None = None,
    ) -> CliResult:
        args = list(argv[1:])
        if args and args[0] == "-W":
            args = args[1:]
        with self._lock:
            self.calls.append(tuple(args))
            op = " ".join(args[:2])
            injected = self.fail_next.pop(op, None) or self.fail_next.pop(args[0], None)
            if injected is not None:
                return CliResult(tuple(argv), injected.returncode, injected.stdout, injected.stderr)
            out, err, rc = self._dispatch(args)
        return CliResult(tuple(argv), rc, out, err)

    def _dispatch(self, args: list[str]) -> tuple[str, str, int]:
        if args == ["--version"]:
            return "Kaggle CLI 2.2.4\n", "", 0
        if args[:2] == ["config", "view"]:
            return (
                f"Configuration values from /sim/.kaggle\n- username: {self.owner}\n"
                "- auth_method: LEGACY_API_KEY\n- path: None\n- proxy: None\n"
                "- competition: None\n",
                "",
                0,
            )
        if args[:1] == ["quota"]:
            return self._quota(), "", 0
        if args[:1] == ["datasets"] and len(args) >= 2:
            return self._datasets(args)
        if args[:1] != ["kernels"] or len(args) < 2:
            return "", f"usage: kaggle ... unknown command {args}\n", 2
        sub = args[1]
        if sub == "push":
            return self._push(args)
        ref = args[2] if len(args) > 2 else ""
        kernel = self._kernel(ref)
        if sub == "delete":
            if kernel is None:
                return (
                    "",
                    "403 Client Error: Forbidden for url: "
                    "https://api.kaggle.com/v1/kernels.KernelsApiService/DeleteKernel\n",
                    1,
                )
            kernel.deleted_at = self.clock.now()
            return f"Kernel {ref} deleted successfully\n", "", 0
        if kernel is None:
            return "", CANNOT_ACCESS.format(ref=ref) + "\n", 1
        tl = self._timeline(kernel)
        if sub == "status":
            text = f'{ref} has status "KernelWorkerStatus.{tl.status}"\n'
            if tl.failure:
                text += f'Failure message: "{tl.failure}"\n'
            return text, "", 0
        if sub == "logs":
            if tl.status in ("QUEUED", "RUNNING"):
                return "\n", "", 0
            return json.dumps(self._log_events(kernel, tl)) + "\n", "", 0
        if sub == "output":
            return self._output(args, kernel, tl)
        return "", f"unknown kernels command {sub}\n", 2

    # ------------------------------------------------------------------ model

    def _kernel(self, ref: str) -> SimKernel | None:
        owner, _, slug = ref.partition("/")
        if owner != self.owner:
            return None
        kernel = self.kernels.get(slug)
        if kernel is None or kernel.deleted_at is not None:
            return None
        return kernel

    def _used_s(self, now: float) -> float:
        total = 0.0
        for k in self.kernels.values():
            if not k.gpu:
                continue
            tl = self._timeline(k)
            total += max(0.0, min(now, tl.end_at) - tl.run_start)
        return total

    def _limit_s(self) -> float:
        return self.quota_limit_s if self.quota_limit_s is not None else self.total_h * 3600

    def _timeline(self, k: SimKernel) -> _Timeline:
        d = k.directives
        now = self.clock.now()
        run_start = k.pushed_at + float(d.get("pending_s", 0))
        duration = float(d.get("duration", 10))
        end_at = run_start + duration
        exit_code: int | None = int(d.get("exit_code", 0))
        failure: str | None = None
        if d.get("die_after") is not None and float(d["die_after"]) < duration:
            end_at = run_start + float(d["die_after"])
            exit_code, failure = None, "Your notebook exceeded the maximum run time"
        if k.quota_budget_s is not None and run_start + k.quota_budget_s < end_at:
            end_at = run_start + k.quota_budget_s
            exit_code, failure = None, None
        if k.timeout_s is not None and run_start + k.timeout_s < end_at:
            end_at = run_start + k.timeout_s
            exit_code, failure = None, "Your notebook exceeded the maximum run time"
        if k.deleted_at is not None and k.deleted_at < end_at:
            end_at = max(k.deleted_at, run_start)
        if now < run_start:
            status = "QUEUED"
        elif now < end_at:
            status = "RUNNING"
        else:
            status = "COMPLETE" if exit_code == 0 else "ERROR"
        return _Timeline(status, run_start, end_at, exit_code, failure)

    def _log_lines(self, k: SimKernel, tl: _Timeline) -> list[str]:
        d = k.directives
        steps = int(d.get("steps", 100))
        lines = [
            f"gpu-router: kaggle attempt {k.attempt_key} starting",
            protocol.hello("bootstrap/0.2"),
        ]
        duration = max(tl.end_at - tl.run_start, 0.0)
        planned = float(d.get("duration", 10))
        if steps:
            lines.append(protocol.total(steps))
        done = steps if tl.exit_code is not None else int(steps * duration / max(planned, 1e-9))
        for i in range(1, min(done, steps) + 1):
            loss = round(2.0 * 0.97**i, 4)
            lines.append(protocol.metric(i, {"loss": loss}))
            lines.append(f"step {i}/{steps} loss={loss}")
        every = float(d.get("checkpoint_every", 0))
        if every > 0:
            seq, t = 1, every
            while t <= duration:
                lines.append(protocol.ckpt_begin(seq))
                lines.append(
                    protocol.ckpt_end(seq, f"file:///kaggle/working/.gpu-router/ckpt-{seq}")
                )
                seq, t = seq + 1, t + every
        if tl.exit_code is not None:
            lines.append(protocol.exit_line(tl.exit_code))
        return lines

    def _log_events(self, k: SimKernel, tl: _Timeline) -> list[dict[str, Any]]:
        return [
            {"stream_name": "stdout", "time": round(0.5 + i * 0.01, 3), "data": line + "\n"}
            for i, line in enumerate(self._log_lines(k, tl))
        ]

    def _quota(self) -> str:
        now = self.clock.now()
        used_h = self._used_s(now) / 3600
        total_h = self._limit_s() / 3600
        refresh = datetime.fromtimestamp(now + 7 * 86400, UTC).replace(tzinfo=None, microsecond=0)
        rows = [
            {
                "resource": "GPU",
                "used": f"{used_h:.6f}h",
                "remaining": f"{max(0.0, total_h - used_h):.6f}h",
                "total": f"{total_h:.6f}h",
                "refreshAt": refresh.isoformat(),
            },
            {
                "resource": "TPU",
                "used": "0.00h",
                "remaining": "20.00h",
                "total": "20.00h",
                "refreshAt": refresh.isoformat(),
            },
        ]
        return json.dumps(rows, indent=2) + "\n"

    def _datasets(self, args: list[str]) -> tuple[str, str, int]:
        sub = args[1]
        if sub == "status":
            ds = self.datasets.get(args[2] if len(args) > 2 else "")
            if ds is None:
                return (
                    "",
                    "404 Client Error: Not Found for url: https://api.kaggle.com/v1/"
                    "datasets.DatasetApiService/GetDatasetStatus\n",
                    1,
                )
            if ds.pending > 0:
                ds.pending -= 1
                return "blobs_received\n", "", 0
            return "ready\n", "", 0
        if sub not in ("create", "version"):
            return "", f"unknown datasets command {sub}\n", 2
        folder = Path(args[args.index("-p") + 1])
        meta = json.loads((folder / "dataset-metadata.json").read_text())
        ref = str(meta["id"])
        files = {
            p.name: p.read_bytes()
            for p in folder.iterdir()
            if p.is_file() and p.name != "dataset-metadata.json"
        }
        url = f"https://www.kaggle.com/datasets/{ref}"
        if sub == "create":
            if ref in self.datasets:
                return "Dataset creation error: The requested title is already in use\n", "", 0
            self.datasets[ref] = SimDataset(
                ref, files, public="-u" in args, pending=self.dataset_ready_after
            )
            kind = "public" if "-u" in args else "private"
            return f"Your {kind} Dataset is being created. Please check progress at {url}\n", "", 0
        ds = self.datasets.get(ref)
        if ds is None:
            return "Dataset version creation error: Dataset not found\n", "", 0
        ds.files, ds.version, ds.pending = files, ds.version + 1, self.dataset_ready_after
        ds.deleted_old += int("-d" in args)
        return f"Dataset version is being created. Please check progress at {url}\n", "", 0

    def _push(self, args: list[str]) -> tuple[str, str, int]:
        folder = Path(args[args.index("-p") + 1])
        timeout_s = int(args[args.index("-t") + 1]) if "-t" in args else None
        try:
            meta = json.loads((folder / "kernel-metadata.json").read_text())
        except OSError:
            return "", f"Metadata file not found: {folder}/kernel-metadata.json\n", 1
        title = str(meta.get("title") or "")
        if title and len(title) < 5:
            return "", "Title must be at least five characters\n", 1
        run_py = (folder / str(meta.get("code_file", "run.py"))).read_text()
        owner, _, slug = str(meta["id"]).partition("/")
        m = _KEY_IN_RUNPY.search(run_py)
        key = m.group(1) if m else slug
        d = self.directives.get(slug, {})
        job_id = key.split("-")[1] if key.count("-") >= 2 else key
        counts = self.counters.setdefault(job_id, {})
        if d.get("quota_limit") is not None:
            self.quota_limit_s = float(d["quota_limit"])
        if d.get("auth_required"):
            return AUTH_HELP, "", 1
        if counts.get("rate", 0) < int(d.get("rate_limit_n", 0)):
            counts["rate"] = counts.get("rate", 0) + 1
            return (
                "",
                "429 Client Error: Too Many Requests for url: "
                "https://api.kaggle.com/v1/kernels.KernelsApiService/SaveKernel\n",
                1,
            )
        if counts.get("unavail", 0) < int(d.get("unavailable_n", 0)):
            counts["unavail"] = counts.get("unavail", 0) + 1
            return (
                "",
                "Traceback (most recent call last):\n  ...\nrequests.exceptions.ConnectionError: "
                "HTTPSConnectionPool(host='api.kaggle.com', port=443): Max retries exceeded\n",
                1,
            )
        if d.get("invalid"):
            return "Kernel push error: Invalid machine shape NvidiaTeslaK80\n", "", 0
        if d.get("permanent"):
            return "Kernel push error: This notebook violates the Kaggle terms of service\n", "", 0
        now = self.clock.now()
        gpu = bool(meta.get("enable_gpu"))
        budget: float | None = None
        if gpu and self.quota_limit_s is not None:
            left = self.quota_limit_s - self._used_s(now)
            if left <= 0:
                return "Kernel push error: You have exceeded your weekly GPU quota\n", "", 0
            budget = left
        existing = self.kernels.get(slug)
        version = existing.version + 1 if existing is not None else 1
        self.kernels[slug] = SimKernel(
            slug=slug,
            attempt_key=key,
            pushed_at=now,
            metadata=meta,
            run_py=run_py,
            timeout_s=timeout_s,
            directives=d,
            version=version,
            quota_budget_s=budget,
            gpu=gpu,
        )
        self.pushes.append(slug)
        return (
            f"Kernel version {version} successfully pushed.  Please check progress at "
            f"https://www.kaggle.com/code/{owner}/{slug}\n",
            "",
            0,
        )

    def _output(self, args: list[str], k: SimKernel, tl: _Timeline) -> tuple[str, str, int]:
        target = Path(args[args.index("-p") + 1])
        pattern = args[args.index("--file-pattern") + 1] if "--file-pattern" in args else None
        rx = re.compile(pattern) if pattern else None
        files: dict[str, str] = {}
        if tl.status == "COMPLETE":
            files = {
                "outputs/result.json": json.dumps({"slug": k.slug, "ok": True}) + "\n",
                "outputs/model/weights.txt": f"sim weights for {k.slug}\n",
                ".gpu-router/checkpoints/ckpt-1.tar.gz": "not an output\n",
            }
        target.mkdir(parents=True, exist_ok=True)
        for name, body in files.items():
            if rx is not None and not rx.search(name):
                continue
            path = target / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body)
        if tl.status in ("COMPLETE", "ERROR"):
            (target / f"{k.slug}.log").write_text(json.dumps(self._log_events(k, tl)))
        return "", "", 0
