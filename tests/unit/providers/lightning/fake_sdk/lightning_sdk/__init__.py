"""Fake lightning_sdk (see ../README.txt)."""

from __future__ import annotations

import os
import threading
import time
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from lightning_sdk import _state

__version__ = "fake-2026.9.18"

_state.call(
    "import",
    home=os.environ.get("HOME"),
    credential_path=os.environ.get("LIGHTNING_CREDENTIAL_PATH"),
    settings_path=os.environ.get("LIGHTNING_SETTINGS_PATH"),
    version_check=os.environ.get("LIGHTNING_DISABLE_VERSION_CHECK"),
    browser=os.environ.get("BROWSER"),
    has_key=bool(os.environ.get("LIGHTNING_API_KEY")),
)


class Status(Enum):
    NotCreated = "NotCreated"
    Pending = "Pending"
    Running = "Running"
    Stopping = "Stopping"
    Stopped = "Stopped"
    Completed = "Completed"
    Failed = "Failed"

    def __str__(self) -> str:
        return self.value


class Machine:
    def __init__(self, name: str, cost: float | None = None) -> None:
        self.name = name
        self.cost = cost
        self.interruptible_cost = None
        self.wait_time = None

    def __str__(self) -> str:
        return self.name


Machine.T4 = Machine("T4")  # type: ignore[attr-defined]
Machine.L4 = Machine("L4")  # type: ignore[attr-defined]
Machine.T4_SMALL = Machine("T4_SMALL")  # type: ignore[attr-defined]


class _Owner:
    def __init__(self, name: str) -> None:
        self.name = name
        self.id = f"owner-{name}"


class User:
    def __init__(self, name: str | None = None) -> None:
        self.name = name or _state.STATE.get("user", "me")

    @property
    def teamspaces(self) -> list[Teamspace]:
        return [Teamspace(s) for s in _state.STATE.get("teamspaces", ["me/default"])]


def _dt(value: Any, *, fraction: bool = False) -> str | None:
    """Timestamps the way the live API returns them (seen 2026-09-25): ISO-8601 strings,
    `Job.started_at` = "2026-09-25T07:00:43Z", `V1Job.created_at` with microseconds."""
    if value is None:
        return None
    dt = datetime.fromtimestamp(float(value), tz=UTC)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ" if fraction else "%Y-%m-%dT%H:%M:%SZ")


class Teamspace:
    def __init__(self, name: str | None = None, org: Any = None, user: Any = None) -> None:
        spaces = _state.STATE.get("teamspaces", ["me/default"])
        match = [s for s in spaces if s == name or s.split("/", 1)[1] == name]
        if not match:
            raise ValueError(f"Teamspace {name} does not exist")
        owner, self.name = match[0].split("/", 1)
        self.owner = _Owner(owner)
        self.id = f"ts-{self.name}"
        self.default_cloud_account = "lightning-public-prod"

    def upload_file(
        self,
        file_path: Any,
        remote_path: str | None = None,
        progress_bar: bool = True,
        cloud_account: str | None = None,
    ) -> None:
        _state.maybe_raise("Teamspace.upload_file")
        data = Path(file_path).read_bytes()
        _state.call(
            "Teamspace.upload_file",
            remote=remote_path,
            size=len(data),
            progress_bar=progress_bar,
            cloud_account=cloud_account,
        )
        _state.drive_put(str(remote_path), data)

    def download_file(
        self, remote_path: str, file_path: str | None = None, cloud_account: str | None = None
    ) -> None:
        _state.call("Teamspace.download_file", remote=remote_path)
        _state.maybe_raise("Teamspace.download_file")
        data = _state.drive_get(remote_path)
        if data is None:
            raise RuntimeError(f"Failed to download {remote_path}: 404")
        Path(file_path or remote_path).write_bytes(data)

    def list_jobs(self, tags: Any = None) -> tuple[Job, ...]:
        _state.call("Teamspace.list_jobs")
        out = []
        for name in _state.STATE.get("jobs", {}):
            job = Job(name, teamspace=self, _fetch_job=False)
            job._job = job._raw()
            out.append(job)
        return tuple(out)

    def list_machines(
        self, cloud_account: str | None = None, machine: str | None = None
    ) -> list[Machine]:
        _state.call("Teamspace.list_machines", machine=machine)
        info = (_state.STATE.get("machines") or {}).get(machine or "")
        if not info:
            return []
        m = Machine(str(machine), cost=info.get("cost"))
        m.interruptible_cost = info.get("interruptible_cost")
        m.wait_time = info.get("wait_time")
        return [m]


class Studio:
    _skip_setup = threading.local()
    _skip_init = threading.local()

    def __init__(
        self,
        name: str | None = None,
        teamspace: Any = None,
        org: Any = None,
        user: Any = None,
        cloud: Any = None,
        create_ok: bool = True,
        source: str | None = None,
        disable_secrets: bool = False,
        studio_type: str | None = None,
    ) -> None:
        skipped = bool(getattr(self._skip_setup, "value", False))
        _state.call("Studio", name=name, create_ok=create_ok, skip_setup=skipped)
        _state.maybe_raise("Studio")
        studios = _state.STATE.setdefault("studios", [])
        if name not in studios:
            if not create_ok:
                raise ValueError(f"Studio '{name}' does not exist.")
            studios.append(name)
            _state.save()
            _state.call("Studio.create", name=name)
        self.name = name
        self.teamspace = teamspace
        self._studio = type("S", (), {"id": f"studio-{name}"})()

    @property
    def cloud_account(self) -> str:
        return "lightning-public-prod"


class _Raw:
    def __init__(self, d: dict[str, Any]) -> None:
        self.message = d.get("message")
        self.server_error = d.get("server_error")
        self.interruption_notice_received = d.get("interrupted", False)
        self.created_at = _dt(d.get("created_at"), fraction=True)
        self.total_cost = d.get("total_cost", 0.0)
        self.id = f"id-{d.get('name')}"


class FileEntry:
    def __init__(self, path: str, is_dir: bool) -> None:
        self.path = path
        self.is_dir = is_dir


class Job:
    def __init__(
        self,
        name: str,
        teamspace: Any = None,
        org: Any = None,
        user: Any = None,
        *,
        _fetch_job: bool = True,
        _num_machines: int = 1,
    ) -> None:
        self._name = name
        self._teamspace = teamspace
        self._job: Any = None
        self._prevent_refetch_latest = False
        if _fetch_job:
            _state.maybe_raise("Job")
            if name not in _state.STATE.get("jobs", {}):
                raise ValueError(f"Job {name} does not exist in Teamspace {teamspace.name}")
            self._job = self._raw()

    def _d(self) -> dict[str, Any]:
        return dict(_state.STATE["jobs"][self._name], name=self._name)

    def _raw(self) -> _Raw:
        return _Raw(self._d())

    def _latest(self) -> dict[str, Any]:
        if not self._prevent_refetch_latest:
            _state.call("Job.refetch", name=self._name)
        return self._d()

    @property
    def name(self) -> str:
        return self._name

    @property
    def id(self) -> str:
        return f"id-{self._name}"

    @property
    def status(self) -> Status:
        _state.call("Job.refetch", name=self._name)
        _state.maybe_raise("Job.status")
        return Status(self._d()["status"])

    @property
    def started_at(self) -> str | None:
        return _dt(self._latest().get("started_at"))

    @property
    def stopped_at(self) -> str | None:
        return _dt(self._latest().get("stopped_at"))

    @property
    def total_cost(self) -> float:
        return float(self._latest().get("total_cost", 0.0))

    @property
    def link(self) -> str:
        return f"https://lightning.ai/{self._teamspace.owner.name}/{self._teamspace.name}/studios/gpu-router/app?app_id=jobs&job_name={self._name}"

    @property
    def logs(self) -> Any:
        def fetch(**kwargs: Any) -> str:
            d = self._d()
            tail = kwargs.get("tail")
            _state.call("Job.logs", name=self._name, follow=kwargs.get("follow"), tail=tail)
            if d.get("log_sleep"):
                time.sleep(float(d["log_sleep"]))
            if tail is None and d.get("full_log_sleep"):  # only whole-log reads are slow
                time.sleep(float(d["full_log_sleep"]))
            if d.get("log_error"):
                raise RuntimeError(d["log_error"])
            text = str(d.get("log", ""))
            return "\n".join(text.splitlines()[-int(tail) :]) if tail is not None else text

        return fetch

    def stop(self) -> None:
        _state.call("Job.stop", name=self._name)
        d = _state.STATE["jobs"][self._name]
        if d["status"] in ("Stopped", "Completed", "Failed"):
            return
        d["status"] = "Stopping"
        _state.save()
        if d.get("stop_sleep"):
            time.sleep(float(d["stop_sleep"]))
        d["status"] = "Stopped"
        _state.save()

    def delete(self) -> None:
        _state.call("Job.delete", name=self._name)
        _state.STATE["jobs"].pop(self._name, None)
        _state.save()

    def list_artifacts(self, path: str = "", recursive: bool = False) -> list[FileEntry]:
        _state.call("Job.list_artifacts", recursive=recursive)
        prefix = f"jobs/{self._name}/"
        return [
            FileEntry(k[len(prefix) :], False)
            for k in (_state.STATE.get("drive") or {})
            if k.startswith(prefix)
        ]

    def download_artifacts(self, target_dir: Any = ".", path: str = "") -> None:
        _state.call("Job.download_artifacts", path=path)
        prefix = f"jobs/{self._name}/{path}".rstrip("/") + "/"
        for k in list(_state.STATE.get("drive") or {}):
            if k.startswith(prefix):
                dest = Path(target_dir) / k[len(prefix) :]
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(_state.drive_get(k) or b"")

    @classmethod
    def run(
        cls,
        name: str,
        machine: Any,
        cloud: Any = None,
        command: str | None = None,
        studio: Any = None,
        image: str | None = None,
        teamspace: Any = None,
        org: Any = None,
        user: Any = None,
        env: dict[str, str] | None = None,
        interruptible: bool = False,
        image_credentials: str | None = None,
        cloud_account_auth: bool = False,
        entrypoint: str | None = None,
        path_mappings: dict[str, str] | None = None,
        max_runtime: int | None = None,
        max_run_attempts: int | None = None,
        reuse_snapshot: bool = True,
        scratch_disks: dict[str, int] | None = None,
        placement_group_id: str | None = None,
        num_machines: int = 1,
        tags: list[str] | None = None,
    ) -> Job:
        _state.call(
            "Job.run",
            name=name,
            machine=str(machine),
            studio=getattr(studio, "name", studio),
            teamspace=getattr(teamspace, "name", teamspace),
            command=command,
            env=env,
            interruptible=interruptible,
            max_run_attempts=max_run_attempts,
            image=image,
        )
        _state.maybe_raise("Job.run")
        actual = _state.STATE.pop("rename_to", None) or name
        _state.STATE.setdefault("jobs", {})[actual] = {"status": "Pending", "total_cost": 0.0}
        if actual != name:
            _state.STATE["jobs"].setdefault(name, {"status": "Running", "total_cost": 0.0})
        _state.save()
        _state.maybe_raise("Job.run.after")  # the real run() keeps calling the API (job.link)
        job = cls(actual, teamspace=teamspace, _fetch_job=False)
        job._job = job._raw()
        return job
