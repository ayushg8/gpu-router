"""gpu-router's bridge into the lightning-sdk. Runs in the SDK's own environment.

This file is executed as a script by `sdk.SdkBridge` with the interpreter of an isolated
env that has `lightning-sdk` installed (`uv tool install lightning-sdk`, or `uv run
--no-project --with lightning-sdk==<pin>`, D3). It never imports gpu_router and never runs
inside the daemon's interpreter.

Protocol: one JSON document on stdin, `{"op": "<name>", "params": {...}}`; exactly one
result line on stdout, `@@GRL:<base64 JSON>`, which is `{"ok": true, "result": {...}}` or
`{"ok": false, "kind": "<class>", "error": "<text>", "stage": "pre"|"run"|"post"|null,
"status": <http status or null>}`. Everything the SDK prints goes to stderr.

Credentials arrive ONLY as LIGHTNING_USER_ID / LIGHTNING_API_KEY in this process's
environment (the adapter injects them into the child env, invariant 12). Before the SDK is
imported, HOME, LIGHTNING_CREDENTIAL_PATH and LIGHTNING_SETTINGS_PATH are pointed at
GR_LIGHTNING_HOME (a private dir under the gpu-router data dir), so the SDK can neither
read nor write ~/.lightning, and the version check is off. Without credentials the SDK
would open a browser and start a local auth server (NOTES.md "Login"): the driver refuses
to import it at all then.

Error kinds (the adapter maps them to the taxonomy): auth, verify, not_found, rate, quota,
invalid, config, permanent, unavailable, sdk.

Python 3.9+ (the SDK's env), stdlib + lightning_sdk only.
"""

from __future__ import annotations

import base64
import contextlib
import importlib
import json
import os
import re
import socket
import sys
import threading
import traceback
from datetime import datetime, timezone
from typing import Any

#: not datetime.UTC: the driver may run under an SDK env older than Python 3.11
_UTC = timezone.utc  # noqa: UP017

MARKER = "@@GRL:"
TERMINAL = ("Completed", "Failed", "Stopped")
ERROR_CHARS = 600


class DriverError(Exception):
    """A classified failure: `kind` tells the adapter which taxonomy class to raise."""

    def __init__(
        self, kind: str, message: str, *, status: int | None = None, stage: str | None = None
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.status = status
        self.stage = stage


# --------------------------------------------------------------------------- setup


def prepare_env() -> None:
    """Private HOME + SDK paths, before lightning_sdk is imported (it reads them then)."""
    home = os.environ.get("GR_LIGHTNING_HOME")
    if home:
        os.makedirs(home, mode=0o700, exist_ok=True)
        os.environ["HOME"] = home
        dot = os.path.join(home, ".lightning")
        os.environ["LIGHTNING_CREDENTIAL_PATH"] = os.path.join(dot, "credentials.json")
        os.environ["LIGHTNING_SETTINGS_PATH"] = os.path.join(dot, "settings.json")
    os.environ["LIGHTNING_DISABLE_VERSION_CHECK"] = "1"
    os.environ["BROWSER"] = "true"  # webbrowser.open() must never reach a real browser
    os.environ.setdefault("TQDM_DISABLE", "1")


def have_credentials() -> bool:
    env = os.environ
    return bool(env.get("LIGHTNING_API_KEY") and env.get("LIGHTNING_USER_ID")) or bool(
        env.get("LIGHTNING_AUTH_TOKEN")
    )


def sdk() -> Any:
    if not have_credentials():
        raise DriverError("auth", "no lightning credentials were passed to the SDK")
    return importlib.import_module("lightning_sdk")


def resolve_mod() -> Any:
    return importlib.import_module("lightning_sdk.utils.resolve")


# --------------------------------------------------------------------------- helpers


_ISO = re.compile(r"^(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}:\d{2})(?:\.(\d+))?(Z|[+-]\d{2}:?\d{2})?$")


def epoch(value: Any) -> float | None:
    """Epoch seconds from the SDK's timestamps: the live API hands back ISO-8601 strings
    (`Job.started_at` = "2026-09-25T07:00:43Z", `V1Job.created_at` with 6+ fraction digits,
    seen 2026-09-25), older models a datetime. Anything else = None."""
    if isinstance(value, datetime):
        aware: datetime = value if value.tzinfo is not None else value.replace(tzinfo=_UTC)
        return float(aware.timestamp())
    if not isinstance(value, str):
        return None
    m = _ISO.match(value.strip())
    if m is None:
        return None
    day, clock, frac, zone = m.groups()
    try:
        base = datetime.strptime(f"{day}T{clock}", "%Y-%m-%dT%H:%M:%S").replace(tzinfo=_UTC)
    except ValueError:
        return None
    offset = 0
    if zone and zone != "Z":
        sign = -1 if zone[0] == "-" else 1
        digits = zone[1:].replace(":", "")
        offset = sign * (int(digits[:2]) * 3600 + int(digits[2:]) * 60)
    seconds = base.timestamp() - offset
    return seconds + (float(f"0.{frac}") if frac else 0.0)


def safe(fn: Any, default: Any = None) -> Any:
    try:
        return fn()
    except Exception:
        return default


def run_bounded(fn: Any, wait_s: float) -> tuple[bool, Any]:
    """Run fn() on a daemon thread; (finished, result). A slow SDK call can never hold
    the driver past its budget (the adapter also kills the whole process group)."""
    box: dict[str, Any] = {}

    def work() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # re-raised on the caller's thread
            box["error"] = exc

    t = threading.Thread(target=work, daemon=True)
    t.start()
    t.join(max(0.1, wait_s))
    if t.is_alive():
        return False, None
    if "error" in box:
        raise box["error"]
    return True, box.get("value")


def teamspace(mod: Any, params: dict[str, Any]) -> Any:
    """The teamspace to work in: `params["teamspace"]` ("owner/name"), else the authed
    user's only teamspace. Several and none configured = a config problem naming them."""
    name = params.get("teamspace")
    if name:
        try:
            return mod.Teamspace(str(name))
        except ValueError as exc:
            raise DriverError("config", f"lightning teamspace {name!r}: {exc}") from None
    user = resolve_mod()._get_authed_user()
    spaces = list(user.teamspaces)
    if len(spaces) == 1:
        return spaces[0]
    names = ", ".join(f"{t.owner.name}/{t.name}" for t in spaces) or "none"
    raise DriverError(
        "config",
        f"cannot pick a lightning teamspace automatically (yours: {names}); set "
        "providers.lightning.teamspace to owner/name",
    )


def ts_slug(ts: Any) -> str:
    return f"{ts.owner.name}/{ts.name}"


def find_job(mod: Any, ts: Any, name: str) -> Any:
    try:
        return mod.Job(name, teamspace=ts)
    except ValueError as exc:
        if "does not exist" in str(exc):
            return None
        raise


def summary(job: Any) -> dict[str, Any]:
    """One status read (the SDK refetches the job on every property access, so the
    refresh is pinned after the first read)."""
    status = str(job.status)
    with contextlib.suppress(Exception):
        job._prevent_refetch_latest = True
    raw: Any = getattr(job, "_job", None)
    return {
        "name": job.name,
        "id": safe(lambda: job.id),
        "status": status,
        "started_at": epoch(safe(lambda: job.started_at)),
        "stopped_at": epoch(safe(lambda: job.stopped_at)),
        "total_cost": safe(lambda: float(job.total_cost)),
        "message": safe(lambda: str(raw.message or "")) or None,
        "server_error": safe(lambda: str(raw.server_error or "")) or None,
        "interrupted": bool(safe(lambda: raw.interruption_notice_received, False)),
    }


def log_lines(job: Any, wait_s: float, *, tail: int | None = None) -> dict[str, Any]:
    """The job's log so far as lines (snapshot; the SDK reads saved logs, else tails the
    live stream until it is quiet). `tail`: only the last N lines (fast, server-side)."""

    def read() -> str:
        if tail is not None:
            return str(job.logs(follow=False, tail=int(tail)))
        return str(job.logs(follow=False))

    try:
        done, text = run_bounded(read, wait_s)
    except RuntimeError as exc:  # "Logs are not available while the job is Pending."
        return {"lines": [], "note": str(exc)[:200]}
    if not done:
        return {"lines": [], "note": f"log read took longer than {wait_s:.0f}s", "timeout": True}
    return {"lines": str(text or "").splitlines()}


def verdict_log(job: Any, params: dict[str, Any]) -> dict[str, Any]:
    """A finished job's log for its verdict: the whole log when it arrives within
    `log_wait_s`, else its last `log_tail` lines (the exit line and the launcher's wall
    mark are at the end) flagged `timeout` + `tail`, so the adapter never mistakes a slow
    read for an empty log."""
    full = log_lines(job, float(params.get("log_wait_s", 20)))
    tail = params.get("log_tail")
    if not full.get("timeout") or not tail:
        return full
    try:
        part = log_lines(job, float(params.get("tail_wait_s", 8)), tail=int(tail))
    except Exception as exc:  # the tail is a fallback: its failure must not hide the timeout
        return {**full, "tail_error": f"{type(exc).__name__}: {_text(exc)[:160]}"}
    if part.get("timeout") or not part.get("lines"):
        return full
    return {"lines": part["lines"], "timeout": True, "tail": True, "note": full.get("note")}


# --------------------------------------------------------------------------- ops


def op_whoami(params: dict[str, Any]) -> dict[str, Any]:
    mod = sdk()
    user = resolve_mod()._get_authed_user()
    spaces = [f"{t.owner.name}/{t.name}" for t in user.teamspaces]
    chosen = None
    if params.get("teamspace"):
        chosen = ts_slug(teamspace(mod, params))
    elif len(spaces) == 1:
        chosen = spaces[0]
    return {
        "user": user.name,
        "teamspaces": spaces,
        "teamspace": chosen,
        "sdk_version": getattr(mod, "__version__", None),
    }


def balance_info() -> dict[str, Any]:
    """The account's credit balance (`balance`) and its settled spend (`total_spent`, which
    lags: a finished job is already out of the balance before it shows there)."""
    rest = importlib.import_module("lightning_sdk.lightning_cloud.rest_client")
    client = rest.LightningClient(retry=False)
    resp = client.billing_service_get_user_balance()
    out: dict[str, Any] = {}
    for key in ("balance", "total_spent"):
        value = getattr(resp, key, None)
        if value is not None:
            out[key] = float(value)
    return out


def user_balance() -> float | None:
    return balance_info().get("balance")


def op_submit(params: dict[str, Any]) -> dict[str, Any]:
    """Idempotent per job name: an existing job is returned, never a second one.

    Every failure leaves with the stage it happened in (invariant 6, rule A3): "pre" =
    before Job.run was called (nothing can exist), "run" = Job.run raised and a lookup
    right after found no job of this name, "post" = the job may exist (Job.run was called
    and it could not be ruled out). Only "pre" and a 4xx at "run" are definitive."""
    at = ["pre"]
    try:
        return _submit(params, at)
    except DriverError as exc:
        if exc.stage is None:
            exc.stage = at[0]
        raise
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        raise _classified(exc, stage=at[0]) from None


#: find_job's answer when the lookup itself failed or did not finish
_UNKNOWN = object()


def _find_after_error(mod: Any, ts: Any, name: str, wait_s: float = 30) -> Any:
    """The job of this name after Job.run raised: the job, None (Lightning says it does
    not exist) or _UNKNOWN (the lookup failed too, so it may exist)."""
    try:
        done, found = run_bounded(lambda: find_job(mod, ts, name), wait_s)
    except Exception:
        return _UNKNOWN
    return found if done else _UNKNOWN


def _submit(params: dict[str, Any], at: list[str]) -> dict[str, Any]:
    mod = sdk()
    ts = teamspace(mod, params)
    name = str(params["name"])
    existing = find_job(mod, ts, name)
    if existing is not None:
        at[0] = "post"  # it exists: a failed status read of it must not look definitive
        found = safe(lambda: summary(existing), None) or {"name": name, "status": None}
        return {"existed": True, "teamspace": ts_slug(ts), **found}
    machine = getattr(mod.Machine, str(params["machine"]), None)
    if machine is None:
        raise DriverError("invalid", f"lightning has no machine {params['machine']!r}")
    min_balance = params.get("min_balance")
    if min_balance is not None:
        balance = safe(user_balance)
        if balance is not None and balance < float(min_balance):
            raise DriverError("quota", f"lightning credits are used up ({balance:.2f} left)")
    with resolve_mod().skip_studio_setup():  # never starts the Studio's keep-alive
        try:
            studio = mod.Studio(
                name=str(params["studio"]),
                teamspace=ts,
                create_ok=bool(params.get("create_studio", True)),
            )
        except ValueError as exc:
            raise DriverError("config", f"lightning studio: {exc}") from None
    account = safe(lambda: studio.cloud_account)
    for item in params.get("files") or []:
        ts.upload_file(
            item["local"], remote_path=item["remote"], progress_bar=False, cloud_account=account
        )
    at[0] = "post"  # from here on the job may exist
    result: dict[str, Any] = {"existed": False, "teamspace": ts_slug(ts)}
    try:
        job = mod.Job.run(
            name=name,
            machine=machine,
            studio=studio,
            teamspace=ts,
            command=str(params["command"]),
            env={str(k): str(v) for k, v in (params.get("env") or {}).items()},
            interruptible=bool(params.get("interruptible", False)),
            max_run_attempts=1,
        )
    except Exception as exc:
        # Job.run creates the job (submit_job) and then keeps talking to Lightning (it
        # reads job.link, which fetches the Studio): an error can come after the create.
        traceback.print_exc(file=sys.stderr)
        err = _classified(exc)
        found = _find_after_error(mod, ts, name)
        if found is _UNKNOWN:
            err.stage = "post"
            raise err from None
        if found is None:
            err.stage = "run"  # the create call failed and no job of this name exists
            raise err from None
        job = found
        result["run_error"] = str(err)[:ERROR_CHARS]
    if job.name != name:
        # Lightning renames a job whose name is taken: someone (an earlier submit whose
        # answer was lost) created ours meanwhile. Stop the duplicate, return the original.
        with contextlib.suppress(Exception):
            run_bounded(job.stop, 20)
        with contextlib.suppress(Exception):
            run_bounded(job.delete, 10)
        original = find_job(mod, ts, name)
        if original is None:
            raise DriverError("unavailable", f"lightning renamed job {name} to {job.name}")
        result["duplicate_stopped"] = job.name
        job = original
        result["existed"] = True
    # the job exists now: a failed status read must not turn into a failed submit (the
    # adapter would place the job elsewhere while this one runs)
    result.update(safe(lambda: summary(job), None) or {"name": name, "status": None})
    result["link"] = safe(lambda: job.link)
    return result


def op_status(params: dict[str, Any]) -> dict[str, Any]:
    mod = sdk()
    ts = teamspace(mod, params)
    job = find_job(mod, ts, str(params["name"]))
    if job is None:
        raise DriverError("not_found", f"lightning has no job {params['name']}")
    out = summary(job)
    if params.get("with_log") and out["status"] in TERMINAL:
        out["log"] = verdict_log(job, params)
    return out


def op_logs(params: dict[str, Any]) -> dict[str, Any]:
    mod = sdk()
    ts = teamspace(mod, params)
    job = find_job(mod, ts, str(params["name"]))
    if job is None:
        raise DriverError("not_found", f"lightning has no job {params['name']}")
    out = summary(job)
    out["log"] = log_lines(job, float(params.get("log_wait_s", 20)))
    return out


def op_stop(params: dict[str, Any]) -> dict[str, Any]:
    mod = sdk()
    ts = teamspace(mod, params)
    job = find_job(mod, ts, str(params["name"]))
    if job is None:
        return {"missing": True}
    status = str(job.status)
    if status in TERMINAL:
        return {"status": status, "already": True}
    # Job.stop() sends the stop, then waits in a sleep loop until the job is terminal:
    # the wait is bounded here, the stop request goes out in the first second.
    done, _ = run_bounded(job.stop, float(params.get("wait_s", 20)))
    fresh = find_job(mod, ts, str(params["name"]))
    after = str(fresh.status) if fresh is not None else "Stopped"
    return {"status": after, "confirmed": bool(done)}


def op_fetch(params: dict[str, Any]) -> dict[str, Any]:
    """Outputs archive: the drive copy the launcher uploaded, else the job's artifacts."""
    mod = sdk()
    ts = teamspace(mod, params)
    dest = str(params["dest"])
    os.makedirs(dest, exist_ok=True)
    archive = os.path.join(dest, "outputs.tar.gz")
    notes: list[str] = []
    remote = f"{params['drive_dir']}/out/outputs.tar.gz"
    try:
        ts.download_file(remote, archive)
        if os.path.isfile(archive):
            return {"source": "drive", "archive": archive, "notes": notes}
    except Exception as exc:
        notes.append(f"drive: {type(exc).__name__}: {_text(exc)[:160]}")
    job = find_job(mod, ts, str(params["name"]))
    if job is None:
        raise DriverError("not_found", f"lightning has no job {params['name']}")
    entries = safe(lambda: job.list_artifacts(recursive=True), []) or []
    wanted = [
        e.path
        for e in entries
        if not e.is_dir and e.path.endswith("outputs.tar.gz") and "gpu-router" in e.path
    ]
    if wanted:
        folder = os.path.dirname(wanted[0])
        target = os.path.join(dest, "artifacts")
        job.download_artifacts(target, path=folder)
        found = os.path.join(target, "outputs.tar.gz")
        if os.path.isfile(found):
            os.replace(found, archive)
            return {"source": "artifacts", "archive": archive, "notes": notes}
        notes.append("artifacts: archive listed but not downloaded")
    else:
        notes.append(f"artifacts: {len(entries)} entries, no outputs archive")
    return {"source": None, "archive": None, "notes": notes}


def op_cleanup(params: dict[str, Any]) -> dict[str, Any]:
    """Best-effort removal of drive paths we uploaded (secrets files, attempt folders)."""
    mod = sdk()
    ts = teamspace(mod, params)
    fs = importlib.import_module("lightning_sdk.filesystem").Filesystem()
    removed: list[str] = []
    failed: list[str] = []
    for path in params.get("paths") or []:
        uri = f"lit://{ts.owner.name}/{ts.name}/{str(path).strip('/')}"
        try:
            fs.rm(uri, recursive=bool(params.get("recursive", False)))
            removed.append(path)
        except FileNotFoundError:
            removed.append(path)
        except Exception as exc:
            failed.append(f"{path}: {type(exc).__name__}")
    return {"removed": removed, "failed": failed}


def op_quota(params: dict[str, Any]) -> dict[str, Any]:
    """Credits: the account balance when the API answers, plus what jobs in the teamspace
    cost since `since` (epoch) and the per-hour rates of the GPUs we use."""
    mod = sdk()
    ts = teamspace(mod, params)
    out: dict[str, Any] = {"teamspace": ts_slug(ts)}
    try:
        info = balance_info()
        out["balance"] = info.get("balance")
        if "total_spent" in info:
            out["total_spent"] = info["total_spent"]
    except Exception as exc:
        out["balance_error"] = f"{type(exc).__name__}: {_text(exc)[:160]}"
    since = params.get("since")
    spent = 0.0
    counted = 0
    for job in safe(lambda: list(ts.list_jobs()), []) or []:
        raw = getattr(job, "_job", None)
        created = epoch(getattr(raw, "created_at", None))
        if since is not None and created is not None and created < float(since):
            continue
        cost = getattr(raw, "total_cost", None)
        if isinstance(cost, (int, float)):
            spent += float(cost)
            counted += 1
    out["jobs_cost"] = spent
    out["jobs_counted"] = counted
    rates: dict[str, Any] = {}
    for gpu in params.get("machines") or []:
        found = safe(lambda g=gpu: ts.list_machines(machine=g), []) or []
        for m in found:
            rates[str(gpu)] = {
                "cost": safe(lambda m=m: float(m.cost)),
                "interruptible_cost": safe(lambda m=m: float(m.interruptible_cost)),
                "wait_time": safe(lambda m=m: m.wait_time),
            }
            break
    out["rates"] = rates
    return out


OPS = {
    "whoami": op_whoami,
    "submit": op_submit,
    "status": op_status,
    "logs": op_logs,
    "stop": op_stop,
    "fetch": op_fetch,
    "cleanup": op_cleanup,
    "quota": op_quota,
}


# --------------------------------------------------------------------------- errors


_RESPONSE_STATUS = re.compile(r"response: (\d{3})\b")
_QUOTA_WORDS = ("insufficient", "balance", "credit", "quota", "out of funds", "payment")
_VERIFY_WORDS = ("phone", "verif")
_NET_NAMES = (
    "ConnectionError",
    "ConnectTimeout",
    "ReadTimeout",
    "Timeout",
    "MaxRetryError",
    "ProtocolError",
    "NewConnectionError",
    "SSLError",
    "RemoteDisconnected",
)


def _text(exc: BaseException) -> str:
    """Status, reason and body of an HTTP error (never str() of an ApiException: that
    also prints the response headers, cookies included); str() of anything else."""
    status = getattr(exc, "status", None)
    if isinstance(status, int):
        body = getattr(exc, "body", None)
        if isinstance(body, bytes):
            body = body.decode("utf-8", "replace")
        parts = [f"({status})", str(getattr(exc, "reason", None) or ""), str(body or "")]
    else:
        parts = [str(exc)]
    return " ".join(" ".join(parts).split())[:ERROR_CHARS]


def _classified(exc: BaseException, *, stage: str | None = None) -> DriverError:
    """Map an SDK / network exception to a kind. `stage` = where it happened in submit
    ("run" = the create call itself)."""
    if isinstance(exc, DriverError):
        return exc
    status = getattr(exc, "status", None)
    status = status if isinstance(status, int) else None
    text = _text(exc)
    low = text.lower()
    name = type(exc).__name__
    if status is None:
        # the SDK's retry wrapper gives up with `Exception("The <call> request failed to
        # reach the server, response: 403.")`: the status only survives in the text
        found = _RESPONSE_STATUS.search(text)
        status = int(found.group(1)) if found else None
    err: DriverError
    if "authentication failed" in low:
        # the SDK turns a 401 into ConnectionError("Authentication failed. Please run
        # `lightning login`.") (lightning_cloud/rest_client.request_auth_warning_wrapper)
        # ... but gpu-router keeps its own copy of the key: `lightning login` alone never
        # fixes it (a 2026-10-04 session told the user to run it), `gpu login lightning` does
        why = text.replace("Please run `lightning login`.", "").strip() or text
        err = DriverError(
            "auth",
            f"lightning rejected the stored API key ({why}); sign in again with "
            "`gpu login lightning`",
            status=401,
        )
    elif status in (401,):
        err = DriverError("auth", f"lightning rejected the credentials: {text}", status=status)
    elif status == 403 or isinstance(exc, PermissionError):
        kind = "verify" if any(w in low for w in _VERIFY_WORDS) else "auth"
        err = DriverError(kind, f"lightning refused access: {text}", status=status)
    elif status == 404:
        err = DriverError("not_found", f"lightning: not found: {text}", status=status)
    elif status == 429:
        err = DriverError("rate", f"lightning is rate limiting: {text}", status=status)
    elif status is not None and status >= 500:
        err = DriverError("unavailable", f"lightning server error {status}: {text}", status=status)
    elif status is not None and 400 <= status < 500:
        if any(w in low for w in _QUOTA_WORDS):
            kind = "quota"
        elif any(w in low for w in _VERIFY_WORDS):
            kind = "verify"
        else:
            kind = "invalid"
        err = DriverError(kind, f"lightning refused the request ({status}): {text}", status=status)
    elif name in _NET_NAMES or isinstance(exc, (socket.timeout, ConnectionError, TimeoutError)):
        err = DriverError("unavailable", f"lightning is unreachable: {name}: {text}")
    elif isinstance(exc, ValueError):
        low_kind = "quota" if any(w in low for w in _QUOTA_WORDS) else "invalid"
        err = DriverError(low_kind, f"lightning: {text}")
    else:
        err = DriverError("sdk", f"lightning SDK error: {name}: {text}")
    if stage is not None:
        err.stage = stage
    return err


# --------------------------------------------------------------------------- main


def emit(stream: Any, doc: dict[str, Any]) -> None:
    data = json.dumps(doc, separators=(",", ":"), default=str).encode("utf-8")
    stream.write(MARKER + base64.b64encode(data).decode("ascii") + "\n")
    stream.flush()


def main() -> int:
    real_stdout = sys.stdout
    sys.stdout = sys.stderr  # the SDK prints; only the marker line reaches stdout
    try:
        request = json.loads(sys.stdin.read() or "{}")
        op = str(request.get("op") or "")
        params = request.get("params") or {}
        if op not in OPS:
            emit(real_stdout, {"ok": False, "kind": "invalid", "error": f"unknown op {op!r}"})
            return 2
        prepare_env()
        try:
            result = OPS[op](params)
        except DriverError as exc:
            emit(
                real_stdout,
                {
                    "ok": False,
                    "kind": exc.kind,
                    "error": str(exc)[:ERROR_CHARS],
                    "status": exc.status,
                    "stage": exc.stage or ("post" if op == "submit" else None),
                },
            )
            return 1
        except Exception as exc:
            err = _classified(exc)
            traceback.print_exc(file=sys.stderr)
            emit(
                real_stdout,
                {
                    "ok": False,
                    "kind": err.kind,
                    "error": str(err)[:ERROR_CHARS],
                    "status": err.status,
                    "stage": err.stage or ("post" if op == "submit" else None),
                },
            )
            return 1
        emit(real_stdout, {"ok": True, "result": result})
        return 0
    finally:
        sys.stdout = real_stdout


if __name__ == "__main__":
    sys.exit(main())
