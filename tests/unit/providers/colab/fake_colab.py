#!/usr/bin/env python3
"""Stand-in for the `colab` CLI (google-colab-cli 0.7.2) used by the colab adapter tests.

Invoked as `[sys.executable, fake_colab.py, <global flags>, <command>, ...]` exactly like
the real CLI. Messages copy the real CLI's wording (commands/session.py, execution.py,
files.py), because the adapter classifies failures by text.

- Session state: `<config>.sim.json` next to the `--config` file.
- The "VM" is the local filesystem. `exec -f` runs the file with this Python in a fresh
  subprocess (a PATH-first fake `nvidia-smi` reports a Tesla T4); `upload`/`download` copy
  files (upload needs the parent dir, like Jupyter's contents API); `stop` SIGTERMs the
  runner recorded in `<remote_root>/<session>/launched.json`, then SIGKILLs it.
- Like the real CLI, every call appends to `$HOME/.config/colab-cli/colab.log` (a request
  URL with a `colab-runtime-proxy-token`) and exec/upload/download/stop calls append an
  event to `$HOME/.config/colab-cli/history/<session>.jsonl` (exec events carry the code AND
  its output). Never under the real user's home: the sim skips both when HOME is the
  account's own home directory.
- Fault injection: `<config dir>/sim-control.json`, re-read on every call:
    {"remote_root": "...",          # where run dirs live (for stop)
     "new": "quota|busy|scope|auth|hang|crash|slow",
     "new_delay": s,                 # "slow": seconds before the session is registered
     "exec_fail": n,                 # next n exec calls fail with a network error
     "stop_fail": n,                 # next n stop calls fail with a network error
     "lose": ["session", ...],       # these sessions were reclaimed by colab
     "others": [["?", "T4"], ...],   # foreign sessions listed by `sessions`
     "sessions_mode": "auth|network"} # how `sessions` fails
- Every call is appended to `<config dir>/sim-calls.jsonl` as {"argv": [...], "auth",
  "home", "cloudsdk_config"}.
"""

from __future__ import annotations

import json
import os
import pwd
import shutil
import signal
import subprocess
import sys
import time


def _load(path, default):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def _save(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh)
    os.replace(tmp, path)


def _cli_dir():
    home = os.environ.get("HOME", "")
    real = pwd.getpwuid(os.getuid()).pw_dir
    if not home or os.path.realpath(home) == os.path.realpath(real):
        return None  # never write into the real user's ~/.config/colab-cli from tests
    return os.path.join(home, ".config", "colab-cli")


def _history(session, event, **data):
    base = _cli_dir()
    if base is None or not session:
        return
    os.makedirs(os.path.join(base, "history"), exist_ok=True)
    with open(os.path.join(base, "history", session + ".jsonl"), "a") as fh:
        fh.write(json.dumps({"event_type": event, **data}) + "\n")


def _debug_log(line):
    base = _cli_dir()
    if base is None:
        return
    os.makedirs(base, exist_ok=True)
    with open(os.path.join(base, "colab.log"), "a") as fh:
        fh.write(line + "\n")


def main(argv):
    config = os.path.expanduser("~/.config/colab-cli/sessions.json")
    auth = "oauth2"
    rest = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--config":
            config = argv[i + 1]
            i += 2
            continue
        if a.startswith("--auth="):
            auth = a.split("=", 1)[1]
            i += 1
            continue
        rest = argv[i:]
        break
    cfg_dir = os.path.dirname(config)
    os.makedirs(cfg_dir, exist_ok=True)
    with open(os.path.join(cfg_dir, "sim-calls.jsonl"), "a") as fh:
        fh.write(
            json.dumps(
                {
                    "argv": argv,
                    "auth": auth,
                    "home": os.environ.get("HOME"),
                    "cloudsdk_config": os.environ.get("CLOUDSDK_CONFIG"),
                }
            )
            + "\n"
        )
    _debug_log(
        'urllib3.connectionpool - DEBUG - https://sim:443 "GET /api/contents/x?authuser=0'
        '&colab-runtime-proxy-token=SIMPROXYTOKEN HTTP/1.1" 200'
    )
    control_path = os.path.join(cfg_dir, "sim-control.json")
    control = _load(control_path, {})
    state_path = config + ".sim.json"
    state = _load(state_path, {"sessions": {}})
    sessions = state["sessions"]
    if not rest:
        print("usage: colab COMMAND")
        return 2
    cmd, args = rest[0], rest[1:]

    def opt(name, default=None):
        for j, v in enumerate(args):
            if v == name and j + 1 < len(args):
                return args[j + 1]
        return default

    positional = []
    skip = False
    for v in args:
        if skip:
            skip = False
            continue
        if v in ("-s", "--session", "-f", "--file", "--timeout", "--gpu", "--tpu"):
            skip = True
            continue
        positional.append(v)

    name = opt("-s") or opt("--session")

    def lost_check():
        if name in control.get("lose", []) and name in sessions:
            del sessions[name]
            _save(state_path, state)
            print(f"[colab] Session '{name}' appears to be lost (404/401). Cleaning up.")
            return True
        return False

    if cmd == "version":
        print("Version: 0.7.2-sim")
        return 0

    if cmd == "new":
        mode = control.get("new")
        gpu = (opt("--gpu") or "").upper()
        print(f"[colab] Creating session '{name}'...")
        if mode == "quota":
            sys.stderr.write(
                f"[colab] Backend rejected accelerator '{gpu}'. You may not have quota or "
                "entitlement for this accelerator on your account.\n"
            )
            return 1
        if mode == "busy":
            sys.stderr.write(
                "[colab] Allocation refused (precondition failed). This can mean too many "
                "active sessions, or a temporary usage or capacity limit.\n"
            )
            return 1
        if mode == "scope":
            sys.stderr.write(
                "[colab] Keep-alive pre-flight failed: your credentials are missing an OAuth "
                "scope required by Colab.\n"
            )
            return 1
        if mode == "auth":
            sys.stderr.write("No valid default credentials found. To authenticate, run:\n")
            return 1
        if mode == "hang":
            sessions[name] = {"gpu": gpu}
            _save(state_path, state)
            time.sleep(3600)
            return 1
        if mode == "crash":
            sys.stderr.write("Traceback (most recent call last):\nKeyError: 'endpoint'\n")
            return 1
        if mode == "slow":  # the assign takes a while; the session is registered after it
            time.sleep(float(control.get("new_delay", 3)))
            state = _load(state_path, {"sessions": {}})
            state["sessions"][name] = {"gpu": gpu}
            _save(state_path, state)
            print("[colab] Session READY.")
            return 0
        sessions[name] = {"gpu": gpu}
        _save(state_path, state)
        print("[colab] Session READY.")
        return 0

    if cmd == "sessions":
        mode = control.get("sessions_mode")
        if mode == "auth":  # the real CLI swallows the auth failure when its store is empty
            sys.stderr.write("No valid default credentials found. To authenticate, run:\n")
            print("[colab] No active sessions found on server.")
            return 0
        if mode == "network":
            sys.stderr.write("requests.exceptions.ConnectionError: Max retries exceeded\n")
            return 1
        rows = [(n, s["gpu"] or "CPU") for n, s in sorted(sessions.items())]
        rows += [tuple(o) for o in control.get("others", [])]
        if not rows:
            print("[colab] No active sessions found on server.")
            return 0
        for n, hw in rows:
            print(f"[{n}] ep-{n} | Hardware: {hw} | Shape: Standard | Variant: GPU")
        return 0

    if cmd == "stop":
        if name not in sessions:
            print(f"[colab] Session '{name}' not found.")
            return 0
        if control.get("stop_fail", 0) > 0:
            control["stop_fail"] -= 1
            _save(control_path, control)
            sys.stderr.write("requests.exceptions.ConnectionError: Max retries exceeded\n")
            return 1
        print(f"[colab] Stopping session '{name}'...")
        _kill_runner(control.get("remote_root"), name)
        del sessions[name]
        _save(state_path, state)
        _history(name, "session_terminated", reason="stopped")
        print("[colab] Session terminated.")
        return 0

    if cmd in ("exec", "upload", "download"):
        if lost_check():
            return 1
        if name not in sessions:
            print(f"[colab] Session '{name}' not found.")
            return 1

    if cmd == "exec":
        if control.get("exec_fail", 0) > 0:
            control["exec_fail"] -= 1
            _save(control_path, control)
            sys.stderr.write("requests.exceptions.ConnectionError: Max retries exceeded\n")
            return 1
        path = opt("-f") or opt("--file")
        timeout = float(opt("--timeout", "30"))
        env = {k: v for k, v in os.environ.items() if not k.startswith("GPU_ROUTER_")}
        simbin = os.path.join(cfg_dir, "simbin")
        _ensure_nvidia_smi(simbin)
        env["PATH"] = simbin + os.pathsep + env.get("PATH", "")
        try:
            proc = subprocess.run(
                [sys.executable, path],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                cwd=cfg_dir,
            )
        except subprocess.TimeoutExpired:
            sys.stderr.write("Traceback (most recent call last):\nTimeoutError: timed out\n")
            return 1
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        with open(path) as fh:
            code = fh.read()
        _history(name, "execution", code=code, outputs=[{"text": proc.stdout + proc.stderr}])
        return 0  # like the real CLI: kernel errors do not change the exit code

    if cmd == "upload":
        local, remote_path = positional[0], positional[1]
        if not os.path.isfile(local):
            print(f"[colab] Local file '{local}' not found.")
            return 1
        if not os.path.isdir(os.path.dirname(remote_path)):
            print(
                "[colab] Upload failed: 404 Client Error: Not Found for url: "
                f"https://sim/api/contents{remote_path}?authuser=0"
                "&colab-runtime-proxy-token=SIMTOKEN123"
            )
            return 1
        shutil.copyfile(local, remote_path)
        print(f"[colab] Uploaded '{local}' to '{remote_path}'")
        return 0

    if cmd == "download":
        remote_path, local = positional[0], positional[1]
        if not os.path.isfile(remote_path):
            print(f"[colab] Download failed: File or directory not found: {remote_path}")
            return 1
        shutil.copyfile(remote_path, local)
        print(f"[colab] Downloaded '{remote_path}' to '{local}'")
        return 0

    print(f"[colab] unsupported in the simulator: {cmd}")
    return 2


def _ensure_nvidia_smi(simbin):
    path = os.path.join(simbin, "nvidia-smi")
    if os.path.exists(path):
        return
    os.makedirs(simbin, exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as fh:
        fh.write("#!/bin/sh\necho 'Tesla T4, 15360 MiB'\n")
    os.chmod(tmp, 0o755)
    os.replace(tmp, path)


def _kill_runner(remote_root, name):
    if not remote_root:
        return
    marker = os.path.join(remote_root, name, "launched.json")
    try:
        with open(marker) as fh:
            pid = int(json.load(fh)["pid"])
    except (OSError, ValueError, KeyError):
        return
    for sig, wait in ((signal.SIGTERM, 3.0), (signal.SIGKILL, 0.0)):
        try:
            os.killpg(pid, sig)
        except OSError:
            return
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            try:
                os.killpg(pid, 0)
            except OSError:
                return
            time.sleep(0.05)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
