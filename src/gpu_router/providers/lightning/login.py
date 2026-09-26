"""`gpu login lightning`: store Lightning AI credentials in the Keychain (phase 7a).

Sources, never argv (invariant 12):

- a no-echo prompt for the user id and the API key (lightning.ai > Settings > Keys);
- stdin (`--stdin`): the `lightning login` JSON document (`{"user_id": ..., "api_key": ...}`)
  or two lines / two words: user id, then API key;
- `--import`: the file `lightning login` wrote (~/.lightning/credentials.json). On a
  terminal, when that file exists, the prompt offers the import first;
- `--browser`: lightning.ai's own CLI sign-in (the flow `lightning login` uses): the
  browser opens lightning.ai/sign-in?redirectTo=http://localhost:<port>/login-complete
  ?state=<random> and lightning.ai sends the user id and API key back to a one-shot server
  on 127.0.0.1. The values stay in memory (the server logs nothing and redirects the
  browser to a URL without them) and nothing is written to ~/.lightning. The callback is
  accepted only with that request's `state` value, a localhost Host header and (when the
  browser sends it) Sec-Fetch-Mode: navigate, and on a terminal the verified account is
  shown and must be confirmed [y/N] before anything is stored (review fix: any page or
  process that reached the port first could plant its own credentials).

Values go to the Keychain as LIGHTNING_USER_ID / LIGHTNING_API_KEY through
gpu_router.secrets and are never printed. Unless `--no-check`, they are verified first with
one SDK call (who am I, which teamspace): a rejection stores nothing; an unreachable
lightning.ai stores them with a note. The first check may take a minute when uv has to
fetch the SDK.
"""

from __future__ import annotations

import hmac
import json
import os
import secrets as _stdlib_secrets
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import TracebackType
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit

from gpu_router.errors import AdapterError, AuthRequired, InvalidRequest
from gpu_router.providers.lightning import credentials as creds_mod

__all__ = ["BrowserSignIn", "LoginOutcome", "parse_pair", "run_login", "verify"]

VERIFY_TIMEOUT_S = 240.0  # the first check may make uv fetch the SDK
BROWSER_TIMEOUT_S = 600.0  # a first sign-in may include creating the account
DEFAULT_CLOUD_URL = "https://lightning.ai"
CALLBACK_PATH = "/login-complete"
DONE_PATH = "/login-complete/done"


@dataclass
class LoginOutcome:
    stored: bool
    source: str  # prompt | stdin | import | browser
    user: str | None = None
    teamspace: str | None = None
    teamspaces: list[str] = field(default_factory=list)
    verified: bool = False
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "secrets": [creds_mod.KEYCHAIN_USER, creds_mod.KEYCHAIN_KEY],
            "stored": self.stored,
            "source": self.source,
            "user": self.user,
            "teamspace": self.teamspace,
            "teamspaces": self.teamspaces,
            "verified": self.verified,
            "notes": self.notes,
        }


def parse_pair(text: str) -> tuple[str, str]:
    """(user_id, api_key) from stdin text; InvalidRequest (value-free) otherwise."""
    raw = text.strip()
    if raw.startswith("{"):
        try:
            doc = json.loads(raw)
        except ValueError:
            raise InvalidRequest(
                "stdin is not a JSON credentials document; nothing stored"
            ) from None
        user_id = str(doc.get("user_id") or "").strip() if isinstance(doc, dict) else ""
        api_key = str(doc.get("api_key") or "").strip() if isinstance(doc, dict) else ""
    else:
        parts = raw.split()
        if len(parts) != 2:
            raise InvalidRequest(
                "expected the user id and the API key (two lines or two words); nothing stored"
            )
        user_id, api_key = parts
    if not user_id or not api_key:
        raise InvalidRequest("both the user id and the API key are needed; nothing stored")
    return user_id, api_key


_DONE_PAGE = b"""<!doctype html><html><head><meta charset="utf-8"><title>gpu-router</title>
</head>
<body style="font:15px system-ui;margin:3em">
<p><b>gpu-router received your Lightning sign-in.</b></p>
<p>Back in the terminal it is checked and stored in the Keychain. You can close this tab.</p>
</body></html>"""


class BrowserSignIn:
    """A one-shot 127.0.0.1 callback for lightning.ai's CLI sign-in. `url` is the page to
    open (it carries no secret); `wait()` returns (user_id, api_key) once lightning.ai
    redirects back, or None on timeout. Only a callback that carries this sign-in's random
    `state`, with a localhost Host header and no non-navigation Sec-Fetch-Mode, counts;
    the first such callback wins. The request line (which holds the values) is never
    logged; `rejected` counts refused callbacks."""

    def __init__(self, *, cloud_url: str | None = None, host: str = "127.0.0.1") -> None:
        cloud = cloud_url or os.environ.get("LIGHTNING_CLOUD_URL") or DEFAULT_CLOUD_URL
        cloud = cloud.rstrip("/")
        self._result: tuple[str, str] | None = None
        self._got = threading.Event()
        self._page_served = threading.Event()
        self.state = _stdlib_secrets.token_urlsafe(32)
        self.rejected = 0
        owner = self

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                parts = urlsplit(self.path)
                host_ok = (self.headers.get("Host") or "").lower() in owner._hosts
                if not host_ok:  # DNS rebinding or a misdirected request
                    self._reply(421, b"wrong host")
                    return
                if parts.path == DONE_PATH:
                    self._reply(200, _DONE_PAGE, "text/html; charset=utf-8")
                    owner._page_served.set()
                    return
                if parts.path != CALLBACK_PATH:
                    self._reply(404, b"not found")
                    return
                mode = self.headers.get("Sec-Fetch-Mode")
                if mode is not None and mode != "navigate":  # a page's fetch/img, not a redirect
                    owner.rejected += 1
                    self._reply(403, b"not a sign-in redirect")
                    return
                # lightning.ai appends its values to our redirect: accept `&` and a naive
                # second `?` between them and our state
                qs = parse_qs(parts.query.replace("?", "&"))
                state = (qs.get("state") or [""])[0]
                if not hmac.compare_digest(state.encode(), owner.state.encode()):
                    owner.rejected += 1
                    self._reply(403, b"this sign-in was not started by gpu-router; try again")
                    return
                user_id = (qs.get("userID") or [""])[0].strip()
                api_key = (qs.get("key") or [""])[0].strip()
                if not user_id or not api_key:
                    self._reply(400, b"this sign-in carried no user id / API key; try again")
                    return
                if not owner._got.is_set():
                    owner._result = (user_id, api_key)
                    owner._got.set()
                # a redirect, so the address bar (and anything reading it) ends on a URL
                # without the values
                self.send_response(303)
                self.send_header("Location", DONE_PATH)
                self.send_header("Content-Length", "0")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Referrer-Policy", "no-referrer")
                self.end_headers()

            def _reply(self, code: int, body: bytes, ctype: str = "text/plain") -> None:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Referrer-Policy", "no-referrer")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: Any) -> None:
                return  # the request line holds the API key

        self._server = ThreadingHTTPServer((host, 0), _Handler)
        self._server.daemon_threads = True
        self.port = int(self._server.server_address[1])
        self._hosts = {f"localhost:{self.port}", f"127.0.0.1:{self.port}"}
        redirect = f"http://localhost:{self.port}{CALLBACK_PATH}?{urlencode({'state': self.state})}"
        self.url = f"{cloud}/sign-in?{urlencode({'redirectTo': redirect})}"
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="lightning-sign-in", daemon=True
        )
        self._thread.start()

    def wait(self, timeout: float = BROWSER_TIMEOUT_S) -> tuple[str, str] | None:
        self._got.wait(timeout)
        return self._result

    def close(self) -> None:
        if self._got.is_set():  # let the browser load the "received" page first
            self._page_served.wait(3.0)
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def __enter__(self) -> BrowserSignIn:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def _shape_problem(user_id: str, api_key: str) -> str | None:
    for label, value in (("user id", user_id), ("API key", api_key)):
        if any(c.isspace() for c in value):
            return f"the {label} contains whitespace"
        if len(value) < 6 or len(value) > 512:
            return f"the {label} does not look right ({len(value)} characters)"
    return None


def verify(
    user_id: str,
    api_key: str,
    *,
    bridge_factory: Callable[[dict[str, str]], Any] | None = None,
    timeout: float = VERIFY_TIMEOUT_S,
) -> tuple[dict[str, Any] | None, str | None, bool]:
    """(whoami result, problem, rejected). rejected=True: lightning refused the
    credentials (store nothing). problem without rejected: unreachable (store anyway)."""
    env = {creds_mod.ENV_USER: user_id, creds_mod.ENV_KEY: api_key}
    bridge = (bridge_factory or _default_bridge)(env)
    try:
        res = bridge.call("whoami", {}, timeout=timeout)
    except AuthRequired as exc:  # no SDK interpreter: cannot check, not a rejection
        return None, f"{exc.message} ({exc.hint})" if exc.hint else exc.message, False
    except AdapterError as exc:
        return None, exc.message, False
    if res.ok:
        return res.result, None, False
    if res.kind in ("auth", "verify"):
        return None, res.error or "lightning rejected these credentials", True
    return None, res.error or "could not reach lightning.ai", False


def _default_bridge(env: dict[str, str]) -> Any:
    from gpu_router.paths import Paths
    from gpu_router.providers.lightning.sdk import SdkBridge, SubprocessRunner, resolve_interpreter

    home = Paths.from_env().provider_dir("lightning") / "sdk-home"
    prefix = resolve_interpreter()
    return SdkBridge(
        "lightning",
        runner=SubprocessRunner(),
        interpreter=lambda: prefix,
        env_factory=lambda: env,
        home=lambda: home,
    )


def run_login(
    *,
    source: str,
    stdin_text: Callable[[], str] | None = None,
    prompt_secret: Callable[[str], str] | None = None,
    confirm: Callable[[str], bool] | None = None,
    credential_file: Path | None = None,
    check: bool = True,
    bridge_factory: Callable[[dict[str, str]], Any] | None = None,
    browser: Callable[[], tuple[str, str] | None] | None = None,
    confirm_account: Callable[[str], bool] | None = None,
) -> LoginOutcome:
    """Collect, verify and store. `source` = prompt | stdin | import | browser. On a
    prompt, an existing credentials file is offered for import first (`confirm`).
    `browser` runs the sign-in (BrowserSignIn) and returns the pair, or None on timeout.
    `confirm_account` (a terminal's y/N): a browser sign-in's verified account is shown
    and stored only on yes, so credentials planted by someone else are never stored
    silently."""
    from gpu_router import secrets

    path = credential_file or creds_mod.default_file()
    pair: tuple[str, str] | None = None
    offer = source == "prompt" and confirm is not None and path.is_file()
    if offer and confirm is not None and confirm(f"found {path}; import it into the Keychain?"):
        source = "import"
    if source == "import":
        found = creds_mod.read_file(path)
        if found is None:
            raise InvalidRequest(
                f"no usable lightning credentials in {path}",
                hint="run `gpu login lightning` and paste your user id and API key",
            )
        pair = found
    elif source == "browser":
        if browser is None:
            raise InvalidRequest("no browser sign-in available", hint="use --stdin or --import")
        got = browser()
        if got is None:
            raise InvalidRequest(
                "no sign-in arrived from lightning.ai in time; nothing stored",
                hint="run `gpu login lightning --browser` again, or paste the keys with "
                "`gpu login lightning` (lightning.ai > Settings > Keys)",
            )
        pair = got
    elif source == "stdin":
        if stdin_text is None:
            raise InvalidRequest("no stdin to read the credentials from")
        pair = parse_pair(stdin_text())
    else:
        if prompt_secret is None:
            raise InvalidRequest("no terminal to prompt on", hint="use --stdin or --import")
        pair = (
            prompt_secret("Lightning user id (not shown): ").strip(),
            prompt_secret("Lightning API key (not shown): ").strip(),
        )
        if not pair[0] or not pair[1]:
            raise InvalidRequest("both the user id and the API key are needed; nothing stored")
    user_id, api_key = pair
    secrets.register_for_redaction(user_id)
    secrets.register_for_redaction(api_key)
    problem = _shape_problem(user_id, api_key)
    if problem is not None:
        raise InvalidRequest(f"{problem}; nothing stored")
    outcome = LoginOutcome(stored=False, source=source)
    if check:
        who, why, rejected = verify(user_id, api_key, bridge_factory=bridge_factory)
        if rejected:
            raise InvalidRequest(
                f"lightning rejected these credentials ({why}); nothing stored",
                hint="copy the user id and API key again from lightning.ai > Settings > Keys",
            )
        if who is not None:
            outcome.verified = True
            outcome.user = who.get("user")
            outcome.teamspace = who.get("teamspace")
            outcome.teamspaces = list(who.get("teamspaces") or [])
            if not outcome.teamspace and outcome.teamspaces:
                outcome.notes.append(
                    "several teamspaces: set providers.lightning.teamspace to one of "
                    + ", ".join(outcome.teamspaces)
                )
        elif why:
            outcome.notes.append(f"could not verify with lightning.ai ({why}); stored anyway")
    if source == "browser" and confirm_account is not None:
        if outcome.user:
            where = f", teamspace {outcome.teamspace}" if outcome.teamspace else ""
            question = f"lightning.ai signed you in as {outcome.user}{where}; store it?"
        else:
            question = "store the credentials this lightning.ai sign-in sent (not verified)?"
        if not confirm_account(question):
            raise InvalidRequest(
                "not stored: you did not confirm the lightning account",
                hint="run `gpu login lightning --browser` again and sign in with your account",
            )
    secrets.set_secret(creds_mod.KEYCHAIN_USER, user_id)
    secrets.set_secret(creds_mod.KEYCHAIN_KEY, api_key)
    outcome.stored = True
    return outcome
