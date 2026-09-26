"""Step 2, logins, and the logic behind `gpu login kaggle|colab` and bare `gpu login`
(phase 8b).

- login.kaggle: done when the Keychain has `kaggle` (the kaggle.json document) or
  `KAGGLE_API_TOKEN`; imports kaggle.json (~/.kaggle, then ~/Downloads) or a pasted token,
  checked with one `kaggle quota` call first.
- login.colab: done when doctor's check passes (ADC with the colaboratory scope, via
  `colab whoami`); otherwise shows the exact gcloud command and offers to run it (it opens
  the browser).
- login.lightning: done when the Keychain has LIGHTNING_USER_ID + LIGHTNING_API_KEY (or the
  file/env mode config.yaml asks for works); imports ~/.lightning/credentials.json, or the
  lightning.ai browser sign-in, or pasted keys.
- login.hf: done when the Keychain has HF_TOKEN; $HF_TOKEN / the hf CLI's token, or paste.
- login.hf_remote (optional): HF_TOKEN_REMOTE; paste (default no).

Secrets are read from files the user named or pasted into a no-echo prompt, never from
argv, and go to the Keychain through gpu_router.secrets only (invariant 12). They are
registered for redaction the moment they are read and never printed. A credential the
provider rejects is not stored; an unreachable provider stores it with a note. Colab's ADC
file is never read or copied (NOTES.md: the adapter must not); the wizard only checks it.
"""

from __future__ import annotations

import json
import shlex
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gpu_router.doctor.checks import ADC_LOGIN
from gpu_router.doctor.probe import home_label
from gpu_router.errors import InvalidRequest
from gpu_router.setup.base import Ctx, Outcome
from gpu_router.setup.providers import catalog_or_none, config_or_none, enabled_kinds
from gpu_router.setup.ui import Mark

if TYPE_CHECKING:
    from gpu_router.doctor.model import CheckResult
    from gpu_router.providers.catalog import ProviderEntry
    from gpu_router.setup.context import SetupEnv

__all__ = [
    "ADC_LOGIN",
    "ITEMS",
    "KaggleCreds",
    "LoginRow",
    "colab_check",
    "gcloud_path",
    "kaggle_candidates",
    "login_rows",
    "parse_kaggle",
    "run",
    "store_kaggle",
    "verify_kaggle",
]

ITEMS = ("login.kaggle", "login.colab", "login.lightning", "login.hf", "login.hf_remote")
KAGGLE_JSON = "kaggle"
KAGGLE_TOKEN = "KAGGLE_API_TOKEN"  # noqa: S105 - a secret name, not a value
KAGGLE_ENV = ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN", "KAGGLE_CONFIG_DIR")
KAGGLE_TOKEN_URL = "kaggle.com > Settings > API > Create New Token"  # noqa: S105 - a hint
KAGGLE_VERIFY_TIMEOUT_S = 60.0
_KAGGLE_AUTH = ("401", "403", "unauthorized", "invalid credentials", "authentication required")


GCLOUD_INSTALL = "brew install --cask google-cloud-sdk"


# =========================================================================== kaggle


@dataclass(frozen=True)
class KaggleCreds:
    kind: str  # "json" (username + key) | "token" (access token)
    username: str | None
    value: str  # the key or the token (never printed; registered for redaction)

    def __repr__(self) -> str:
        return f"KaggleCreds(kind={self.kind!r}, username={self.username!r})"

    def env(self) -> dict[str, str]:
        if self.kind == "json":
            return {"KAGGLE_USERNAME": self.username or "", "KAGGLE_KEY": self.value}
        return {"KAGGLE_API_TOKEN": self.value}


def parse_kaggle(text: str) -> KaggleCreds:
    """kaggle.json ({"username", "key"}) or a bare access token; InvalidRequest (value-free)
    otherwise. The secret part is registered for redaction first."""
    from gpu_router import secrets

    raw = text.strip()
    if not raw:
        raise InvalidRequest("nothing to store: the kaggle credentials are empty")
    if raw.startswith("{"):
        try:
            doc = json.loads(raw)
        except ValueError:
            raise InvalidRequest("that is not a kaggle.json document; nothing stored") from None
        if not isinstance(doc, dict):
            raise InvalidRequest("that is not a kaggle.json document; nothing stored")
        username = str(doc.get("username") or "").strip()
        key = str(doc.get("key") or "").strip()
        if key:
            secrets.register_for_redaction(key)
        if not username or not key:
            raise InvalidRequest("kaggle.json needs both username and key; nothing stored")
        return KaggleCreds("json", username, key)
    secrets.register_for_redaction(raw)
    if any(c.isspace() for c in raw) or len(raw) < 20:
        raise InvalidRequest(
            "that does not look like a kaggle access token or kaggle.json; nothing stored",
            hint=f"create one at {KAGGLE_TOKEN_URL}",
        )
    return KaggleCreds("token", None, raw)


def kaggle_candidates(env: SetupEnv) -> list[Path]:
    """Where a kaggle.json (or access token) may be: the CLI's dir, then ~/Downloads (where
    kaggle.com's "Create New Token" saves it)."""
    out: list[Path] = []
    for p in (
        env.kaggle_dir / "kaggle.json",
        env.kaggle_dir / "access_token",
        env.user_home / "Downloads" / "kaggle.json",
    ):
        if p.is_file() and p not in out:
            out.append(p)
    return out


def kaggle_stored() -> list[str]:
    from gpu_router import secrets

    try:
        names = set(secrets.secret_names())
    except OSError:
        return []
    return [n for n in (KAGGLE_JSON, KAGGLE_TOKEN) if n in names]


def verify_kaggle(env: SetupEnv, creds: KaggleCreds) -> tuple[str | None, bool]:
    """(problem, rejected): one `kaggle quota` call with these credentials only (an empty
    KAGGLE_CONFIG_DIR, so ~/.kaggle cannot answer for them). rejected=True: kaggle refused
    them (store nothing); a problem without rejected = could not check (store anyway)."""
    from gpu_router import secrets

    exe = env.find("kaggle")
    if exe is None:
        return "the kaggle CLI is not installed, so they were not checked", False
    tmp = Path(tempfile.mkdtemp(prefix="gpu-setup-kaggle-"))
    try:
        child = {k: v for k, v in env.environ.items() if k not in KAGGLE_ENV}
        child.update(creds.env())
        child.update({"KAGGLE_CONFIG_DIR": str(tmp), "PYTHONIOENCODING": "utf-8"})
        res = env.run([exe, "-W", "quota", "--format", "json"], KAGGLE_VERIFY_TIMEOUT_S, env=child)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if res.ok:
        return None, False
    text = secrets.redact(res.text)
    if res.returncode is None:
        return f"kaggle did not answer ({res.error or 'timed out'})", False
    if any(m in text.lower() for m in _KAGGLE_AUTH):
        return "kaggle rejected these credentials", True
    last = next((ln for ln in reversed(text.splitlines()) if ln.strip()), "")
    return f"could not check with kaggle ({last[:160] or f'exit {res.returncode}'})", False


def store_kaggle(creds: KaggleCreds) -> str:
    """Keychain: `kaggle` = the kaggle.json document, or `KAGGLE_API_TOKEN`. Returns the name."""
    from gpu_router import secrets

    if creds.kind == "json":
        doc = json.dumps({"username": creds.username, "key": creds.value})
        secrets.set_secret(KAGGLE_JSON, doc)
        return KAGGLE_JSON
    secrets.set_secret(KAGGLE_TOKEN, creds.value)
    return KAGGLE_TOKEN


def import_kaggle(env: SetupEnv, creds: KaggleCreds, *, check: bool) -> tuple[str, list[str]]:
    """Verify (unless check=False) and store. Returns (secret name, notes). Raises
    InvalidRequest when kaggle rejects them."""
    notes: list[str] = []
    if check:
        problem, rejected = verify_kaggle(env, creds)
        if rejected:
            raise InvalidRequest(
                "kaggle rejected these credentials; nothing stored",
                hint=f"create a new token at {KAGGLE_TOKEN_URL}",
            )
        if problem:
            notes.append(f"{problem}; stored anyway")
    name = store_kaggle(creds)
    return name, notes


def _read_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InvalidRequest(f"cannot read {path}: {exc.strerror or exc}") from None


def _login_kaggle(ctx: Ctx) -> None:
    item = "login.kaggle"
    env = ctx.env
    stored = kaggle_stored()
    if stored:
        ctx.done(
            item, Outcome.ALREADY, f"kaggle: credentials in the Keychain ({', '.join(stored)})"
        )
        return
    found = kaggle_candidates(env)
    in_cli_dir = [p for p in found if p.parent == env.kaggle_dir]
    if found:
        path = found[0]
        label = home_label(path, env.user_home)
        ctx.ui.item(Mark.SKIP, f"kaggle: {label} found; the Keychain has no copy yet")
        if ctx.opts.dry_run:
            ctx.dry(item, f"import {label} into the Keychain (the file stays)")
            return
        answer = ctx.confirm(
            item, f"import {label} into the Keychain? (the file stays where it is)", default=True
        )
        if answer is None:
            ctx.not_asked(item, f"kaggle: {label} not imported", "gpu login kaggle")
            return
        if not answer:
            works = " (the kaggle CLI keeps using it)" if in_cli_dir else ""
            ctx.declined(item, f"kaggle: {label} not imported{works}", "gpu login kaggle")
            return
        try:
            creds = parse_kaggle(_read_file(path))
            name, notes = import_kaggle(env, creds, check=ctx.opts.check)
        except InvalidRequest as exc:
            ctx.done(item, Outcome.FAILED, f"kaggle: {exc.message}", "gpu login kaggle")
            return
        for note in notes:
            ctx.ui.say(note, "yellow")
        who = f" for {creds.username}" if creds.username else ""
        ctx.done(item, Outcome.DONE, f"kaggle: stored {name}{who} in the Keychain")
        return
    ctx.ui.item(Mark.SKIP, f"kaggle: no credentials yet (create a token at {KAGGLE_TOKEN_URL})")
    if ctx.opts.dry_run:
        ctx.dry(item, "ask for kaggle.json or an access token")
        return
    if not ctx.can_type():
        ctx.needs_keyboard(item, "kaggle: no credentials", "gpu login kaggle")
        return
    answer = ctx.confirm(item, "paste kaggle.json or an access token now?", default=True)
    if not answer:
        ctx.declined(item, "kaggle: no credentials", "gpu login kaggle")
        return
    text = ctx.ui.secret("kaggle.json contents or access token (not shown): ")
    try:
        creds = parse_kaggle(text)
        name, notes = import_kaggle(env, creds, check=ctx.opts.check)
    except InvalidRequest as exc:
        ctx.done(item, Outcome.FAILED, f"kaggle: {exc.message}", "gpu login kaggle")
        return
    for note in notes:
        ctx.ui.say(note, "yellow")
    ctx.done(item, Outcome.DONE, f"kaggle: stored {name} in the Keychain")


# =========================================================================== colab


def _entry(env: SetupEnv, kind: str) -> ProviderEntry | None:
    catalog = catalog_or_none(env)
    if catalog is None:
        return None
    return next((e for e in catalog.ordered() if e.kind == kind and e.lane == "gpu"), None)


def colab_check(env: SetupEnv) -> CheckResult | None:
    """doctor's colab login row (ADC present, `colab whoami` has the colaboratory scope)."""
    from gpu_router.doctor.checks import check_colab_login

    entry = _entry(env, "colab")
    if entry is None:
        return None
    return check_colab_login(env.probe(deadline_s=40.0), entry)


def gcloud_path(env: SetupEnv) -> str | None:
    """gcloud on PATH, else where Google's installer puts it (~/google-cloud-sdk)."""
    return env.find("gcloud", env.user_home / "google-cloud-sdk" / "bin" / "gcloud")


def adc_argv(gcloud: str) -> list[str]:
    return [gcloud, *shlex.split(ADC_LOGIN)[1:]]


def run_adc_login(env: SetupEnv, gcloud: str) -> int:
    """gcloud's browser sign-in on the user's terminal (it writes the ADC file itself)."""
    return env.run_attached(adc_argv(gcloud), env.environ)


def _login_colab(ctx: Ctx) -> None:
    from gpu_router.doctor.model import Status

    item = "login.colab"
    env = ctx.env
    row = colab_check(env)
    if row is None:
        ctx.done(item, Outcome.SKIPPED, "colab: not in providers.yaml")
        return
    if row.status is Status.OK:
        ctx.done(item, Outcome.ALREADY, f"colab: {row.summary}")
        return
    if row.status is Status.SKIP:
        ctx.done(item, Outcome.SKIPPED, f"colab: {row.summary}")
        return
    if row.fix != ADC_LOGIN:
        ctx.done(item, Outcome.MANUAL, f"colab: {row.summary}", row.fix)
        return
    ctx.ui.item(Mark.SKIP, f"colab: {row.summary}")
    ctx.ui.say("colab signs in with Google application-default credentials; mint them with:", "dim")
    ctx.ui.command(ADC_LOGIN)
    gcloud = gcloud_path(env)
    if gcloud is None:
        ctx.done(
            item,
            Outcome.MANUAL,
            "colab: gcloud is not installed; install it, then run the command above",
            f"{GCLOUD_INSTALL} && {ADC_LOGIN}",
        )
        return
    if ctx.opts.dry_run:
        ctx.dry(item, "offer to run the gcloud sign-in (it opens your browser)")
        return
    answer = ctx.confirm(
        item,
        "run it now? it opens your browser to sign in to Google (use the account colab uses)",
        default=True,
        needs_human=True,
    )
    if answer is None:
        ctx.not_asked(item, "colab: no application-default credentials", ADC_LOGIN)
        return
    if not answer:
        ctx.declined(item, "colab: not signed in", ADC_LOGIN)
        return
    code = run_adc_login(env, gcloud)
    again = colab_check(env)
    if again is not None and again.status is Status.OK:
        ctx.done(item, Outcome.DONE, f"colab: {again.summary}")
        return
    why = again.summary if again is not None else f"gcloud exited {code}"
    ctx.done(item, Outcome.FAILED, f"colab: still not ready: {why}", ADC_LOGIN)


# =========================================================================== lightning


def lightning_check(env: SetupEnv) -> CheckResult | None:
    from gpu_router.doctor.checks import check_lightning_login

    entry = _entry(env, "lightning")
    if entry is None:
        return None
    return check_lightning_login(env.probe(), entry)


def _browser_sign_in(ctx: Ctx) -> tuple[str, str] | None:
    if ctx.env.lightning_browser is not None:
        return ctx.env.lightning_browser()
    import webbrowser

    from gpu_router.providers.lightning.login import BrowserSignIn

    with BrowserSignIn() as flow:
        ctx.ui.say(f"sign in to lightning.ai to connect gpu-router: {flow.url}")
        if not webbrowser.open(flow.url):
            ctx.ui.say("could not open a browser; open the URL above yourself", "yellow")
        ctx.ui.say("waiting for lightning.ai (up to 10 min; ctrl+c stops)", "dim")
        return flow.wait()


def _lightning_mode(env: SetupEnv) -> str:
    """providers.lightning.login_source, read like doctor and the adapter do."""
    config = config_or_none(env)
    settings = config.providers.get("lightning") if config is not None else None
    extra = (settings.model_extra or {}) if settings is not None else {}
    return str(extra.get("login_source") or extra.get("credentials") or "auto")


def _lightning_login(
    ctx: Ctx, source: str, stdin_text: Callable[[], str] | None = None
) -> tuple[bool, str]:
    """(stored, one-line summary) through providers/lightning/login.run_login."""
    from gpu_router.providers.lightning.login import run_login

    if ctx.opts.check:
        ctx.ui.say(
            "checking with lightning.ai (the first check can fetch the lightning SDK)", "dim"
        )
    try:
        outcome = run_login(
            source=source,
            stdin_text=stdin_text,
            prompt_secret=ctx.ui.secret,
            credential_file=ctx.env.lightning_file,
            check=ctx.opts.check,
            bridge_factory=ctx.env.lightning_bridge,
            browser=lambda: _browser_sign_in(ctx),
            confirm_account=(lambda q: ctx.ui.ask(q, False)) if ctx.can_type() else None,
        )
    except InvalidRequest as exc:
        return False, exc.message
    for note in outcome.notes:
        ctx.ui.say(note, "yellow")
    who = f" for {outcome.user}" if outcome.user else ""
    where = f", teamspace {outcome.teamspace}" if outcome.teamspace else ""
    return True, f"stored LIGHTNING_USER_ID and LIGHTNING_API_KEY{who} in the Keychain{where}"


def _login_lightning(ctx: Ctx) -> None:
    from gpu_router.doctor.model import Status

    item = "login.lightning"
    env = ctx.env
    row = lightning_check(env)
    if row is None:
        ctx.done(item, Outcome.SKIPPED, "lightning: not in providers.yaml")
        return
    uses = row.detail.get("uses")
    mode = _lightning_mode(env)
    file_label = home_label(env.lightning_file, env.user_home)
    if row.status is Status.OK and (uses == "keychain" or mode != "auto"):
        ctx.done(item, Outcome.ALREADY, f"lightning: {row.summary}")
        return
    if uses == "env" and mode == "auto":
        # review fix: variables in this shell are not the launchd daemon's; offer the
        # Keychain (checked with lightning.ai like any login) instead of "already done"
        ctx.ui.item(
            Mark.SKIP,
            "lightning: LIGHTNING_USER_ID / LIGHTNING_API_KEY are only in this shell's "
            "environment; the daemon started at login cannot see them",
        )
        if ctx.opts.dry_run:
            ctx.dry(item, "store the environment's Lightning keys in the Keychain")
            return
        answer = ctx.confirm(
            item, "store them in the Keychain (checked with lightning.ai first)?", default=True
        )
        if answer is None:
            ctx.not_asked(item, "lightning: keys only in the environment", "gpu login lightning")
            return
        if not answer:
            ctx.declined(item, "lightning: keys only in the environment", "gpu login lightning")
            return
        user_id = env.environ.get("LIGHTNING_USER_ID", "")
        pair = f"{user_id}\n{env.environ.get('LIGHTNING_API_KEY', '')}"
        ok, summary = _lightning_login(ctx, "stdin", stdin_text=lambda: pair)
        ctx.done(
            item,
            Outcome.DONE if ok else Outcome.FAILED,
            f"lightning: {summary}",
            None if ok else "gpu login lightning",
        )
        return
    if mode not in ("auto", "keychain"):  # env / file / a typo: the user manages it
        ctx.done(item, Outcome.MANUAL, f"lightning: {row.summary}", row.fix)
        return
    if env.lightning_file.is_file() and mode in ("auto", "keychain"):
        ctx.ui.item(Mark.SKIP, f"lightning: {file_label} found; the Keychain has no copy yet")
        if ctx.opts.dry_run:
            ctx.dry(item, f"import {file_label} into the Keychain")
            return
        answer = ctx.confirm(item, f"import {file_label} into the Keychain?", default=True)
        if answer is None:
            ctx.not_asked(
                item, f"lightning: {file_label} not imported", "gpu login lightning --import"
            )
            return
        if not answer:
            works = " (gpu-router keeps using the file)" if row.status is Status.OK else ""
            ctx.declined(item, f"lightning: not imported{works}", "gpu login lightning --import")
            return
        ok, summary = _lightning_login(ctx, "import")
        ctx.done(
            item,
            Outcome.DONE if ok else Outcome.FAILED,
            f"lightning: {summary}",
            None if ok else "gpu login lightning",
        )
        return
    if row.status is Status.WARN and row.fix and row.fix.startswith("gpu secrets set"):
        ctx.done(item, Outcome.MANUAL, f"lightning: {row.summary}", row.fix)
        return
    ctx.ui.item(
        Mark.SKIP,
        "lightning: not logged in (sign up at lightning.ai with Google, verify your phone; no "
        "card)",
    )
    if ctx.opts.dry_run:
        ctx.dry(item, "offer the lightning.ai browser sign-in")
        return
    if not ctx.can_type():
        ctx.needs_keyboard(item, "lightning: not logged in", "gpu login lightning --browser")
        return
    if ctx.state.resumed and ctx.state.answer(item) == "no" and not ctx.explicit(item):
        ctx.declined(item, "lightning: not logged in", "gpu login lightning --browser")
        return
    choice = ctx.ui.choose(
        "log in to lightning.ai how?",
        [
            ("b", "sign in in the browser (lightning.ai sends the keys back; nothing to paste)"),
            ("p", "paste the user id and API key (lightning.ai > Settings > Keys)"),
            ("s", "skip for now"),
        ],
        default="b",
    )
    ctx.state.record_answer(item, choice != "s")
    if choice == "s":
        ctx.declined(item, "lightning: not logged in", "gpu login lightning --browser")
        return
    ok, summary = _lightning_login(ctx, "browser" if choice == "b" else "prompt")
    ctx.done(
        item,
        Outcome.DONE if ok else Outcome.FAILED,
        f"lightning: {summary}",
        None if ok else "gpu login lightning --browser",
    )


# =========================================================================== hugging face


def _hf_backend_off(env: SetupEnv) -> str | None:
    config = config_or_none(env)
    backend = config.checkpoint.backend if config is not None else None
    return backend if backend in ("off", "local") else None


def store_hf(
    env: SetupEnv, token: str, *, remote: bool, check: bool
) -> tuple[str | None, list[str]]:
    """Shape check, whoami (unless check=False), store. Returns (user, notes); raises
    InvalidRequest (nothing stored) for a malformed or rejected token."""
    from gpu_router import secrets
    from gpu_router.checkpoint import tokens
    from gpu_router.cli.login import _cache_namespace

    value = token.strip()
    problem = tokens.token_shape_problem(value)
    if problem is not None:
        raise InvalidRequest(f"{problem}; nothing stored")
    secrets.register_for_redaction(value)
    user: str | None = None
    notes: list[str] = []
    if check:
        user, rejected = env.hf_whoami(value)
        if rejected:
            raise InvalidRequest(
                f"{rejected}; nothing stored",
                hint="create a token at huggingface.co/settings/tokens and try again",
            )
        if user is None:
            notes.append("could not reach hugging face to verify it; stored anyway")
        elif env.hf_role(value) == "read":
            # review fix: checkpoints need write access; a read-only token was stored silently
            notes.append(
                "this token is read-only: saving checkpoints needs write access (create a "
                "write or fine-grained token and run `gpu login hf"
                + (" --remote`)" if remote else "`)")
            )
    secrets.set_secret(tokens.REMOTE_SECRET if remote else tokens.ADMIN_SECRET, value)
    if user and not remote:
        _cache_namespace(value, user, home=env.paths.home)
    return user, notes


def _hf_names() -> set[str]:
    from gpu_router import secrets

    try:
        return set(secrets.secret_names())
    except OSError:
        return set()


def _hf_token_location(env: SetupEnv) -> str | None:
    """Where `checkpoint.tokens.external_token` would find a token ($HF_TOKEN or the hf
    CLI's token file), judged without reading the file (a dry run reads no secret)."""
    e = env.hf_environ()
    if (e.get("HF_TOKEN") or "").strip():
        return "$HF_TOKEN"
    raw = e.get("HF_TOKEN_PATH")
    path = Path(raw).expanduser() if raw else Path(e["HF_HOME"]).expanduser() / "token"
    try:
        return str(path) if path.is_file() and path.stat().st_size > 0 else None
    except OSError:
        return None


def _login_hf(ctx: Ctx) -> None:
    from gpu_router.checkpoint import tokens

    item = "login.hf"
    env = ctx.env
    off = _hf_backend_off(env)
    if off:
        ctx.done(
            item, Outcome.SKIPPED, f"hugging face: checkpoint.backend is {off}; no token needed"
        )
        return
    if tokens.ADMIN_SECRET in _hf_names():
        ctx.done(item, Outcome.ALREADY, "hugging face: HF_TOKEN is in the Keychain")
        return
    why = (
        "checkpoints move between providers through a private HF bucket; without a token a "
        "job resumes only on this Mac"
    )
    where = _hf_token_location(env)
    if where is not None:
        label = home_label(where, env.user_home)
        ctx.ui.item(Mark.SKIP, f"hugging face: a token is in {label}; the Keychain has none")
        if ctx.opts.dry_run:
            ctx.dry(item, f"store the token from {label} as HF_TOKEN")
            return
        answer = ctx.confirm(item, f"use the token from {label} for checkpoints?", default=True)
        if answer is None:
            ctx.not_asked(item, "hugging face: no HF_TOKEN", "gpu login hf --import")
            return
        if not answer:
            ctx.declined(item, "hugging face: no HF_TOKEN", "gpu login hf")
            return
        found = tokens.external_token(env.hf_environ())  # read only now, after the yes
        if found is None:
            ctx.done(
                item, Outcome.FAILED, f"hugging face: no token in {label} any more", "gpu login hf"
            )
            return
        _finish_hf(ctx, item, found[1], remote=False)
        return
    ctx.ui.item(Mark.SKIP, f"hugging face: no token ({why})")
    if ctx.opts.dry_run:
        ctx.dry(item, "ask for a Hugging Face token")
        return
    if not ctx.can_type():
        ctx.needs_keyboard(item, "hugging face: no HF_TOKEN", "gpu login hf")
        return
    answer = ctx.confirm(
        item,
        "paste a Hugging Face token now? (huggingface.co/settings/tokens, write access)",
        default=True,
    )
    if not answer:
        ctx.declined(item, "hugging face: no HF_TOKEN", "gpu login hf")
        return
    _finish_hf(ctx, item, ctx.ui.secret("Hugging Face token (not shown): "), remote=False)


def _finish_hf(ctx: Ctx, item: str, value: str, *, remote: bool) -> None:
    name = "HF_TOKEN_REMOTE" if remote else "HF_TOKEN"
    fix = "gpu login hf --remote" if remote else "gpu login hf"
    try:
        user, notes = store_hf(ctx.env, value, remote=remote, check=ctx.opts.check)
    except InvalidRequest as exc:
        ctx.done(item, Outcome.FAILED, f"hugging face: {exc.message}", fix)
        return
    for note in notes:
        ctx.ui.say(note, "yellow")
    who = f" for {user}" if user else ""
    ctx.done(item, Outcome.DONE, f"hugging face: stored {name}{who} in the Keychain")


def _login_hf_remote(ctx: Ctx) -> None:
    from gpu_router.checkpoint import tokens

    item = "login.hf_remote"
    if _hf_backend_off(ctx.env):
        return
    names = _hf_names()
    if tokens.REMOTE_SECRET in names:
        ctx.done(item, Outcome.ALREADY, "hugging face: HF_TOKEN_REMOTE is in the Keychain")
        return
    if tokens.ADMIN_SECRET not in names and not ctx.explicit(item):
        return  # no bucket yet: the remote token has nothing to write to
    what = (
        "remote runs (kaggle, colab, lightning) need a second, fine-grained token that can "
        "only write your buckets to save and resume checkpoints"
    )
    ctx.ui.item(Mark.SKIP, f"hugging face: no HF_TOKEN_REMOTE ({what})")
    if ctx.opts.dry_run:
        ctx.dry(item, "ask for the remote token")
        return
    if not ctx.can_type():
        ctx.needs_keyboard(item, "hugging face: no HF_TOKEN_REMOTE", "gpu login hf --remote")
        return
    answer = ctx.confirm(item, "paste the fine-grained remote token now?", default=False)
    if not answer:
        ctx.declined(item, "hugging face: no HF_TOKEN_REMOTE", "gpu login hf --remote")
        return
    _finish_hf(ctx, item, ctx.ui.secret("remote Hugging Face token (not shown): "), remote=True)


# =========================================================================== step


def run(ctx: Ctx) -> None:
    kinds = enabled_kinds(ctx.env)
    steps = (
        ("login.kaggle", "kaggle", _login_kaggle),
        ("login.colab", "colab", _login_colab),
        ("login.lightning", "lightning", _login_lightning),
        ("login.hf", None, _login_hf),
        ("login.hf_remote", None, _login_hf_remote),
    )
    for item, kind, fn in steps:
        if not ctx.selected(item):
            continue
        if kind is not None and kind not in kinds:
            if ctx.explicit(item):
                ctx.done(item, Outcome.SKIPPED, f"{kind} is not enabled in config.yaml")
            continue
        fn(ctx)


# =========================================================================== gpu login


@dataclass
class LoginRow:
    name: str
    status: str  # ok | missing | partial | unknown
    summary: str
    command: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "status": self.status,
            "summary": self.summary,
            "command": self.command,
        }


def login_rows(env: SetupEnv) -> list[LoginRow]:
    """Bare `gpu login`: every login gpu-router uses, from local facts only (Keychain names,
    file stats, the ADC file's presence; no network, no subprocess)."""
    from gpu_router import secrets
    from gpu_router.doctor.checks import _adc_path
    from gpu_router.doctor.probe import stat_mode

    try:
        names = set(secrets.secret_names())
    except OSError:
        names = set()
    kinds = enabled_kinds(env)
    rows: list[LoginRow] = []
    if "kaggle" in kinds:
        stored = [n for n in (KAGGLE_JSON, KAGGLE_TOKEN) if n in names]
        files = [p for p in kaggle_candidates(env) if p.parent == env.kaggle_dir]
        if stored:
            rows.append(
                LoginRow("kaggle", "ok", f"Keychain ({', '.join(stored)})", "gpu login kaggle")
            )
        elif files:
            rows.append(
                LoginRow(
                    "kaggle",
                    "ok",
                    f"{home_label(files[0], env.user_home)} (import it: gpu login kaggle)",
                    "gpu login kaggle",
                )
            )
        else:
            rows.append(LoginRow("kaggle", "missing", "no credentials", "gpu login kaggle"))
    if "colab" in kinds:
        adc = _adc_path(env.probe())
        if stat_mode(adc) is not None:
            rows.append(
                LoginRow(
                    "colab",
                    "ok",
                    f"ADC at {home_label(adc, env.user_home)} (scope: gpu login colab)",
                    "gpu login colab",
                )
            )
        else:
            rows.append(
                LoginRow(
                    "colab", "missing", "no application-default credentials", "gpu login colab"
                )
            )
    if "lightning" in kinds:
        keys = [n for n in ("LIGHTNING_USER_ID", "LIGHTNING_API_KEY") if n in names]
        if len(keys) == 2:
            rows.append(LoginRow("lightning", "ok", "Keychain", "gpu login lightning"))
        elif env.lightning_file.is_file():
            rows.append(
                LoginRow(
                    "lightning",
                    "ok",
                    f"{home_label(env.lightning_file, env.user_home)} (import: gpu login lightning "
                    "--import)",
                    "gpu login lightning",
                )
            )
        elif keys:
            rows.append(LoginRow("lightning", "partial", f"only {keys[0]}", "gpu login lightning"))
        else:
            rows.append(
                LoginRow("lightning", "missing", "not logged in", "gpu login lightning --browser")
            )
    hf = [n for n in ("HF_TOKEN", "HF_TOKEN_REMOTE") if n in names]
    rows.append(
        LoginRow(
            "hf",
            "ok" if "HF_TOKEN" in hf else "missing",
            f"Keychain ({', '.join(hf)})" if hf else "no token: checkpoints stay on this Mac",
            "gpu login hf",
        )
    )
    rows += _inference_rows(env, names)
    return rows


def _inference_rows(env: SetupEnv, names: set[str]) -> list[LoginRow]:
    catalog = catalog_or_none(env)
    if catalog is None:
        return []
    try:
        from gpu_router.inference.catalog import load_inference_catalog

        infer = load_inference_catalog(catalog)
    except Exception:  # a broken inference section must not hide the GPU logins
        return []
    rows: list[LoginRow] = []
    for entry in infer.ordered():
        if not entry.secrets or entry.login_name == "hf":
            continue
        wanted = list(entry.secrets.values())
        have = [n for n in wanted if n in names]
        status = "ok" if len(have) == len(wanted) else ("partial" if have else "missing")
        summary = {
            "ok": "Keychain (inference lane)",
            "partial": f"missing {', '.join(n for n in wanted if n not in have)}",
            "missing": "optional: inference lane (LLM calls, evals)",
        }[status]
        rows.append(LoginRow(entry.login_name, status, summary, f"gpu login {entry.login_name}"))
    return rows
