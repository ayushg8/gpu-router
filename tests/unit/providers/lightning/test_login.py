"""`gpu login lightning`: prompt / stdin / import, verification, never argv, never shown."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from gpu_router import secrets
from gpu_router.cli.app import app
from gpu_router.errors import InvalidRequest
from gpu_router.providers.lightning import login as login_mod
from gpu_router.providers.lightning.login import parse_pair, run_login
from gpu_router.providers.lightning.sdk import CallResult

USER = "user-0123456789"
KEY = "key-abcdef0123456789"


class _Bridge:
    def __init__(self, result: CallResult | Exception) -> None:
        self.result = result
        self.env: dict[str, str] = {}

    def call(self, op: str, params: Any, *, timeout: float) -> CallResult:
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _factory(result: CallResult | Exception) -> Any:
    def make(env: dict[str, str]) -> _Bridge:
        bridge = _Bridge(result)
        bridge.env = env
        return bridge

    return make


OK = CallResult(
    ok=True,
    result={"user": "demo", "teamspace": "demo/default", "teamspaces": ["demo/default"]},
)


def _stored() -> tuple[str | None, str | None]:
    return secrets.get_secret("LIGHTNING_USER_ID"), secrets.get_secret("LIGHTNING_API_KEY")


def test_parse_pair_accepts_json_lines_and_words() -> None:
    assert parse_pair(json.dumps({"user_id": USER, "api_key": KEY})) == (USER, KEY)
    assert parse_pair(f"{USER}\n{KEY}\n") == (USER, KEY)
    assert parse_pair(f"{USER} {KEY}") == (USER, KEY)
    for bad in ("", "only-one", "a b c", '{"user_id": "x"}', "{broken"):
        with pytest.raises(InvalidRequest) as info:
            parse_pair(bad)
        assert "nothing stored" in info.value.message


def test_stdin_login_verifies_then_stores() -> None:
    out = run_login(
        source="stdin", stdin_text=lambda: f"{USER}\n{KEY}\n", bridge_factory=_factory(OK)
    )
    assert out.stored
    assert out.verified
    assert (out.user, out.teamspace) == ("demo", "demo/default")
    assert _stored() == (USER, KEY)


def test_rejected_credentials_store_nothing() -> None:
    rejected = CallResult(ok=False, result={}, kind="auth", error="401 unauthorized")
    with pytest.raises(InvalidRequest, match="rejected"):
        run_login(
            source="stdin", stdin_text=lambda: f"{USER} {KEY}", bridge_factory=_factory(rejected)
        )
    assert _stored() == (None, None)


def test_unreachable_lightning_stores_with_a_note() -> None:
    down = CallResult(ok=False, result={}, kind="unavailable", error="connection reset")
    out = run_login(
        source="stdin", stdin_text=lambda: f"{USER} {KEY}", bridge_factory=_factory(down)
    )
    assert out.stored
    assert not out.verified
    assert any("stored anyway" in n for n in out.notes)
    assert _stored() == (USER, KEY)


def test_several_teamspaces_are_named_in_a_note() -> None:
    many = CallResult(
        ok=True, result={"user": "a", "teamspace": None, "teamspaces": ["a/x", "b/y"]}
    )
    out = run_login(
        source="stdin", stdin_text=lambda: f"{USER} {KEY}", bridge_factory=_factory(many)
    )
    assert any("a/x, b/y" in n for n in out.notes)


def test_import_reads_the_lightning_login_file(tmp_path: Path) -> None:
    path = tmp_path / "credentials.json"
    path.write_text(json.dumps({"user_id": USER, "api_key": KEY, "auth_token": ""}))
    out = run_login(source="import", credential_file=path, check=False)
    assert out.stored
    assert out.source == "import"
    assert _stored() == (USER, KEY)
    with pytest.raises(InvalidRequest, match="no usable"):
        run_login(source="import", credential_file=tmp_path / "missing.json", check=False)


def test_the_prompt_offers_an_existing_file_first(tmp_path: Path) -> None:
    path = tmp_path / "credentials.json"
    path.write_text(json.dumps({"user_id": USER, "api_key": KEY}))
    asked: list[str] = []

    def never(prompt: str) -> str:
        raise AssertionError("should not prompt")

    out = run_login(
        source="prompt",
        prompt_secret=never,
        confirm=lambda q: asked.append(q) or True,
        credential_file=path,
        check=False,
    )
    assert out.source == "import"
    assert asked
    secrets.delete_secret("LIGHTNING_USER_ID")
    answers = iter([USER, KEY])
    out = run_login(
        source="prompt",
        prompt_secret=lambda p: next(answers),
        confirm=lambda q: False,
        credential_file=path,
        check=False,
    )
    assert out.source == "prompt"
    assert _stored() == (USER, KEY)


def test_values_that_do_not_look_right_are_refused() -> None:
    with pytest.raises(InvalidRequest, match="does not look right"):
        run_login(source="stdin", stdin_text=lambda: "abc def", check=False)
    assert _stored() == (None, None)


def test_the_cli_command_never_prints_the_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(login_mod, "_default_bridge", _factory(OK))
    res = CliRunner().invoke(
        app,
        ["login", "lightning", "--stdin", "--json"],
        input=f"{USER}\n{KEY}\n",
        catch_exceptions=False,
    )
    assert res.exit_code == 0, res.output
    doc = json.loads(res.stdout.strip().splitlines()[-1])
    assert doc["stored"]
    assert doc["verified"]
    assert doc["teamspace"] == "demo/default"
    assert KEY not in res.output
    assert USER not in res.output
    assert _stored() == (USER, KEY)
    plain = CliRunner().invoke(
        app,
        ["login", "lightning", "--stdin", "--no-check"],
        input=f"{USER} {KEY}",
        catch_exceptions=False,
    )
    assert plain.exit_code == 0
    assert "stored LIGHTNING_USER_ID and LIGHTNING_API_KEY" in plain.output
    assert KEY not in plain.output


def test_the_cli_help_takes_no_secret_arguments() -> None:
    res = CliRunner().invoke(app, ["login", "lightning", "--help"], catch_exceptions=False)
    assert res.exit_code == 0
    assert "--stdin" in res.output
    assert "--import" in res.output
    assert "--key" not in res.output
    assert "--api-key" not in res.output


def _callback(url: str, **params: str) -> tuple[int, str, str]:
    """Play lightning.ai's redirect back to the sign-in callback: (status, body, url)."""
    import urllib.error
    import urllib.parse
    import urllib.request

    qs = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
    target = qs["redirectTo"][0] + ("?" + urllib.parse.urlencode(params) if params else "")
    target = target.replace("://localhost:", "://127.0.0.1:")
    try:
        with urllib.request.urlopen(target, timeout=5) as resp:
            return resp.status, resp.read().decode(), resp.url
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode(), target


def test_browser_sign_in_takes_the_keys_from_lightnings_redirect(
    capfd: pytest.CaptureFixture[str],
) -> None:
    with login_mod.BrowserSignIn(cloud_url="https://lightning.example") as flow:
        assert flow.url.startswith("https://lightning.example/sign-in?redirectTo=")
        assert f"localhost%3A{flow.port}%2Flogin-complete" in flow.url
        assert _callback(flow.url, token="t")[0] == 400  # no user id / key: keep waiting
        code, body, landed = _callback(flow.url, token="jwt", key=KEY, userID=USER)
        assert code == 200
        assert landed.endswith("/login-complete/done")  # redirected off the values
        assert KEY not in body
        assert USER not in body
        assert "received your Lightning sign-in" in body
        assert flow.wait(1) == (USER, KEY)
        other = _callback(flow.url, key="key-someone-else-000", userID="user-someone-else")
        assert other[0] == 200
        assert flow.wait(1) == (USER, KEY)  # the first complete callback wins
    err = capfd.readouterr()
    assert KEY not in err.err + err.out  # the request line is never logged


def _raw_get(port: int, path: str, headers: dict[str, str]) -> int:
    import http.client

    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.request("GET", path, headers=headers)
        return conn.getresponse().status
    finally:
        conn.close()


def test_browser_sign_in_takes_only_its_own_redirect() -> None:
    """Review fix: the callback stored the first request carrying any userID/key, so a web
    page or local process that found the port could plant its own Lightning credentials."""
    from urllib.parse import urlencode

    with login_mod.BrowserSignIn(cloud_url="https://lightning.example") as flow:
        assert "state%3D" in flow.url  # the redirect carries this sign-in's random state
        bad = urlencode({"userID": "attacker-user-000", "key": "attacker-key-000000"})
        host = {"Host": f"localhost:{flow.port}"}
        # no state, or someone else's
        assert _raw_get(flow.port, f"/login-complete?{bad}", host) == 403
        assert _raw_get(flow.port, f"/login-complete?state=guess&{bad}", host) == 403
        good = f"/login-complete?state={flow.state}&{bad}"
        # a page's fetch()/<img> (Sec-Fetch-Mode: cors / no-cors), not a redirect
        assert _raw_get(flow.port, good, {**host, "Sec-Fetch-Mode": "no-cors"}) == 403
        # DNS rebinding: a foreign Host header
        assert _raw_get(flow.port, good, {"Host": "evil.example:80"}) == 421
        assert flow.rejected == 3
        assert flow.wait(0.05) is None  # nothing was taken
        # the real redirect: right state, a navigation, lightning.ai's naive `?` append
        ok = f"/login-complete?state={flow.state}?userID={USER}&key={KEY}"
        assert _raw_get(flow.port, ok, {**host, "Sec-Fetch-Mode": "navigate"}) == 303
        assert flow.wait(1) == (USER, KEY)


def test_a_browser_login_needs_the_account_confirmed_on_a_terminal() -> None:
    asked: list[str] = []

    def no(question: str) -> bool:
        asked.append(question)
        return False

    with pytest.raises(InvalidRequest, match="did not confirm"):
        run_login(
            source="browser",
            browser=lambda: (USER, KEY),
            bridge_factory=_factory(OK),
            confirm_account=no,
        )
    assert _stored() == (None, None)
    assert asked
    assert "signed you in as" in asked[0]
    out = run_login(
        source="browser",
        browser=lambda: (USER, KEY),
        bridge_factory=_factory(OK),
        confirm_account=lambda q: True,
    )
    assert out.stored
    assert _stored() == (USER, KEY)


def test_browser_sign_in_times_out_to_none() -> None:
    with login_mod.BrowserSignIn(cloud_url="https://lightning.example") as flow:
        assert flow.wait(0.05) is None


def test_browser_login_verifies_then_stores() -> None:
    out = run_login(source="browser", browser=lambda: (USER, KEY), bridge_factory=_factory(OK))
    assert out.stored
    assert out.verified
    assert out.source == "browser"
    assert _stored() == (USER, KEY)


def test_a_browser_login_that_never_arrives_stores_nothing() -> None:
    with pytest.raises(InvalidRequest, match="nothing stored"):
        run_login(source="browser", browser=lambda: None, bridge_factory=_factory(OK))
    assert _stored() == (None, None)


def test_the_cli_browser_login_prints_the_url_but_never_the_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading
    import webbrowser

    monkeypatch.setattr(login_mod, "_default_bridge", _factory(OK))
    opened: list[str] = []

    def fake_open(url: str) -> bool:  # the "browser": lightning.ai redirects back
        opened.append(url)
        threading.Thread(target=_callback, args=(url,), kwargs={"key": KEY, "userID": USER}).start()
        return True

    monkeypatch.setattr(webbrowser, "open", fake_open)
    res = CliRunner().invoke(
        app, ["login", "lightning", "--browser", "--json"], catch_exceptions=False
    )
    assert res.exit_code == 0, res.output
    doc = json.loads(res.stdout.strip().splitlines()[-1])
    assert doc["stored"]
    assert doc["source"] == "browser"
    assert opened
    assert opened[0] in res.output.replace("\n", "")
    assert KEY not in res.output
    assert USER not in res.output
    assert _stored() == (USER, KEY)
