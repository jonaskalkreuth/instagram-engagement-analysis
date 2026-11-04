"""Unit tests for the auth helper — all network mocked via MockTransport."""

from __future__ import annotations

import httpx
import pytest

from insta_analytics.collect import auth

VERSION = "v21.0"
APP = {"app_id": "111", "app_secret": "shh", "version": VERSION}


def _client(handler) -> httpx.Client:
    return httpx.Client(base_url=auth.GRAPH_BASE, transport=httpx.MockTransport(handler))


def test_exchange_token_parses_and_sends_params():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, json={"access_token": "LONG", "token_type": "bearer", "expires_in": 5183944})

    token, expires_in = auth.exchange_token("SHORT", **APP, client=_client(handler))

    assert (token, expires_in) == ("LONG", 5183944)
    assert seen["path"] == f"/{VERSION}/oauth/access_token"
    assert seen["params"]["grant_type"] == "fb_exchange_token"
    assert seen["params"]["fb_exchange_token"] == "SHORT"
    assert seen["params"]["client_id"] == "111"


def test_find_ig_user_id_returns_first_linked_account():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"data": [
                {"name": "Some Page", "id": "p1"},
                {"name": "Music Page", "instagram_business_account": {"id": "17890", "username": "musician"}},
            ]},
        )

    assert auth.find_ig_user_id("LONG", version=VERSION, client=_client(handler)) == ("17890", "musician")


def test_find_ig_user_id_raises_when_none_linked():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"name": "Page", "id": "p1"}]})

    with pytest.raises(auth.AuthError, match="No Instagram Business account"):
        auth.find_ig_user_id("LONG", version=VERSION, client=_client(handler))


def test_debug_token_unwraps_data():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": {"is_valid": True, "scopes": ["instagram_basic"]}})

    out = auth.debug_token("LONG", **APP, client=_client(handler))
    assert out == {"is_valid": True, "scopes": ["instagram_basic"]}


def test_verify_returns_profile():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "17890", "username": "musician", "followers_count": 42, "media_count": 7})

    prof = auth.verify("LONG", "17890", version=VERSION, client=_client(handler))
    assert prof["username"] == "musician"
    assert prof["followers_count"] == 42


def test_graph_error_body_raises_authhelper():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"code": 190, "message": "Invalid OAuth access token."}})

    with pytest.raises(auth.AuthError, match="190: Invalid OAuth"):
        auth.exchange_token("SHORT", **APP, client=_client(handler))


def test_mask_token():
    assert auth.mask_token("EAAGabcdef1234567890WXYZ") == "EAAGab…WXYZ"
    assert auth.mask_token("short") == "***"


def test_upsert_env_replaces_and_appends(tmp_path):
    env = tmp_path / ".env"
    env.write_text("INSTA_META_APP_ID=111\nINSTA_IG_ACCESS_TOKEN=old\n")

    auth._upsert_env(env, {"INSTA_IG_ACCESS_TOKEN": "new", "INSTA_IG_USER_ID": "17890"})

    lines = env.read_text().splitlines()
    assert "INSTA_META_APP_ID=111" in lines  # untouched
    assert "INSTA_IG_ACCESS_TOKEN=new" in lines  # replaced, not duplicated
    assert lines.count("INSTA_IG_ACCESS_TOKEN=new") == 1
    assert "INSTA_IG_USER_ID=17890" in lines  # appended
