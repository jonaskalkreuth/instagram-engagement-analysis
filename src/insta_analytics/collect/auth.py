"""Exchange a short-lived Meta token for ``INSTA_IG_ACCESS_TOKEN`` and ``INSTA_IG_USER_ID``.

Run after the manual steps in ``docs/setup-meta-auth.md``. Never commit a real token.

    uv run python -m insta_analytics.collect.auth --short-token <TOKEN> --write
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import httpx
from pydantic_settings import BaseSettings, SettingsConfigDict

GRAPH_BASE = "https://graph.facebook.com"
_ENV_PATH = Path(__file__).resolve().parents[3] / ".env"


class AuthError(RuntimeError):
    """A Graph API call failed or returned no usable data."""


class _AuthEnv(BaseSettings):
    """App credentials only; ``config.Settings`` requires the token this script creates."""

    model_config = SettingsConfigDict(
        env_prefix="INSTA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    meta_app_id: str
    meta_app_secret: str
    graph_api_version: str = "v21.0"


def _request(client: httpx.Client, path: str, params: dict[str, str]) -> dict:
    """GET ``path`` and return parsed JSON, raising :class:`AuthError` on failure."""
    resp = client.get(path, params=params)
    try:
        data = resp.json()
    except ValueError:
        resp.raise_for_status()
        raise AuthError(f"Non-JSON response from {path}: {resp.text[:200]}")
    if isinstance(data, dict) and "error" in data:
        err = data["error"]
        raise AuthError(f"Graph API error {err.get('code')}: {err.get('message')}")
    resp.raise_for_status()
    return data


def exchange_token(
    short_token: str,
    *,
    app_id: str,
    app_secret: str,
    version: str,
    client: httpx.Client | None = None,
) -> tuple[str, int | None]:
    """Exchange a short-lived token. Returns ``(long_lived_token, expires_in_seconds)``."""
    close = client is None
    client = client or httpx.Client(base_url=GRAPH_BASE, timeout=30)
    try:
        data = _request(
            client,
            f"/{version}/oauth/access_token",
            {
                "grant_type": "fb_exchange_token",
                "client_id": app_id,
                "client_secret": app_secret,
                "fb_exchange_token": short_token,
            },
        )
    finally:
        if close:
            client.close()
    return data["access_token"], data.get("expires_in")


def debug_token(
    token: str,
    *,
    app_id: str,
    app_secret: str,
    version: str,
    client: httpx.Client | None = None,
) -> dict:
    """Return the ``/debug_token`` payload (validity, scopes, ``expires_at``)."""
    close = client is None
    client = client or httpx.Client(base_url=GRAPH_BASE, timeout=30)
    try:
        data = _request(
            client,
            f"/{version}/debug_token",
            {"input_token": token, "access_token": f"{app_id}|{app_secret}"},
        )
    finally:
        if close:
            client.close()
    return data.get("data", {})


def find_ig_user_id(
    token: str,
    *,
    version: str,
    client: httpx.Client | None = None,
) -> tuple[str, str | None]:
    """Return ``(ig_user_id, ig_username)`` of the IG account linked to the user's Pages."""
    close = client is None
    client = client or httpx.Client(base_url=GRAPH_BASE, timeout=30)
    try:
        data = _request(
            client,
            f"/{version}/me/accounts",
            {
                "fields": "name,instagram_business_account{id,username}",
                "access_token": token,
            },
        )
    finally:
        if close:
            client.close()
    for page in data.get("data", []):
        iba = page.get("instagram_business_account")
        if iba:
            return iba["id"], iba.get("username")
    raise AuthError(
        "No Instagram Business account linked to any Page for this token. "
        "Ensure your IG account is Business/Creator and linked to a Facebook Page, "
        "and that the token has the pages_show_list + instagram_basic scopes."
    )


def verify(
    token: str,
    ig_user_id: str,
    *,
    version: str,
    client: httpx.Client | None = None,
) -> dict:
    """Confirm the token can read the IG node; return its basic profile fields."""
    close = client is None
    client = client or httpx.Client(base_url=GRAPH_BASE, timeout=30)
    try:
        return _request(
            client,
            f"/{version}/{ig_user_id}",
            {"fields": "id,username,followers_count,media_count", "access_token": token},
        )
    finally:
        if close:
            client.close()


def mask_token(token: str) -> str:
    """Return a log-safe rendering that reveals only the first 6 / last 4 chars."""
    if len(token) <= 12:
        return "***"
    return f"{token[:6]}…{token[-4:]}"


def _upsert_env(path: Path, updates: dict[str, str]) -> None:
    """Set each ``KEY=value`` in ``updates`` in ``path``, replacing or appending."""
    lines = path.read_text().splitlines() if path.exists() else []
    out: list[str] = []
    seen: set[str] = set()
    for line in lines:
        m = re.match(r"\s*([A-Z0-9_]+)\s*=", line)
        if m and m.group(1) in updates:
            out.append(f"{m.group(1)}={updates[m.group(1)]}")
            seen.add(m.group(1))
        else:
            out.append(line)
    for key, val in updates.items():
        if key not in seen:
            out.append(f"{key}={val}")
    path.write_text("\n".join(out) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Exchange + verify a Meta/IG token.")
    parser.add_argument("--short-token", required=True, help="Short-lived token from the Graph API Explorer.")
    parser.add_argument("--write", action="store_true", help="Save token + user id to .env (git-ignored).")
    args = parser.parse_args()

    env = _AuthEnv()  # type: ignore[call-arg]  # app id/secret come from .env
    common = {"app_id": env.meta_app_id, "app_secret": env.meta_app_secret, "version": env.graph_api_version}

    print("→ Exchanging short-lived token for a long-lived one…")
    long_token, expires_in = exchange_token(args.short_token, **common)
    days = f"{expires_in / 86400:.0f} days" if expires_in else "unknown"
    print(f"  long-lived token: {mask_token(long_token)}  (expires in ~{days})")

    print("→ Finding the linked Instagram Business account…")
    ig_id, username = find_ig_user_id(long_token, version=env.graph_api_version)
    print(f"  ig_user_id: {ig_id}  (@{username})")

    print("→ Verifying the token can read the account…")
    prof = verify(long_token, ig_id, version=env.graph_api_version)
    print(f"  ok: @{prof.get('username')} — {prof.get('followers_count')} followers, {prof.get('media_count')} media")

    if args.write:
        _upsert_env(
            _ENV_PATH,
            {"INSTA_IG_ACCESS_TOKEN": long_token, "INSTA_IG_USER_ID": ig_id},
        )
        print(f"✓ Wrote INSTA_IG_ACCESS_TOKEN + INSTA_IG_USER_ID to {_ENV_PATH} (git-ignored).")
    else:
        print("• Not saved. Re-run with --write to persist to .env, or copy the values manually.")


if __name__ == "__main__":
    main()
