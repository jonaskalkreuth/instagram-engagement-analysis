"""Read-only Instagram Graph API client for media and shared request handling.

    uv run python -m insta_analytics.collect.client --limit 5
"""

from __future__ import annotations

import argparse
import json

import httpx

from ..config import Settings, get_settings

GRAPH_BASE = "https://graph.facebook.com"

MEDIA_FIELDS = (
    "id,caption,media_type,media_product_type,permalink,timestamp,"
    "like_count,comments_count"
)


class GraphAPIError(RuntimeError):
    """A Graph API call failed or returned an error body."""


def _request(client: httpx.Client, path: str, params: dict[str, str]) -> dict:
    """GET ``path`` and return parsed JSON, raising :class:`GraphAPIError` on failure."""
    resp = client.get(path, params=params)
    try:
        data = resp.json()
    except ValueError:
        resp.raise_for_status()
        raise GraphAPIError(f"Non-JSON response from {path}: {resp.text[:200]}")
    if isinstance(data, dict) and "error" in data:
        err = data["error"]
        raise GraphAPIError(f"Graph API error {err.get('code')}: {err.get('message')}")
    resp.raise_for_status()
    return data


def fetch_media(
    *,
    limit: int = 5,
    settings: Settings | None = None,
    client: httpx.Client | None = None,
) -> list[dict]:
    """Fetch up to ``limit`` recent raw media dicts for the configured account."""
    settings = settings or get_settings()
    close = client is None
    client = client or httpx.Client(base_url=GRAPH_BASE, timeout=30)
    try:
        data = _request(
            client,
            f"/{settings.graph_api_version}/{settings.ig_user_id}/media",
            {
                "fields": MEDIA_FIELDS,
                "limit": str(limit),
                "access_token": settings.ig_access_token,
            },
        )
    finally:
        if close:
            client.close()
    return data.get("data", [])


def fetch_all_media(
    *,
    page_size: int = 100,
    max_pages: int | None = None,
    settings: Settings | None = None,
    client: httpx.Client | None = None,
) -> list[dict]:
    """Follow the paging cursor and return every raw media dict, up to ``max_pages``."""
    settings = settings or get_settings()
    close = client is None
    client = client or httpx.Client(base_url=GRAPH_BASE, timeout=30)
    path = f"/{settings.graph_api_version}/{settings.ig_user_id}/media"
    out: list[dict] = []
    after: str | None = None
    pages = 0
    try:
        while True:
            params = {
                "fields": MEDIA_FIELDS,
                "limit": str(page_size),
                "access_token": settings.ig_access_token,
            }
            if after:
                params["after"] = after
            data = _request(client, path, params)
            out.extend(data.get("data", []))
            pages += 1
            after = (data.get("paging", {}).get("cursors", {}) or {}).get("after")
            # No more media once the API stops handing back a next cursor.
            if not after or "next" not in data.get("paging", {}):
                break
            if max_pages is not None and pages >= max_pages:
                break
    finally:
        if close:
            client.close()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch a few recent IG media.")
    parser.add_argument("--limit", type=int, default=5, help="How many recent media to fetch.")
    args = parser.parse_args()

    media = fetch_media(limit=args.limit)
    print(f"→ Fetched {len(media)} media:")
    print(json.dumps(media, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
