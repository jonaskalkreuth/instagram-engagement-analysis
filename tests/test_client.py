"""Unit tests for the media client — all network mocked via MockTransport."""

from __future__ import annotations

import httpx
import pytest

from insta_analytics.collect import client
from insta_analytics.config import Settings

SETTINGS = Settings(
    meta_app_id="111",
    meta_app_secret="shh",
    ig_access_token="LONG",
    ig_user_id="17890",
    graph_api_version="v21.0",
)


def _client(handler) -> httpx.Client:
    return httpx.Client(base_url=client.GRAPH_BASE, transport=httpx.MockTransport(handler))


def test_fetch_media_hits_right_endpoint_with_fields_and_limit():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, json={"data": [{"id": "1"}, {"id": "2"}]})

    media = client.fetch_media(limit=2, settings=SETTINGS, client=_client(handler))

    assert media == [{"id": "1"}, {"id": "2"}]
    assert seen["path"] == "/v21.0/17890/media"
    assert seen["params"]["fields"] == client.MEDIA_FIELDS
    assert seen["params"]["limit"] == "2"
    assert seen["params"]["access_token"] == "LONG"


def test_fetch_media_missing_data_key():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    assert client.fetch_media(settings=SETTINGS, client=_client(handler)) == []


def test_graph_error_body_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"code": 190, "message": "Invalid OAuth access token."}})

    with pytest.raises(client.GraphAPIError, match="190: Invalid OAuth"):
        client.fetch_media(settings=SETTINGS, client=_client(handler))


# --- pagination -----------------------------------------


def test_fetch_all_media_follows_cursor_across_pages():
    seen_after: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        after = request.url.params.get("after")
        seen_after.append(after)
        if after is None:  # page 1 → hands back a next cursor
            return httpx.Response(
                200,
                json={"data": [{"id": "1"}, {"id": "2"}], "paging": {"next": "u", "cursors": {"after": "CUR2"}}},
            )
        return httpx.Response(200, json={"data": [{"id": "3"}]})  # page 2 → no cursor, stop

    media = client.fetch_all_media(page_size=2, settings=SETTINGS, client=_client(handler))

    assert [m["id"] for m in media] == ["1", "2", "3"]
    assert seen_after == [None, "CUR2"]


def test_fetch_all_media_respects_max_pages():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"data": [{"id": "x"}], "paging": {"next": "u", "cursors": {"after": "MORE"}}}
        )

    media = client.fetch_all_media(page_size=1, max_pages=2, settings=SETTINGS, client=_client(handler))
    assert len(media) == 2  # stopped after 2 pages despite endless cursors
