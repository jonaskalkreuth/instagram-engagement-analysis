"""Historical backfill — insights mapping, batched+fallback, upsert."""

from __future__ import annotations

import httpx
import pytest

from insta_analytics.collect import backfill
from insta_analytics.config import Settings
from insta_analytics.storage import db

SETTINGS = Settings(
    meta_app_id="111",
    meta_app_secret="shh",
    ig_access_token="LONG",
    ig_user_id="u1",
    graph_api_version="v21.0",
    data_source="real",  # collectors refuse the synthetic source
)


@pytest.fixture
def con():
    c = db.connect(":memory:")
    db.bootstrap(c)
    yield c
    c.close()


def _count(con, table):
    return con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


def _client(handler) -> httpx.Client:
    return httpx.Client(base_url=backfill.client.GRAPH_BASE, transport=httpx.MockTransport(handler))


def _metric(name, value):
    return {"name": name, "period": "lifetime", "values": [{"value": value}]}


# --- pure value extraction / mapping -------------------------------------


def test_metric_value_handles_both_shapes():
    assert backfill._metric_value({"values": [{"value": 7}]}) == 7
    assert backfill._metric_value({"total_value": {"value": 9}}) == 9
    assert backfill._metric_value({}) is None


def test_insights_to_fields_keeps_only_known_metrics():
    data = [_metric("reach", 100), _metric("saved", 5), _metric("profile_visits", 3)]
    assert backfill.insights_to_fields(data) == {"reach": 100, "saved": 5}


# --- batched + per-metric fallback ---------------------------------------


def test_fetch_media_insights_batched_success():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["metric"] == "reach,saved,shares,views,total_interactions"
        return httpx.Response(200, json={"data": [_metric("reach", 100), _metric("views", 300)]})

    got = backfill.fetch_media_insights("m1", settings=SETTINGS, client_=_client(handler))
    assert got == {"reach": 100, "views": 300}


def test_fetch_media_insights_falls_back_and_skips_bad_metric():
    def handler(request: httpx.Request) -> httpx.Response:
        metric = request.url.params["metric"]
        if "," in metric:  # batched call fails → forces fallback
            return httpx.Response(400, json={"error": {"code": 100, "message": "views not supported"}})
        if metric == "views":  # the offending single metric still errors → skipped
            return httpx.Response(400, json={"error": {"code": 100, "message": "unsupported"}})
        return httpx.Response(200, json={"data": [_metric(metric, 1)]})

    got = backfill.fetch_media_insights("m1", settings=SETTINGS, client_=_client(handler))
    assert "views" not in got
    assert got == {"reach": 1, "saved": 1, "shares": 1, "total_interactions": 1}


def test_request_with_backoff_retries_rate_limit(monkeypatch):
    monkeypatch.setattr(backfill.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(400, json={"error": {"code": 4, "message": "rate limited"}})
        return httpx.Response(200, json={"data": [_metric("reach", 42)]})

    with _client(handler) as c:
        data = backfill._request_with_backoff(c, "/v21.0/m1/insights", {"metric": "reach"})
    assert calls["n"] == 2  # retried once after the rate-limit error
    assert data["data"][0]["values"][0]["value"] == 42


def test_request_with_backoff_reraises_non_rate_limit():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"code": 190, "message": "bad token"}})

    with _client(handler) as c:
        with pytest.raises(backfill.client.GraphAPIError, match="190"):
            backfill._request_with_backoff(c, "/v21.0/m1/insights", {"metric": "reach"})


# --- end-to-end backfill into DuckDB -------------------------------------


def _media(post_id, ts="2024-06-01T12:00:00+0000"):
    return {
        "id": post_id,
        "caption": "c",
        "media_type": "IMAGE",
        "media_product_type": "FEED",
        "permalink": "http://x",
        "timestamp": ts,
        "like_count": 10,
        "comments_count": 2,
    }


def _backfill_handler(pages, insights_value=100):
    """Serve `pages` (list of media-lists) then insights for any media id."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/insights"):
            return httpx.Response(200, json={"data": [_metric("reach", insights_value)]})
        # /media page — index by the `after` cursor we hand out.
        after = request.url.params.get("after")
        idx = 0 if after is None else int(after)
        data = pages[idx]
        body = {"data": data}
        if idx + 1 < len(pages):
            body["paging"] = {"next": "u", "cursors": {"after": str(idx + 1)}}
        return httpx.Response(200, json=body)

    return handler


def test_backfill_populates_posts_and_insights(con):
    http = _client(_backfill_handler([[_media("a"), _media("b")]]))
    counts = backfill.backfill(con=con, settings=SETTINGS, client_=http, sleep=0, land_raw=False)

    assert counts == {"posts": 2, "post_metrics": 2}
    assert _count(con, "posts") == 2
    # insights column now populated (land.py leaves it NULL).
    reach = con.execute("SELECT reach FROM post_metrics WHERE post_id='a'").fetchone()[0]
    assert reach == 100


def test_backfill_max_posts_caps_pull(con):
    http = _client(_backfill_handler([[_media("a"), _media("b")]]))
    counts = backfill.backfill(con=con, settings=SETTINGS, client_=http, sleep=0, max_posts=1, land_raw=False)
    assert counts["posts"] == 1


def test_backfill_survives_heterogeneous_insight_support(con):
    """Regression: a post whose media type rejects a metric yields a metrics row
    missing that key. db.upsert derives its columns from row[0], so if the first
    post had all metrics and a later one didn't, it raised KeyError and the whole
    metrics upsert (an hour of live fetches) was lost. Every row is now
    schema-complete, so the unsupported metric lands as NULL instead of crashing."""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/insights"):
            media_id = path.split("/")[-2]
            metric = request.url.params["metric"]
            # Post 'a' supports everything; 'b' rejects 'shares' on batch + single.
            if media_id == "b" and ("," in metric or metric == "shares"):
                return httpx.Response(400, json={"error": {"code": 100, "message": "unsupported"}})
            names = metric.split(",")
            return httpx.Response(200, json={"data": [_metric(n, 100) for n in names]})
        return httpx.Response(200, json={"data": [_media("a"), _media("b")]})

    http = _client(handler)
    counts = backfill.backfill(con=con, settings=SETTINGS, client_=http, sleep=0, land_raw=False)

    assert counts == {"posts": 2, "post_metrics": 2}
    assert con.execute("SELECT shares FROM post_metrics WHERE post_id='a'").fetchone()[0] == 100
    assert con.execute("SELECT shares FROM post_metrics WHERE post_id='b'").fetchone()[0] is None


def test_backfill_rerun_keeps_theme(con):
    http = _client(_backfill_handler([[_media("a")]]))
    backfill.backfill(con=con, settings=SETTINGS, client_=http, sleep=0, land_raw=False)
    con.execute("UPDATE posts SET theme = 'Studio/BTS' WHERE post_id = 'a'")

    backfill.backfill(con=con, settings=SETTINGS, client_=http, sleep=0, land_raw=False)

    assert _count(con, "posts") == 1
    assert con.execute("SELECT theme FROM posts WHERE post_id='a'").fetchone()[0] == "Studio/BTS"
