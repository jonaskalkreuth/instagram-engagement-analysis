"""Account snapshot pull — insights mapping + rolling-window upsert."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from insta_analytics.collect import account
from insta_analytics.config import Settings
from insta_analytics.storage import db

COLLECTED_AT = datetime(2024, 6, 10, 9, 0, tzinfo=timezone.utc)

SETTINGS = Settings(
    meta_app_id="111",
    meta_app_secret="shh",
    ig_access_token="LONG",
    ig_user_id="u1",
    graph_api_version="v21.0",
    data_source="real",  # collectors refuse the synthetic source
)


def _metric(name, points):
    """A raw insights metric dict: points is a list of (end_time, value)."""
    return {
        "name": name,
        "period": "day",
        "values": [{"end_time": et, "value": val} for et, val in points],
    }


@pytest.fixture
def con():
    c = db.connect(":memory:")
    db.bootstrap(c)
    yield c
    c.close()


def _count(con, table):
    return con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


# --- end_time -> date convention -----------------------------------------


def test_end_time_maps_to_covered_day():
    # 07:00 UTC boundary of June 2 is the value *for* June 1.
    assert account._end_time_to_date("2024-06-02T07:00:00+0000") == date(2024, 6, 1)


# --- pure mapping ---------------------------------------------------------


def test_insights_to_rows_flattens_long_format():
    data = [
        _metric("reach", [("2024-06-02T07:00:00+0000", 100), ("2024-06-03T07:00:00+0000", 120)]),
        _metric("views", [("2024-06-02T07:00:00+0000", 300)]),
    ]
    rows = account.insights_to_rows(data, ig_user_id="u1", collected_at=COLLECTED_AT)
    assert rows == [
        {"ig_user_id": "u1", "date": date(2024, 6, 1), "metric": "reach", "value": 100, "collected_at": COLLECTED_AT},
        {"ig_user_id": "u1", "date": date(2024, 6, 2), "metric": "reach", "value": 120, "collected_at": COLLECTED_AT},
        {"ig_user_id": "u1", "date": date(2024, 6, 1), "metric": "views", "value": 300, "collected_at": COLLECTED_AT},
    ]


def test_insights_to_rows_empty_values_contribute_nothing():
    data = [_metric("follower_count", [])]  # e.g. <100 followers hides the series
    assert account.insights_to_rows(data, ig_user_id="u1", collected_at=COLLECTED_AT) == []


# --- end-to-end land into DuckDB -----------------------------------------


def test_snapshot_account_populates_table(con):
    data = [_metric("reach", [("2024-06-02T07:00:00+0000", 100)]), _metric("views", [("2024-06-02T07:00:00+0000", 300)])]
    counts = account.snapshot_account(data, con=con, settings=SETTINGS, land_raw=False)
    assert counts == {"account_snapshots": 2}
    assert _count(con, "account_snapshots") == 2
    row = con.execute(
        "SELECT value FROM account_snapshots WHERE metric='reach' AND date='2024-06-01'"
    ).fetchone()
    assert row[0] == 100


def test_snapshot_account_rolling_window_is_idempotent(con):
    # Two overlapping pulls: the second re-sends June 1 and adds June 2.
    week1 = [_metric("reach", [("2024-06-02T07:00:00+0000", 100)])]
    week2 = [_metric("reach", [("2024-06-02T07:00:00+0000", 100), ("2024-06-03T07:00:00+0000", 120)])]
    account.snapshot_account(week1, con=con, settings=SETTINGS, land_raw=False)
    account.snapshot_account(week2, con=con, settings=SETTINGS, land_raw=False)
    assert _count(con, "account_snapshots") == 2  # dedup on (ig_user_id, date, metric)
    val = con.execute(
        "SELECT value FROM account_snapshots WHERE metric='reach' AND date='2024-06-02'"
    ).fetchone()
    assert val[0] == 120


def test_snapshot_account_revised_value_overwrites(con):
    # IG revises a day's value (48h lag): same key, new value should win.
    account.snapshot_account([_metric("reach", [("2024-06-02T07:00:00+0000", 100)])], con=con, settings=SETTINGS, land_raw=False)
    account.snapshot_account([_metric("reach", [("2024-06-02T07:00:00+0000", 150)])], con=con, settings=SETTINGS, land_raw=False)
    val = con.execute("SELECT value FROM account_snapshots WHERE date='2024-06-01'").fetchone()
    assert val[0] == 150


# --- fetch skips a broken metric -----------------------------------------


def test_fetch_skips_metric_that_errors():
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("metric") == "views":
            return httpx.Response(400, json={"error": {"code": 100, "message": "deprecated"}})
        return httpx.Response(200, json={"data": [_metric(request.url.params["metric"], [("2024-06-02T07:00:00+0000", 5)])]})

    transport = httpx.MockTransport(handler)
    with httpx.Client(base_url=account.client.GRAPH_BASE, transport=transport) as c:
        data = account.fetch_account_insights(days=30, settings=SETTINGS, client_=c)

    names = {m["name"] for m in data}
    assert "views" not in names  # errored → skipped
    assert {"reach", "follower_count"} <= names  # survivors returned
