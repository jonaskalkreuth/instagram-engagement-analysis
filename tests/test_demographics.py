"""Follower demographics — breakdown mapping + snapshot upsert."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from insta_analytics.collect import demographics
from insta_analytics.config import Settings
from insta_analytics.storage import db

DAY1 = datetime(2024, 6, 10, 9, 0, tzinfo=timezone.utc)
DAY2 = datetime(2024, 6, 17, 9, 0, tzinfo=timezone.utc)

SETTINGS = Settings(
    meta_app_id="111",
    meta_app_secret="shh",
    ig_access_token="LONG",
    ig_user_id="u1",
    graph_api_version="v21.0",
    data_source="real",  # collectors refuse the synthetic source
)


def _demo_metric(breakdown, results):
    """A raw follower_demographics dict: results is a list of (dimension, value)."""
    return {
        "name": "follower_demographics",
        "period": "lifetime",
        "total_value": {
            "breakdowns": [
                {
                    "dimension_keys": [breakdown],
                    "results": [
                        {"dimension_values": [dv], "value": val} for dv, val in results
                    ],
                }
            ]
        },
    }


@pytest.fixture
def con():
    c = db.connect(":memory:")
    db.bootstrap(c)
    yield c
    c.close()


def _count(con, table):
    return con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


# --- pure mapping ---------------------------------------------------------


def test_demographics_to_rows_flattens_breakdowns():
    data = [
        _demo_metric("age", [("18-24", 100), ("25-34", 200)]),
        _demo_metric("country", [("US", 250)]),
    ]
    rows = demographics.demographics_to_rows(
        data, ig_user_id="u1", collected_at=DAY1, snapshot_date=date(2024, 6, 10)
    )
    assert rows == [
        {"ig_user_id": "u1", "snapshot_date": date(2024, 6, 10), "breakdown": "age", "dimension": "18-24", "value": 100, "collected_at": DAY1},
        {"ig_user_id": "u1", "snapshot_date": date(2024, 6, 10), "breakdown": "age", "dimension": "25-34", "value": 200, "collected_at": DAY1},
        {"ig_user_id": "u1", "snapshot_date": date(2024, 6, 10), "breakdown": "country", "dimension": "US", "value": 250, "collected_at": DAY1},
    ]


def test_demographics_to_rows_missing_total_value_contributes_nothing():
    # <100 followers: IG withholds the breakdown, so no total_value key.
    data = [{"name": "follower_demographics", "period": "lifetime"}]
    assert demographics.demographics_to_rows(
        data, ig_user_id="u1", collected_at=DAY1, snapshot_date=date(2024, 6, 10)
    ) == []


# --- end-to-end land into DuckDB -----------------------------------------


def test_snapshot_populates_table(con):
    data = [_demo_metric("age", [("18-24", 100)]), _demo_metric("gender", [("F", 400)])]
    counts = demographics.snapshot_demographics(
        data, con=con, settings=SETTINGS, collected_at=DAY1, land_raw=False
    )
    assert counts == {"follower_demographics": 2}
    assert _count(con, "follower_demographics") == 2
    val = con.execute(
        "SELECT value FROM follower_demographics WHERE breakdown='gender' AND dimension='F'"
    ).fetchone()
    assert val[0] == 400


def test_same_day_rerun_is_idempotent_and_overwrites(con):
    # Same snapshot_date, revised value: dedup on the key, new value wins.
    demographics.snapshot_demographics(
        [_demo_metric("age", [("18-24", 100)])], con=con, settings=SETTINGS, collected_at=DAY1, land_raw=False
    )
    demographics.snapshot_demographics(
        [_demo_metric("age", [("18-24", 150)])], con=con, settings=SETTINGS, collected_at=DAY1, land_raw=False
    )
    assert _count(con, "follower_demographics") == 1
    val = con.execute("SELECT value FROM follower_demographics").fetchone()
    assert val[0] == 150


def test_different_days_build_a_trend(con):
    # Two weekly snapshots of the same dimension → two rows, one per snapshot_date.
    demographics.snapshot_demographics(
        [_demo_metric("age", [("18-24", 100)])], con=con, settings=SETTINGS, collected_at=DAY1, land_raw=False
    )
    demographics.snapshot_demographics(
        [_demo_metric("age", [("18-24", 130)])], con=con, settings=SETTINGS, collected_at=DAY2, land_raw=False
    )
    rows = con.execute(
        "SELECT snapshot_date, value FROM follower_demographics ORDER BY snapshot_date"
    ).fetchall()
    assert rows == [(date(2024, 6, 10), 100), (date(2024, 6, 17), 130)]


# --- fetch skips a broken breakdown --------------------------------------


def test_fetch_skips_breakdown_that_errors():
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        breakdown = request.url.params.get("breakdown")
        if breakdown == "gender":
            return httpx.Response(400, json={"error": {"code": 100, "message": "deprecated"}})
        return httpx.Response(200, json={"data": [_demo_metric(breakdown, [("x", 1)])]})

    transport = httpx.MockTransport(handler)
    with httpx.Client(base_url=demographics.client.GRAPH_BASE, transport=transport) as c:
        data = demographics.fetch_follower_demographics(settings=SETTINGS, client_=c)

    keys = {bd["dimension_keys"][0] for m in data for bd in m["total_value"]["breakdowns"]}
    assert "gender" not in keys  # errored → skipped
    assert {"age", "city", "country"} <= keys  # survivors returned
