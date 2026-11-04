"""Pull daily account insights (reach, views, follower_count) into ``account_snapshots``.

The API only serves the last ~30 days, so run this regularly to build the series.

    uv run python -m insta_analytics.collect.account --days 30
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

import duckdb
import httpx

from ..config import Settings, get_settings, require_real_source
from ..storage import db
from . import client

# Metrics that return a daily series but require metric_type=time_series (v21+).
_TIME_SERIES_METRICS = ("reach", "views")
# Legacy daily-series metrics that predate (and reject) the metric_type param.
_LEGACY_SERIES_METRICS = ("follower_count",)


def _end_time_to_date(end_time: str):
    """Map an insights ``end_time`` to its date; IG stamps the end of the day, so subtract one."""
    return (datetime.fromisoformat(end_time) - timedelta(days=1)).date()


def fetch_account_insights(
    *,
    days: int = 30,
    settings: Settings | None = None,
    client_: httpx.Client | None = None,
) -> list[dict]:
    """Fetch account insights for the last ``days``, one request per metric; failing metrics are skipped."""
    settings = settings or get_settings()
    until = datetime.now(timezone.utc)
    since = until - timedelta(days=days)
    base = {
        "period": "day",
        "since": str(int(since.timestamp())),
        "until": str(int(until.timestamp())),
        "access_token": settings.ig_access_token,
    }

    close = client_ is None
    client_ = client_ or httpx.Client(base_url=client.GRAPH_BASE, timeout=30)
    path = f"/{settings.graph_api_version}/{settings.ig_user_id}/insights"
    out: list[dict] = []
    try:
        for metric in (*_TIME_SERIES_METRICS, *_LEGACY_SERIES_METRICS):
            params = {**base, "metric": metric}
            if metric in _TIME_SERIES_METRICS:
                params["metric_type"] = "time_series"
            try:
                data = client._request(client_, path, params)
            except client.GraphAPIError as exc:
                print(f"  · skipping account metric {metric!r}: {exc}")
                continue
            out.extend(data.get("data", []))
    finally:
        if close:
            client_.close()
    return out


def insights_to_rows(
    data: list[dict],
    *,
    ig_user_id: str,
    collected_at: datetime,
) -> list[dict]:
    """Flatten raw insights metric dicts into long ``account_snapshots`` rows."""
    rows: list[dict] = []
    for metric_obj in data:
        metric = metric_obj["name"]
        for v in metric_obj.get("values", []):
            rows.append(
                {
                    "ig_user_id": ig_user_id,
                    "date": _end_time_to_date(v["end_time"]),
                    "metric": metric,
                    "value": v.get("value"),
                    "collected_at": collected_at,
                }
            )
    return rows


def snapshot_account(
    data: list[dict] | None = None,
    *,
    con: duckdb.DuckDBPyConnection | None = None,
    settings: Settings | None = None,
    days: int = 30,
    raw_dir: str | None = None,
    land_raw: bool = True,
) -> dict[str, int]:
    """Fetch (if needed) and upsert account insights. Returns rows-upserted counts."""
    settings = settings or get_settings()
    require_real_source(settings)
    if data is None:
        data = fetch_account_insights(days=days, settings=settings)
    if land_raw:
        db.land_raw({"data": data}, "account_insights", raw_dir=raw_dir)

    collected_at = datetime.now(timezone.utc)
    rows = insights_to_rows(data, ig_user_id=settings.ig_user_id, collected_at=collected_at)

    with db.connection(con, settings) as con:
        db.bootstrap(con)
        n = db.upsert(con, "account_snapshots", rows)
    return {"account_snapshots": n}


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull daily account insights into DuckDB.")
    parser.add_argument("--days", type=int, default=30, help="Rolling window size (max ~30).")
    args = parser.parse_args()

    counts = snapshot_account(days=args.days)
    print(f"→ Landed {counts['account_snapshots']} account-snapshot rows.")


if __name__ == "__main__":
    main()
