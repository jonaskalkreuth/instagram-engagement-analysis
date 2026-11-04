"""Pull current follower demographics (age, city, country, gender) into ``follower_demographics``.

The API only returns the current makeup, so each run is stored as a dated snapshot.

    uv run python -m insta_analytics.collect.demographics
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone

import duckdb
import httpx

from ..config import Settings, get_settings, require_real_source
from ..storage import db
from . import client

_BREAKDOWNS = ("age", "city", "country", "gender")


def fetch_follower_demographics(
    *,
    settings: Settings | None = None,
    client_: httpx.Client | None = None,
) -> list[dict]:
    """Fetch follower demographics, one request per breakdown; failing breakdowns are skipped."""
    settings = settings or get_settings()
    base = {
        "metric": "follower_demographics",
        "period": "lifetime",
        "metric_type": "total_value",
        "access_token": settings.ig_access_token,
    }

    close = client_ is None
    client_ = client_ or httpx.Client(base_url=client.GRAPH_BASE, timeout=30)
    path = f"/{settings.graph_api_version}/{settings.ig_user_id}/insights"
    out: list[dict] = []
    try:
        for breakdown in _BREAKDOWNS:
            params = {**base, "breakdown": breakdown}
            try:
                data = client._request(client_, path, params)
            except client.GraphAPIError as exc:
                print(f"  · skipping demographics breakdown {breakdown!r}: {exc}")
                continue
            out.extend(data.get("data", []))
    finally:
        if close:
            client_.close()
    return out


def demographics_to_rows(
    data: list[dict],
    *,
    ig_user_id: str,
    collected_at: datetime,
    snapshot_date: date,
) -> list[dict]:
    """Flatten raw ``follower_demographics`` dicts into one row per breakdown result."""
    rows: list[dict] = []
    for metric_obj in data:
        total_value = metric_obj.get("total_value") or {}
        for bd in total_value.get("breakdowns", []):
            keys = bd.get("dimension_keys") or []
            breakdown = keys[0] if keys else None
            for res in bd.get("results", []):
                dims = res.get("dimension_values") or []
                rows.append(
                    {
                        "ig_user_id": ig_user_id,
                        "snapshot_date": snapshot_date,
                        "breakdown": breakdown,
                        "dimension": dims[0] if dims else None,
                        "value": res.get("value"),
                        "collected_at": collected_at,
                    }
                )
    return rows


def snapshot_demographics(
    data: list[dict] | None = None,
    *,
    con: duckdb.DuckDBPyConnection | None = None,
    settings: Settings | None = None,
    collected_at: datetime | None = None,
    raw_dir: str | None = None,
    land_raw: bool = True,
) -> dict[str, int]:
    """Fetch (if needed) and upsert today's demographics snapshot. Returns rows-upserted counts."""
    settings = settings or get_settings()
    require_real_source(settings)
    if data is None:
        data = fetch_follower_demographics(settings=settings)
    if land_raw:
        db.land_raw({"data": data}, "follower_demographics", raw_dir=raw_dir)

    collected_at = collected_at or datetime.now(timezone.utc)
    rows = demographics_to_rows(
        data,
        ig_user_id=settings.ig_user_id,
        collected_at=collected_at,
        snapshot_date=collected_at.date(),
    )

    with db.connection(con, settings) as con:
        db.bootstrap(con)
        n = db.upsert(con, "follower_demographics", rows)
    return {"follower_demographics": n}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pull current follower demographics into DuckDB."
    )
    parser.parse_args()

    counts = snapshot_demographics()
    print(f"→ Landed {counts['follower_demographics']} follower-demographics rows.")


if __name__ == "__main__":
    main()
