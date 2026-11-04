"""Pull every post plus its per-post insights into ``posts`` and ``post_metrics``.

Each post gets one request for all metrics; if a media type rejects one, that post
falls back to one request per metric. Rate-limit errors are retried with backoff.

    uv run python -m insta_analytics.collect.backfill --sleep 1
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone

import duckdb
import httpx

from ..config import Settings, get_settings, require_real_source
from ..storage import db
from . import client, land

_POST_INSIGHT_METRICS = land._POST_INSIGHT_COLUMNS

# Graph error codes for rate limiting.
_RATE_LIMIT_CODES = (4, 17, 32, 613)


def _metric_value(metric_obj: dict) -> int | None:
    """Return the value from either the ``values`` or the ``total_value`` shape, else None."""
    values = metric_obj.get("values")
    if values:
        return values[0].get("value")
    total = metric_obj.get("total_value")
    if isinstance(total, dict):
        return total.get("value")
    return None


def insights_to_fields(data: list[dict]) -> dict[str, int | None]:
    """Flatten raw media-insights metric dicts into ``{metric: value}`` for stored metrics."""
    out: dict[str, int | None] = {}
    for metric_obj in data:
        name = metric_obj.get("name")
        if name in _POST_INSIGHT_METRICS:
            out[name] = _metric_value(metric_obj)
    return out


def _is_rate_limit(exc: client.GraphAPIError) -> bool:
    return any(f"error {code}:" in str(exc) for code in _RATE_LIMIT_CODES)


def _request_with_backoff(
    client_: httpx.Client,
    path: str,
    params: dict[str, str],
    *,
    max_retries: int = 5,
    base_delay: float = 2.0,
) -> dict:
    """``client._request`` with exponential backoff; only rate-limit errors are retried."""
    for attempt in range(max_retries):
        try:
            return client._request(client_, path, params)
        except client.GraphAPIError as exc:
            if not _is_rate_limit(exc) or attempt == max_retries - 1:
                raise
            wait = base_delay * (2**attempt)
            print(f"  · rate-limited, backing off {wait:.0f}s: {exc}")
            time.sleep(wait)
    raise AssertionError("unreachable")  # pragma: no cover


def fetch_media_insights(
    media_id: str,
    *,
    settings: Settings | None = None,
    client_: httpx.Client | None = None,
) -> dict[str, int | None]:
    """Fetch insights for one post, falling back to one request per metric if the batch fails."""
    settings = settings or get_settings()
    close = client_ is None
    client_ = client_ or httpx.Client(base_url=client.GRAPH_BASE, timeout=30)
    path = f"/{settings.graph_api_version}/{media_id}/insights"
    token = settings.ig_access_token
    try:
        try:
            data = _request_with_backoff(
                client_, path, {"metric": ",".join(_POST_INSIGHT_METRICS), "access_token": token}
            )
            return insights_to_fields(data.get("data", []))
        except client.GraphAPIError as exc:
            print(f"  · batched insights failed for {media_id}, per-metric fallback: {exc}")
            fields: dict[str, int | None] = {}
            for metric in _POST_INSIGHT_METRICS:
                try:
                    data = _request_with_backoff(
                        client_, path, {"metric": metric, "access_token": token}
                    )
                except client.GraphAPIError as e2:
                    print(f"    · skipping metric {metric!r} for {media_id}: {e2}")
                    continue
                fields.update(insights_to_fields(data.get("data", [])))
            return fields
    finally:
        if close:
            client_.close()


def backfill(
    *,
    con: duckdb.DuckDBPyConnection | None = None,
    settings: Settings | None = None,
    client_: httpx.Client | None = None,
    page_size: int = 100,
    max_posts: int | None = None,
    sleep: float = 1.0,
    raw_dir: str | None = None,
    land_raw: bool = True,
) -> dict[str, int]:
    """Pull all posts and their insights into DuckDB, sleeping ``sleep`` seconds between posts."""
    settings = settings or get_settings()
    require_real_source(settings)

    close_client = client_ is None
    http = client_ or httpx.Client(base_url=client.GRAPH_BASE, timeout=30)
    try:
        media = client.fetch_all_media(page_size=page_size, settings=settings, client=http)
        if max_posts is not None:
            media = media[:max_posts]
        if land_raw:
            db.land_raw({"data": media}, "media_backfill", raw_dir=raw_dir)

        collected_at = datetime.now(timezone.utc)
        posts, metrics = land.media_to_rows(
            media, ig_user_id=settings.ig_user_id, collected_at=collected_at
        )

        for i, m in enumerate(media):
            insights = fetch_media_insights(m["id"], settings=settings, client_=http)
            metrics[i].update(insights)
            if sleep:
                time.sleep(sleep)
    finally:
        if close_client:
            http.close()

    with db.connection(con, settings) as con:
        db.bootstrap(con)
        n_posts = db.upsert(con, "posts", posts)
        n_metrics = db.upsert(con, "post_metrics", metrics)
    return {"posts": n_posts, "post_metrics": n_metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Historical backfill: all posts + insights.")
    parser.add_argument("--page-size", type=int, default=100, help="Media per page.")
    parser.add_argument("--sleep", type=float, default=1.0, help="Seconds between insight calls.")
    parser.add_argument("--max-posts", type=int, default=None, help="Cap posts pulled (testing).")
    args = parser.parse_args()

    counts = backfill(page_size=args.page_size, sleep=args.sleep, max_posts=args.max_posts)
    print(f"→ Backfilled {counts['posts']} posts, {counts['post_metrics']} metric snapshots.")


if __name__ == "__main__":
    main()
