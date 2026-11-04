"""Land recent IG media into ``posts`` and ``post_metrics``; insight columns stay NULL.

    uv run python -m insta_analytics.collect.land --limit 5
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any

import duckdb

from ..config import Settings, get_settings, require_real_source
from ..storage import db
from . import client

# IG media fields that carry over unchanged into a ``posts`` row.
_POST_PASSTHROUGH = ("media_type", "media_product_type", "caption", "permalink")

# Every metrics row carries all insight columns, because the upsert takes its columns from row 0.
_POST_INSIGHT_COLUMNS = ("reach", "saved", "shares", "views", "total_interactions")


def _parse_timestamp(raw: str | None) -> datetime | None:
    """Parse IG's ISO timestamp (e.g. ``2024-06-01T12:34:56+0000``)."""
    if not raw:
        return None
    return datetime.fromisoformat(raw)


def media_to_rows(
    media: list[dict],
    *,
    ig_user_id: str,
    collected_at: datetime,
) -> tuple[list[dict], list[dict]]:
    """Split raw media dicts into ``(posts, post_metrics)`` rows; missing fields become NULL."""
    posts: list[dict] = []
    metrics: list[dict] = []
    for m in media:
        post_id = m["id"]
        posts.append(
            {
                "post_id": post_id,
                "ig_user_id": ig_user_id,
                **{c: m.get(c) for c in _POST_PASSTHROUGH},
                "timestamp": _parse_timestamp(m.get("timestamp")),
                "theme": None,
                "ingested_at": collected_at,
            }
        )
        metrics.append(
            {
                "post_id": post_id,
                "collected_at": collected_at,
                "like_count": m.get("like_count"),
                "comments_count": m.get("comments_count"),
                **{c: None for c in _POST_INSIGHT_COLUMNS},
            }
        )
    return posts, metrics


def land_media(
    media: list[dict] | None = None,
    *,
    con: duckdb.DuckDBPyConnection | None = None,
    settings: Settings | None = None,
    limit: int = 5,
    raw_dir: str | None = None,
    land_raw: bool = True,
) -> dict[str, int]:
    """Fetch (if needed) and upsert media into DuckDB. Returns rows-upserted counts."""
    settings = settings or get_settings()
    require_real_source(settings)
    if media is None:
        media = client.fetch_media(limit=limit, settings=settings)
    if land_raw:
        db.land_raw({"data": media}, "media", raw_dir=raw_dir)

    collected_at = datetime.now(timezone.utc)
    posts, metrics = media_to_rows(media, ig_user_id=settings.ig_user_id, collected_at=collected_at)

    with db.connection(con, settings) as con:
        db.bootstrap(con)
        n_posts = db.upsert(con, "posts", posts)
        n_metrics = db.upsert(con, "post_metrics", metrics)
    return {"posts": n_posts, "post_metrics": n_metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and land recent IG media into DuckDB.")
    parser.add_argument("--limit", type=int, default=5, help="How many recent media to fetch.")
    args = parser.parse_args()

    counts = land_media(limit=args.limit)
    print(f"→ Landed {counts['posts']} posts, {counts['post_metrics']} metric snapshots.")


if __name__ == "__main__":
    main()
