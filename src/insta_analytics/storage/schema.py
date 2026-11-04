"""DuckDB DDL and dedup keys for the four tables; real and synthetic data share them.

Columns are tagged *public* (safe to publish) or *insights* (login-only, publish
relative values only). Field names match the Graph API.
"""

from __future__ import annotations

# posts: one row per media, keyed by the IG media id
# public:  post_id, ig_user_id, media_type, media_product_type, caption,
#          permalink, timestamp
# derived: theme (hand-labeled later), ingested_at (our bookkeeping)
POSTS_DDL = """
CREATE TABLE IF NOT EXISTS posts (
    post_id            TEXT        NOT NULL,
    ig_user_id         TEXT        NOT NULL,
    media_type         TEXT,        -- IMAGE / VIDEO / CAROUSEL_ALBUM
    media_product_type TEXT,        -- FEED / REELS
    caption            TEXT,
    permalink          TEXT,
    timestamp          TIMESTAMPTZ, -- when the post was published
    theme              TEXT,        -- hand-labeled taxonomy; NULL until tagging
    ingested_at        TIMESTAMP,   -- when we first landed this post
    PRIMARY KEY (post_id)
)
"""

# post_metrics: time-varying numbers, one snapshot per collection
# public:   like_count, comments_count
# insights: reach, saved, shares, views, total_interactions (login-only)
POST_METRICS_DDL = """
CREATE TABLE IF NOT EXISTS post_metrics (
    post_id            TEXT      NOT NULL,
    collected_at       TIMESTAMP NOT NULL, -- snapshot time
    like_count         BIGINT,   -- public
    comments_count     BIGINT,   -- public
    reach              BIGINT,   -- insights
    saved              BIGINT,   -- insights
    shares             BIGINT,   -- insights
    views              BIGINT,   -- insights (replaced impressions, Aug 2024)
    total_interactions BIGINT,   -- insights
    PRIMARY KEY (post_id, collected_at)
)
"""

# account_snapshots: daily account time series, long format
# Long (metric/value) so overlapping rolling-30-day weekly pulls dedup cleanly
# on (ig_user_id, date, metric). Audience demographics don't fit a single daily
# scalar, so they live in their own long table below (follower_demographics).
# insights: every value here is login-only.
ACCOUNT_SNAPSHOTS_DDL = """
CREATE TABLE IF NOT EXISTS account_snapshots (
    ig_user_id   TEXT      NOT NULL,
    date         DATE      NOT NULL, -- the day the value is for
    metric       TEXT      NOT NULL, -- e.g. reach / follower_count / views
    value        BIGINT,
    collected_at TIMESTAMP,          -- when this value was pulled
    PRIMARY KEY (ig_user_id, date, metric)
)
"""

# follower_demographics: audience breakdowns, long format
# The API returns only the current makeup, so each pull is stamped with a
# snapshot_date and weekly pulls build the trend.
# insights: every value here is login-only.
FOLLOWER_DEMOGRAPHICS_DDL = """
CREATE TABLE IF NOT EXISTS follower_demographics (
    ig_user_id    TEXT NOT NULL,
    snapshot_date DATE NOT NULL, -- the day this snapshot was taken
    breakdown     TEXT NOT NULL, -- age / city / country / gender
    dimension     TEXT NOT NULL, -- e.g. "18-24" / "US" / "F"
    value         BIGINT,
    collected_at  TIMESTAMP,     -- when this snapshot was pulled
    PRIMARY KEY (ig_user_id, snapshot_date, breakdown, dimension)
)
"""

# Manifest consumed by the storage layer: table name -> (DDL, dedup keys).
# ``dedup_keys`` = the columns that identify one logical row (the upsert target).
# ``keep_on_conflict`` = columns an upsert never overwrites on an existing row:
# the hand-labelled theme and the first-landed time survive a collector re-run.
TABLES: dict[str, dict[str, object]] = {
    "posts": {
        "ddl": POSTS_DDL,
        "dedup_keys": ("post_id",),
        "keep_on_conflict": ("theme", "ingested_at"),
    },
    "post_metrics": {
        "ddl": POST_METRICS_DDL,
        "dedup_keys": ("post_id", "collected_at"),
    },
    "account_snapshots": {
        "ddl": ACCOUNT_SNAPSHOTS_DDL,
        "dedup_keys": ("ig_user_id", "date", "metric"),
    },
    "follower_demographics": {
        "ddl": FOLLOWER_DEMOGRAPHICS_DDL,
        "dedup_keys": ("ig_user_id", "snapshot_date", "breakdown", "dimension"),
    },
}

ALL_DDL: tuple[str, ...] = tuple(t["ddl"] for t in TABLES.values())  # type: ignore[misc]
