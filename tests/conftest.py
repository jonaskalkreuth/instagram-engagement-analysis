"""Shared fixtures."""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def db_path(tmp_path):
    """A minimal DuckDB with the four tables the data layer reads."""
    n = 16
    rng = np.random.default_rng(0)
    ts = pd.date_range("2021-01-01", periods=n, freq="30D", tz="UTC")
    themes = np.array(["Live/Performance", "Show/Tour"] * (n // 2))
    formats = np.array(["Reel", "Image", "Carousel", "Video"] * (n // 4))
    log2 = rng.normal(0, 0.5, n)

    posts = pd.DataFrame(
        {
            "post_id": [f"p{i}" for i in range(n)],
            "timestamp": ts,
            "theme": themes,
            "format": formats,
            "caption_chars": rng.integers(50, 800, n),
            "caption_words": rng.integers(10, 150, n),
            "hashtag_count": rng.integers(0, 30, n),
            "mention_count": rng.integers(0, 5, n),
            "days_since_prev": rng.integers(1, 20, n),
            "posts_prev_30d": rng.integers(1, 10, n),
            "hour": rng.integers(0, 24, n),
            "dow": rng.integers(0, 7, n),
            "is_weekend": rng.integers(0, 2, n),
            "month": ts.month,
            "rel_engagement_log2": log2,
        }
    )
    reach = posts.assign(
        reach=rng.integers(1000, 5000, n),
        saved=rng.integers(10, 200, n),
        shares=rng.integers(5, 100, n),
        views=rng.integers(2000, 9000, n),
        total_interactions=rng.integers(100, 900, n),
        rel_reach_log2=rng.normal(0, 0.5, n),
    )
    split = pd.DataFrame(
        {
            "post_id": posts["post_id"],
            "like_count": rng.integers(100, 900, n),
            "comments_count": rng.integers(0, 60, n),
            "collected_at": ts,
        }
    )
    demo = pd.DataFrame(
        {
            "ig_user_id": ["1"] * 9,
            "snapshot_date": pd.Timestamp("2026-07-11"),
            "breakdown": (
                ["gender"] * 3 + ["age"] * 2 + ["country"] * 2 + ["city"] * 2
            ),
            "dimension": ["M", "F", "U", "25-34", "35-44", "BR", "MX", "Mexico City", "Berlin"],
            "value": [750.0, 138.0, 112.0, 500.0, 300.0, 220.0, 190.0, 180.0, 70.0],
            "collected_at": pd.Timestamp("2026-07-11", tz="UTC"),
        }
    )

    path = tmp_path / "test.duckdb"
    con = duckdb.connect(str(path))
    con.register("posts_v", posts)
    con.register("reach_v", reach)
    con.register("split_v", split)
    con.register("demo_v", demo)
    con.execute("CREATE TABLE analysis_posts AS SELECT * FROM posts_v")
    con.execute("CREATE TABLE analysis_reach AS SELECT * FROM reach_v")
    con.execute("CREATE TABLE post_metrics AS SELECT * FROM split_v")
    con.execute("CREATE TABLE follower_demographics AS SELECT * FROM demo_v")
    con.close()
    return path
