"""Saves & shares deep-dive — pure derivations + a DB round-trip."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from insta_analytics.analysis import insight_effects as ie
from insta_analytics.config import Settings
from insta_analytics.storage import db

SETTINGS = Settings(
    meta_app_id="111", meta_app_secret="shh", ig_access_token="LONG", ig_user_id="u1"
)
BASE_TS = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
INGESTED = datetime(2024, 6, 2, 10, 0, tzinfo=timezone.utc)


def _reach(n, **cols):
    """Minimal analysis_reach-shaped frame: post_id + timestamp + insight columns."""
    base = {
        "post_id": [f"p{i}" for i in range(n)],
        "timestamp": [BASE_TS + timedelta(days=i) for i in range(n)],
        "theme": ["a"] * n,
        "format": ["Reel"] * n,
        "reach": [100.0] * n,
        "saved": [10.0] * n,
        "shares": [5.0] * n,
        "total_interactions": [20.0] * n,
    }
    base.update(cols)
    return pd.DataFrame(base)


def _split(n, like=100, comment=10):
    return pd.DataFrame(
        {
            "post_id": [f"p{i}" for i in range(n)],
            "like_count": [float(like)] * n,
            "comments_count": [float(comment)] * n,
        }
    )


# --- rates ------------------------------------------------------------------
def test_rates_are_simple_ratios():
    df = ie.derive_insight_metrics(_reach(6), _split(6), window=3, min_periods=2)
    assert df["save_rate"].iloc[0] == pytest.approx(0.10)          # 10/100
    assert df["share_rate"].iloc[0] == pytest.approx(0.05)         # 5/100
    assert df["interaction_rate"].iloc[0] == pytest.approx(0.20)   # 20/100
    assert df["comment_share"].iloc[0] == pytest.approx(10 / 110)  # comments/(likes+comments)


def test_zero_denominator_rates_are_nan_not_inf():
    reach = _reach(4, reach=[0.0, 0.0, 0.0, 0.0])
    split = _split(4, like=0, comment=0)  # comment_share denom 0 too
    df = ie.derive_insight_metrics(reach, split, window=3, min_periods=2)
    assert df["save_rate"].isna().all()
    assert df["interaction_rate"].isna().all()
    assert df["comment_share"].isna().all()
    assert not np.isinf(df[["save_rate", "comment_share"]].to_numpy()).any()


# --- era-relative saves/shares ---------------------------------------------
def test_rel_saved_reflects_a_spike_above_baseline():
    saved = [10.0] * 11
    saved[5] = 40.0  # a 4x spike against ~10 neighbours
    df = ie.derive_insight_metrics(
        _reach(11, saved=saved), _split(11), window=5, min_periods=2
    )
    spike = df.loc[df["post_id"] == "p5"].iloc[0]
    assert spike["rel_saved"] == pytest.approx(4.0)  # 40 / median(neighbours=10)
    assert spike["rel_saved_log2"] == pytest.approx(2.0)


def test_split_joins_on_post_id_not_row_order():
    reach = _reach(3)
    split = _split(3).iloc[::-1].reset_index(drop=True)  # reversed order
    df = ie.derive_insight_metrics(reach, split, window=3, min_periods=2)
    # every post still gets its own like/comment counts (all equal here) → no NaN join
    assert df["comment_share"].notna().all()


# --- figures ----------------------------------------------------------------
def test_save_figures_writes_all_pngs(tmp_path):
    n = 18
    reach = _reach(
        n,
        theme=(["a", "b", "c"] * (n // 3)),
        format=(["Reel", "Image", "Carousel"] * (n // 3)),
        saved=list(10 + np.arange(n)),
        shares=list(5 + np.arange(n)),
    )
    df = ie.derive_insight_metrics(reach, _split(n), window=5, min_periods=2)
    paths = ie.save_figures(df, figures_dir=tmp_path, n_boot=100, min_n=1)
    assert len(paths) == 6
    assert all(p.exists() for p in paths)


# --- DB round-trip loader ---------------------------------------------------
@pytest.fixture
def con():
    c = db.connect(":memory:")
    db.bootstrap(c)
    yield c
    c.close()


def test_load_like_comment_split_takes_latest_snapshot(con):
    con.execute(
        "INSERT INTO posts (post_id, ig_user_id, media_type, media_product_type, "
        "caption, permalink, timestamp, theme, ingested_at) "
        "VALUES ('p0', 'u1', 'IMAGE', 'FEED', 'c', 'http://x', ?, 'a', ?)",
        [BASE_TS, INGESTED],
    )
    # two snapshots for p0 — the later collected_at must win
    con.execute(
        "INSERT INTO post_metrics (post_id, collected_at, like_count, comments_count) "
        "VALUES ('p0', ?, 5, 1), ('p0', ?, 50, 9)",
        [INGESTED, INGESTED + timedelta(days=1)],
    )
    out = ie.load_like_comment_split(con=con, settings=SETTINGS)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["like_count"] == 50 and row["comments_count"] == 9
