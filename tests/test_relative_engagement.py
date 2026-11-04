"""Era-baseline relative engagement — the crux, tested on known fixtures."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from insta_analytics.config import Settings
from insta_analytics.storage import db
from insta_analytics.transform import relative_engagement as re

SETTINGS = Settings(
    meta_app_id="111", meta_app_secret="shh", ig_access_token="LONG", ig_user_id="u1"
)
BASE_TS = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _frame(engagements):
    """Ordered engagement frame with one row per synthetic post."""
    return pd.DataFrame(
        {
            "post_id": [f"p{i}" for i in range(len(engagements))],
            "timestamp": [BASE_TS + timedelta(days=i) for i in range(len(engagements))],
            "engagement": [float(e) for e in engagements],
        }
    )


# --- era_baseline: the median math ---------------------------------------


def test_baseline_is_leave_one_out_centered_median():
    # window=3 → one neighbor each side, self excluded → median of the two.
    vals = [10.0, 20.0, 40.0, 80.0, 160.0]
    out = re.era_baseline(vals, window=3, min_periods=2)
    # i=1: neighbors [10, 40] → 25 ; i=2: [20, 80] → 50 ; i=3: [40, 160] → 100
    assert out[1] == 25.0 and out[2] == 50.0 and out[3] == 100.0


def test_baseline_endpoints_use_available_neighbors():
    out = re.era_baseline([2.0, 4.0, 6.0, 8.0], window=3, min_periods=1)
    assert out[0] == 4.0  # only right neighbor available
    assert out[-1] == 6.0  # only left neighbor available


def test_baseline_nan_when_below_min_periods():
    out = re.era_baseline([2.0, 4.0, 6.0, 8.0], window=3, min_periods=2)
    assert np.isnan(out[0]) and np.isnan(out[-1])  # ends have only 1 neighbor
    assert out[1] == 4.0 and out[2] == 6.0  # interior has 2


def test_baseline_ignores_nan_neighbors():
    out = re.era_baseline([np.nan, 10.0, np.nan, 30.0, np.nan], window=5, min_periods=1)
    assert out[2] == 20.0  # median of [10, 30], NaNs dropped


def test_baseline_rejects_degenerate_window():
    with pytest.raises(ValueError):
        re.era_baseline([1.0, 2.0], window=1, min_periods=1)


# --- relative_engagement: ratio + log2 -----------------------------------


def test_relative_is_engagement_over_baseline():
    df = _frame([10, 10, 40, 10, 10])  # middle post at 4× its flat era
    out = re.relative_engagement(df, window=5, min_periods=3)
    mid = out.loc[out["post_id"] == "p2"].iloc[0]
    assert mid["era_baseline"] == 10.0
    assert mid["rel_engagement"] == 4.0
    assert mid["rel_engagement_log2"] == pytest.approx(2.0)  # log2(4)


def test_relative_resorts_by_timestamp_before_windowing():
    df = _frame([10, 40, 10]).iloc[::-1].reset_index(drop=True)  # shuffled
    out = re.relative_engagement(df, window=3, min_periods=1)
    # p1 (the 40) must still be measured against its two 10-neighbors.
    assert out.loc[out["post_id"] == "p1", "rel_engagement"].iloc[0] == 4.0


def test_relative_nan_baseline_yields_nan_not_crash():
    df = _frame([5, 7])  # window too small → both ends NaN baseline
    out = re.relative_engagement(df, window=3, min_periods=2)
    assert out["rel_engagement"].isna().all()  # no ZeroDivision / inf


def test_relative_zero_baseline_yields_nan():
    df = _frame([0, 0, 0, 50, 0])  # baseline of the 50 is median of zeros = 0
    out = re.relative_engagement(df, window=5, min_periods=3)
    assert np.isnan(out.loc[out["post_id"] == "p3", "rel_engagement"].iloc[0])


# --- window_sensitivity ---------------------------------------------------


def test_sensitivity_reports_a_row_per_window_and_stable_ranking():
    # Flat era (10s) with a few distinct spikes: one spike never moves a median
    # of ~30 tens, so every window scores flats≈1 and spikes at 2×/3×/5×. That
    # ranking is identical across windows → Spearman ≈ 1 (the robustness claim).
    eng = [10] * 40
    eng[10], eng[20], eng[30] = 20, 30, 50
    out = re.window_sensitivity(_frame(eng), windows=(11, 21, 31), min_periods=3, reference=21)
    assert list(out["window"]) == [11, 21, 31]
    assert (out["spearman_vs_ref"] > 0.95).all()


# --- load_engagement: DB integration + snapshot collapse ------------------


@pytest.fixture
def con():
    c = db.connect(":memory:")
    db.bootstrap(c)
    yield c
    c.close()


def _post(con, post_id, ts):
    con.execute(
        "INSERT INTO posts (post_id, ig_user_id, media_type, media_product_type, "
        "caption, permalink, timestamp, theme, ingested_at) "
        "VALUES (?, 'u1', 'IMAGE', 'FEED', 'c', 'http://x', ?, NULL, ?)",
        [post_id, ts, ts],
    )


def _metric(con, post_id, collected_at, likes, comments):
    con.execute(
        "INSERT INTO post_metrics (post_id, collected_at, like_count, comments_count) "
        "VALUES (?, ?, ?, ?)",
        [post_id, collected_at, likes, comments],
    )


def test_load_engagement_sums_and_orders_by_timestamp(con):
    _post(con, "b", BASE_TS + timedelta(days=1))
    _post(con, "a", BASE_TS)
    _metric(con, "a", BASE_TS, 10, 2)
    _metric(con, "b", BASE_TS + timedelta(days=1), 20, 5)
    out = re.load_engagement(con=con, settings=SETTINGS)
    assert list(out["post_id"]) == ["a", "b"]  # timestamp order
    assert list(out["engagement"]) == [12, 25]  # likes + comments


def test_load_engagement_takes_latest_snapshot_per_post(con):
    _post(con, "a", BASE_TS)
    _metric(con, "a", BASE_TS + timedelta(hours=1), 10, 0)  # earlier
    _metric(con, "a", BASE_TS + timedelta(hours=9), 99, 1)  # latest wins
    out = re.load_engagement(con=con, settings=SETTINGS)
    assert len(out) == 1 and out["engagement"].iloc[0] == 100
