"""Analysis-table assembly — pure joins + DB round-trip."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from insta_analytics.config import Settings
from insta_analytics.storage import db
from insta_analytics.transform import analysis_table as at
from insta_analytics.transform import relative_engagement as re

SETTINGS = Settings(
    meta_app_id="111", meta_app_secret="shh", ig_access_token="LONG", ig_user_id="u1"
)
BASE_TS = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
INGESTED = datetime(2024, 6, 2, 10, 0, tzinfo=timezone.utc)


def _features(n):
    """Minimal predictor frame: post_id + timestamp + one stand-in predictor."""
    return pd.DataFrame(
        {
            "post_id": [f"p{i}" for i in range(n)],
            "timestamp": [BASE_TS + timedelta(days=i) for i in range(n)],
            "hashtag_count": list(range(n)),
        }
    )


def _target(post_ids):
    """Engagement target frame keyed by post_id (values are arbitrary but keyed)."""
    return pd.DataFrame(
        {
            "post_id": list(post_ids),
            "engagement": [10.0] * len(post_ids),
            "era_baseline": [5.0] * len(post_ids),
            "rel_engagement": [2.0] * len(post_ids),
            "rel_engagement_log2": [1.0] * len(post_ids),
        }
    )


# --- build_analysis_table -------------------------------------------------


def test_join_attaches_target_columns_and_keeps_all_predictors():
    out = at.build_analysis_table(_features(3), _target(["p0", "p1", "p2"]))
    assert len(out) == 3
    for col in ("hashtag_count", *at._TARGET_COLUMNS):
        assert col in out.columns
    assert list(out["post_id"]) == ["p0", "p1", "p2"]  # timestamp-ordered


def test_join_raises_when_a_predictor_has_no_target():
    # target missing p2 → inner join drops a row → bug, must raise.
    with pytest.raises(ValueError, match="missing its engagement target"):
        at.build_analysis_table(_features(3), _target(["p0", "p1"]))


# --- build_reach_subset ---------------------------------------------------


def test_reach_subset_drops_null_reach_rows():
    primary = at.build_analysis_table(_features(3), _target(["p0", "p1", "p2"]))
    reach = pd.DataFrame(
        {
            "post_id": ["p0", "p1", "p2"],
            "reach": [100.0, np.nan, 300.0],  # p1 predates reach
            "saved": [1, 2, 3],
            "shares": [1, 2, 3],
            "views": [10, 20, 30],
            "total_interactions": [5, 6, 7],
        }
    )
    subset = at.build_reach_subset(primary, reach)
    assert list(subset["post_id"]) == ["p0", "p2"]  # p1 dropped
    assert "reach" in subset.columns and "rel_reach" in subset.columns


def test_rel_reach_matches_era_baseline_of_reach():
    n = 7
    primary = at.build_analysis_table(_features(n), _target([f"p{i}" for i in range(n)]))
    reach_vals = [10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0]
    reach = pd.DataFrame(
        {
            "post_id": [f"p{i}" for i in range(n)],
            "reach": reach_vals,
            "saved": 0, "shares": 0, "views": 0, "total_interactions": 0,
        }
    )
    subset = at.build_reach_subset(primary, reach, window=3, min_periods=2)
    expected_base = re.era_baseline(reach_vals, window=3, min_periods=2)
    np.testing.assert_allclose(subset["reach_baseline"].to_numpy(), expected_base)
    with np.errstate(invalid="ignore"):
        expected_rel = np.asarray(reach_vals) / expected_base
    np.testing.assert_allclose(subset["rel_reach"].to_numpy(), expected_rel)


# --- assemble (DB round-trip) ---------------------------------------------


@pytest.fixture
def con():
    c = db.connect(":memory:")
    db.bootstrap(c)
    yield c
    c.close()


def _seed(con, n, *, reach_from=2):
    """Seed n themed posts + one metrics snapshot each; reach NULL before reach_from."""
    for i in range(n):
        con.execute(
            "INSERT INTO posts (post_id, ig_user_id, media_type, media_product_type, "
            "caption, permalink, timestamp, theme, ingested_at) "
            "VALUES (?, 'u1', 'IMAGE', 'FEED', ?, 'http://x', ?, 'Show/Tour', ?)",
            [f"p{i}", f"caption {i}", BASE_TS + timedelta(days=i), INGESTED],
        )
        reach = None if i < reach_from else 100 + i
        con.execute(
            "INSERT INTO post_metrics (post_id, collected_at, like_count, comments_count, "
            "reach, saved, shares, views, total_interactions) VALUES (?, ?, ?, ?, ?, 1, 1, 1, 1)",
            [f"p{i}", INGESTED, 10 + i, i, reach],
        )


def test_assemble_writes_both_tables_with_expected_rows(con):
    _seed(con, 8, reach_from=2)  # 8 posts, 6 with reach
    tables = at.assemble(con=con, settings=SETTINGS, window=3, min_periods=2)

    primary = con.execute(f"SELECT * FROM {at.PRIMARY_TABLE}").df()
    reach = con.execute(f"SELECT * FROM {at.REACH_TABLE}").df()
    assert len(primary) == 8
    assert len(reach) == 6  # only reach-available posts
    assert set(at._TARGET_COLUMNS) <= set(primary.columns)
    assert {"reach", "rel_reach", "rel_reach_log2"} <= set(reach.columns)
    # returned frames match what landed in the DB
    assert len(tables[at.PRIMARY_TABLE]) == 8 and len(tables[at.REACH_TABLE]) == 6


def test_assemble_is_idempotent_rebuild(con):
    _seed(con, 8, reach_from=2)
    at.assemble(con=con, settings=SETTINGS, window=3, min_periods=2)
    at.assemble(con=con, settings=SETTINGS, window=3, min_periods=2)  # no duplicate-table error
    assert con.execute(f"SELECT count(*) FROM {at.PRIMARY_TABLE}").fetchone()[0] == 8
