"""Audience demographics analysis — pure share transforms + a DB round-trip."""

from __future__ import annotations

from datetime import date, datetime, timezone

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from insta_analytics.analysis import demographics as demo
from insta_analytics.config import Settings
from insta_analytics.storage import db

SETTINGS = Settings(
    meta_app_id="111", meta_app_secret="shh", ig_access_token="LONG", ig_user_id="u1"
)


def _long(rows, snapshot=date(2026, 7, 11)):
    """rows = list of (breakdown, dimension, value) → long demographics frame."""
    bd, dim, val = zip(*rows)
    return pd.DataFrame(
        {
            "breakdown": bd,
            "dimension": dim,
            "value": val,
            "snapshot_date": [snapshot] * len(rows),
        }
    )


# --- shares -----------------------------------------------------------------
def test_shares_sum_to_one_and_are_share_desc():
    df = _long([("gender", "M", 70), ("gender", "F", 20), ("gender", "U", 10)])
    s = demo.shares(df, "gender")
    assert s["share"].sum() == pytest.approx(1.0)
    assert s["dimension"].tolist() == ["M", "F", "U"]  # share-descending
    assert s.iloc[0]["share"] == pytest.approx(0.7)


def test_shares_respects_fixed_order():
    # age arrives scrambled; fixed order must reassemble the ramp
    df = _long([("age", "25-34", 50), ("age", "13-17", 5), ("age", "18-24", 45)])
    s = demo.shares(df, "age", order=demo.AGE_ORDER)
    assert s["dimension"].tolist() == ["13-17", "18-24", "25-34"]
    assert s["share"].sum() == pytest.approx(1.0)


def test_shares_empty_breakdown_returns_empty():
    df = _long([("gender", "M", 70)])
    assert demo.shares(df, "country").empty


# --- top_n_shares -----------------------------------------------------------
def test_top_n_aggregates_tail_into_other():
    rows = [("country", c, v) for c, v in
            [("BR", 40), ("MX", 30), ("DE", 15), ("PT", 8), ("IN", 4), ("JP", 3)]]
    df = _long(rows)
    out = demo.top_n_shares(df, "country", n=3)
    assert out["dimension"].tolist() == ["BR", "MX", "DE", "Other (3)"]
    assert out["share"].sum() == pytest.approx(1.0)
    # Other = PT+IN+JP shares
    assert out.iloc[-1]["share"] == pytest.approx((8 + 4 + 3) / 100)


def test_top_n_no_other_row_when_within_n():
    df = _long([("city", "A", 10), ("city", "B", 5)])
    out = demo.top_n_shares(df, "city", n=8)
    assert out["dimension"].tolist() == ["A", "B"]  # no Other bucket
    assert not out["dimension"].str.startswith("Other").any()


# --- figures ----------------------------------------------------------------
def test_save_figures_writes_four_pngs(tmp_path):
    rows = [("gender", "M", 70), ("gender", "F", 30)]
    rows += [("age", a, 10) for a in demo.AGE_ORDER]
    rows += [("country", c, v) for c, v in [("BR", 40), ("MX", 30), ("DE", 20), ("PT", 10)]]
    rows += [("city", c, v) for c, v in [("Mexico City", 40), ("Sao Paulo", 30)]]
    paths = demo.save_figures(_long(rows), figures_dir=tmp_path, top_n=3)
    assert len(paths) == 4
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)


# --- DB round-trip ----------------------------------------------------------
@pytest.fixture
def con():
    c = db.connect(":memory:")
    db.bootstrap(c)
    yield c
    c.close()


def test_load_demographics_takes_latest_snapshot(con):
    now = datetime(2026, 7, 11, 12, 0, tzinfo=timezone.utc)
    con.execute(
        "INSERT INTO follower_demographics "
        "(ig_user_id, snapshot_date, breakdown, dimension, value, collected_at) VALUES "
        "('u1', DATE '2026-07-01', 'gender', 'M', 10, ?), "
        "('u1', DATE '2026-07-11', 'gender', 'M', 99, ?)",
        [now, now],
    )
    out = demo.load_demographics(con=con, settings=SETTINGS)
    assert len(out) == 1
    assert out.iloc[0]["value"] == 99


def test_load_demographics_empty_table(con):
    out = demo.load_demographics(con=con, settings=SETTINGS)
    assert out.empty
