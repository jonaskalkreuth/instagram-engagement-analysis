"""Seasonality & drift — pure functions + figure smoke."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insta_analytics.analysis import trends


def _posts(rows):
    """rows = list of (timestamp, theme, month, log2) → analysis_posts-shaped frame."""
    ts, theme, month, log2 = zip(*rows)
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(list(ts), utc=True),
            "theme": theme,
            "month": month,
            "rel_engagement_log2": log2,
        }
    )


# --- with_year --------------------------------------------------------------
@pytest.mark.parametrize("zone", ["UTC", "America/New_York"])
def test_with_year_uses_berlin_year_whatever_the_input_zone(zone):
    # 23:30 UTC on 31 Dec 2023 is already 2024 in Berlin.
    df = _posts([("2023-12-31 23:30", "a", 1, 0.0)])
    df["timestamp"] = df["timestamp"].dt.tz_convert(zone)
    assert trends.with_year(df)["year"].tolist() == [2024]


# --- seasonality ------------------------------------------------------------
def test_seasonality_is_month_ordered_not_ratio_sorted():
    # Dec (month 12) is the strongest, but rows must come back Jan→Dec.
    rows = [(f"2022-{m:02d}-01", "a", m, 0.0) for m in range(1, 13) for _ in range(5)]
    rows += [("2022-12-15", "a", 12, 2.0)] * 5  # spike December
    eff = trends.seasonality(_posts(rows), n_boot=100, min_n=1)
    assert eff["month"].tolist() == list(range(1, 13))
    assert eff["month_name"].tolist()[0] == "Jan"
    assert eff["month_name"].tolist()[-1] == "Dec"


# --- theme_mix --------------------------------------------------------------
def test_theme_mix_shares_sum_to_one_per_year():
    rows = (
        [("2021-01-01", "a", 1, 0.0)] * 3
        + [("2021-02-01", "b", 2, 0.0)] * 1
        + [("2022-01-01", "a", 1, 0.0)] * 2
        + [("2022-02-01", "b", 2, 0.0)] * 2
    )
    share = trends.theme_mix(_posts(rows))
    assert share.loc[2021].sum() == pytest.approx(1.0)
    assert share.loc[2021, "a"] == pytest.approx(0.75)
    assert share.loc[2022, "a"] == pytest.approx(0.5)


def test_theme_mix_absent_theme_is_zero_share():
    rows = [("2021-01-01", "a", 1, 0.0)] * 2 + [("2022-01-01", "b", 1, 0.0)] * 2
    share = trends.theme_mix(_posts(rows))
    assert share.loc[2021, "b"] == pytest.approx(0.0)  # b absent in 2021
    assert share.loc[2022, "a"] == pytest.approx(0.0)


# --- cadence_per_year -------------------------------------------------------
def test_cadence_counts_per_year_ascending():
    rows = [("2021-01-01", "a", 1, 0.0)] * 3 + [("2023-01-01", "a", 1, 0.0)] * 5
    cad = trends.cadence_per_year(_posts(rows))
    assert cad["year"].tolist() == [2021, 2023]
    assert cad["posts"].tolist() == [3, 5]


# --- theme_era_effect -------------------------------------------------------
def test_theme_era_splits_at_median_and_signs_delta():
    # theme "a": under-performs early (-1), over-performs late (+1) → positive delta.
    rows = (
        [("2020-01-01", "a", 1, -1.0)] * 6
        + [("2025-01-01", "a", 1, 1.0)] * 6
    )
    era = trends.theme_era_effect(_posts(rows), n_boot=100, min_n=1)
    a = era.loc[era["theme"] == "a"].iloc[0]
    assert a["early_ratio"] == pytest.approx(0.5)   # 2**-1
    assert a["late_ratio"] == pytest.approx(2.0)    # 2**+1
    assert a["delta"] == pytest.approx(1.5)


# --- figures ----------------------------------------------------------------
def test_save_figures_writes_all_pngs(tmp_path):
    rng = np.random.default_rng(0)
    n = 60
    years = rng.choice([2021, 2022, 2023, 2024], size=n)
    months = rng.integers(1, 13, size=n)
    ts = [f"{y}-{m:02d}-01" for y, m in zip(years, months)]
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(ts, utc=True),
            "theme": rng.choice(["a", "b", "c"], size=n),
            "month": months,
            "rel_engagement_log2": rng.normal(0, 1, size=n),
        }
    )
    paths = trends.save_figures(df, figures_dir=tmp_path, n_boot=100)
    assert len(paths) == 4
    assert all(p.exists() for p in paths)
