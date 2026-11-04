"""Tests for the dashboard data layer.

Builds a tiny throwaway DuckDB (schema-shaped like the analysis tables) and drives
:func:`compute_dashboard_data` end-to-end — no real DB, no secrets, no Streamlit.
Asserts the bundle carries every panel field with sane shapes, and that the
country geo mapping resolves ISO-2 → name/ISO-3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from insta_analytics.dashboard import data


def test_bundle_has_every_panel(db_path):
    b = data.compute_dashboard_data(db_path)
    # EDA
    assert set(b.eng_theme["theme"]) == {"Live/Performance", "Show/Tour"}
    assert {"ratio", "ratio_lo", "ratio_hi", "n"} <= set(b.eng_format.columns)
    assert list(b.reach_theme.columns) == list(b.eng_theme.columns)
    assert b.timing.index.tolist() == list(range(7))  # dow rows
    assert len(b.spearman) == len(data.eda.CONTINUOUS_PREDICTORS)
    # Saves & shares
    assert set(b.saves_shares) == {"saves", "shares"}
    assert set(b.saves_shares["saves"]) == {"theme", "format"}
    assert set(b.rates) == set(data.insight_effects.RATE_COLS)
    # Theme × format
    assert b.cell_ratio.shape == b.cell_n.shape == b.cell_flags.shape
    assert b.cell_flags.dtypes.eq(bool).all()
    assert not b.caption_sweet.empty and not b.hashtag_sweet.empty
    # Trends
    assert len(b.seasonality) == 12  # Jan..Dec
    assert np.allclose(b.theme_mix.sum(axis=1), 1.0)  # each year's mix sums to 1
    assert b.cadence["posts"].sum() == 16
    assert "delta" in b.theme_era.columns


def test_kpis(db_path):
    b = data.compute_dashboard_data(db_path)
    assert b.kpis["n_posts"] == 16
    assert b.kpis["n_reach"] == 16
    assert b.kpis["followers"] == 1000  # gender total 750+138+112
    assert b.kpis["demo_snapshot_date"] == pd.Timestamp("2026-07-11")


def test_demographics_shares_and_geo(db_path):
    b = data.compute_dashboard_data(db_path)
    assert b.gender["dimension"].tolist() == ["F", "M", "U"]  # GENDER_ORDER
    assert np.isclose(b.gender["share"].sum(), 1.0)
    assert b.age["dimension"].tolist() == ["25-34", "35-44"]  # AGE_ORDER subset
    # country geo: ISO-2 → name + ISO-3
    br = b.country[b.country["dimension"] == "BR"].iloc[0]
    assert br["country_name"] == "Brazil" and br["iso_alpha"] == "BRA"
    assert {"country_name", "iso_alpha"} <= set(b.country.columns)


def test_country_geo_unknown_code_passes_through():
    df = pd.DataFrame({"dimension": ["ZZ", "Other (3)"], "share": [0.6, 0.4]})
    out = data.country_geo(df)
    assert out["country_name"].tolist() == ["ZZ", "Other (3)"]
    assert out["iso_alpha"].isna().all()


def test_year_filter_subsets_posts(db_path):
    # Fixture posts span 2021 → 2022 (16 posts, 30-day steps from 2021-01-01).
    lo, hi = data.year_bounds(db_path)
    full = data.compute_dashboard_data(db_path)
    one_year = data.compute_dashboard_data(db_path, year_range=(lo, lo))
    assert (lo, hi) == (2021, 2022)
    assert one_year.kpis["n_posts"] < full.kpis["n_posts"]  # 2021-only is fewer
    assert one_year.kpis["n_posts"] > 0
    # Demographics is a current snapshot — the year filter must not touch it.
    assert one_year.kpis["followers"] == full.kpis["followers"]
