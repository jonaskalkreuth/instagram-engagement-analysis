"""Read the DuckDB file read-only and precompute every dashboard panel into one bundle.

The bootstrap uses fewer resamples than the reports so the first load stays fast.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import pandas as pd
import pycountry

from ..analysis import demographics, eda, insight_effects, trends
from ..config import db_path_for_source
from ..transform.analysis_table import PRIMARY_TABLE, REACH_TABLE

YearRange = tuple[int, int]

# Synthetic by default; set ``INSTA_DATA_SOURCE=real`` for the real DB.
DEFAULT_DB_PATH = db_path_for_source()

# Lighter than the reports' 10k: interactive first-load speed over exact repro.
N_BOOT = 2_000


def _read_tables(
    db_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return ``(posts, reach, demographics, like/comment split)`` from ``db_path``, read-only."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        posts = con.execute(f"SELECT * FROM {PRIMARY_TABLE}").df()
        reach = con.execute(f"SELECT * FROM {REACH_TABLE}").df()
        demo = con.execute(f"SELECT * FROM {demographics.TABLE}").df()
        split = con.execute(
            """
            SELECT post_id,
                   arg_max(like_count, collected_at)     AS like_count,
                   arg_max(comments_count, collected_at) AS comments_count
            FROM post_metrics
            GROUP BY post_id
            """
        ).df()
    finally:
        con.close()
    return posts, reach, demo, split


def country_geo(country_shares: pd.DataFrame) -> pd.DataFrame:
    """Add ``country_name`` and ISO-3 ``iso_alpha`` to ISO-2 country shares; unknown codes get None."""
    if country_shares.empty:
        return country_shares.assign(country_name=[], iso_alpha=[])
    names, iso3 = [], []
    for code in country_shares["dimension"].astype(str):
        match = pycountry.countries.get(alpha_2=code)
        names.append(match.name if match else code)
        iso3.append(match.alpha_3 if match else None)
    return country_shares.assign(country_name=names, iso_alpha=iso3)


def _timing_pivot(posts: pd.DataFrame, target: str = eda.ENGAGEMENT_TARGET) -> pd.DataFrame:
    """Day-of-week (0–6 index) × hour median relative-engagement ratio matrix."""
    return (
        posts.dropna(subset=[target])
        .assign(ratio=lambda d: 2.0 ** d[target])
        .pivot_table(index="dow", columns="hour", values="ratio", aggfunc="median")
        .reindex(index=range(7))
    )


def year_bounds(db_path: str | Path = DEFAULT_DB_PATH) -> YearRange:
    """``(min, max)`` local post year in the DB — drives the sidebar year slider."""
    posts, *_ = _read_tables(db_path)
    years = trends.with_year(posts)["year"]
    return int(years.min()), int(years.max())


def _filter_years(df: pd.DataFrame, year_range: YearRange | None) -> pd.DataFrame:
    """Keep posts whose local year is within ``year_range`` (inclusive); era baselines are not recomputed."""
    if year_range is None:
        return df
    lo, hi = year_range
    years = trends.with_year(df)["year"]
    return df[(years >= lo) & (years <= hi)].reset_index(drop=True)


def _followers(demo: pd.DataFrame) -> int | None:
    """Approx follower base = sum of the gender breakdown (empty table → None)."""
    g = demo[demo["breakdown"] == "gender"]
    return int(g["value"].sum()) if not g.empty else None


@dataclass
class DashboardData:
    """Every panel's DataFrame, precomputed once. Fields map 1:1 to the charts."""

    # EDA — relative engagement + reach twins, timing, continuous
    eng_format: pd.DataFrame
    eng_theme: pd.DataFrame
    reach_format: pd.DataFrame
    reach_theme: pd.DataFrame
    timing: pd.DataFrame
    spearman: pd.DataFrame
    # Saves & shares — era-relative effects + raw rates, by theme & format
    saves_shares: dict[str, dict[str, pd.DataFrame]]
    rates: dict[str, dict[str, pd.DataFrame]]
    # Theme × format + sweet-spots
    cell_ratio: pd.DataFrame
    cell_n: pd.DataFrame
    cell_flags: pd.DataFrame
    caption_sweet: pd.DataFrame
    hashtag_sweet: pd.DataFrame
    # Seasonality & drift
    seasonality: pd.DataFrame
    theme_mix: pd.DataFrame
    cadence: pd.DataFrame
    theme_era: pd.DataFrame
    # Audience — shares only, publish-safe
    gender: pd.DataFrame
    age: pd.DataFrame
    country: pd.DataFrame
    city: pd.DataFrame
    # KPI tiles
    kpis: dict[str, object] = field(default_factory=dict)


def compute_dashboard_data(
    db_path: str | Path = DEFAULT_DB_PATH, *, year_range: YearRange | None = None
) -> DashboardData:
    """Load the tables and compute every panel; ``year_range`` filters posts but not demographics."""
    posts, reach, demo, split = _read_tables(db_path)
    posts = _filter_years(posts, year_range)
    reach = _filter_years(reach, year_range)
    demo = demographics.latest_snapshot(demo)  # current audience only
    boot = dict(n_boot=N_BOOT)

    # --- EDA ---
    eng_format = eda.group_effect(posts, "format", **boot)
    eng_theme = eda.group_effect(posts, "theme", **boot)
    reach_format = eda.group_effect(reach, "format", target=eda.REACH_TARGET, **boot)
    reach_theme = eda.group_effect(reach, "theme", target=eda.REACH_TARGET, **boot)
    spearman = eda.spearman_table(posts, **boot)
    timing = _timing_pivot(posts)

    # --- Saves & shares ---
    ins = insight_effects.derive_insight_metrics(reach, split)
    saves_shares = {
        label: {cut: eda.group_effect(ins, cut, target=target, **boot)
                for cut in insight_effects.CUTS}
        for target, label in insight_effects.REL_TARGETS.items()
    }
    rates = {
        rate: {cut: eda.group_median(ins, cut, target=rate, **boot)
               for cut in insight_effects.CUTS}
        for rate in insight_effects.RATE_COLS
    }

    # --- Theme × format + sweet-spots ---
    cell_ratio, cell_n = eda.cell_medians(posts)
    cell_flags = eda.beats_both_marginals(posts)
    caption_sweet = eda.sweet_spot(posts, "caption_chars", **boot)
    hashtag_sweet = eda.sweet_spot(posts, "hashtag_count", **boot)

    # --- Seasonality & drift ---
    seasonality = trends.seasonality(posts, **boot)
    theme_mix = trends.theme_mix(posts)
    cadence = trends.cadence_per_year(posts)
    theme_era = trends.theme_era_effect(posts, **boot)

    # --- Audience — shares only ---
    gender = demographics.shares(demo, "gender", order=demographics.GENDER_ORDER)
    age = demographics.shares(demo, "age", order=demographics.AGE_ORDER)
    country = country_geo(demographics.shares(demo, "country"))
    city = demographics.top_n_shares(demo, "city")

    kpis = {
        "n_posts": len(posts),
        "n_reach": len(reach),
        "followers": _followers(demo),
        "demo_snapshot_date": (None if demo.empty else demo["snapshot_date"].max()),
    }

    return DashboardData(
        eng_format=eng_format,
        eng_theme=eng_theme,
        reach_format=reach_format,
        reach_theme=reach_theme,
        timing=timing,
        spearman=spearman,
        saves_shares=saves_shares,
        rates=rates,
        cell_ratio=cell_ratio,
        cell_n=cell_n,
        cell_flags=cell_flags,
        caption_sweet=caption_sweet,
        hashtag_sweet=hashtag_sweet,
        seasonality=seasonality,
        theme_mix=theme_mix,
        cadence=cadence,
        theme_era=theme_era,
        gender=gender,
        age=age,
        country=country,
        city=city,
        kpis=kpis,
    )


def load(
    db_path: str | Path = DEFAULT_DB_PATH, *, year_range: YearRange | None = None
) -> DashboardData:
    """Cached :func:`compute_dashboard_data`; Streamlit is imported lazily."""
    import streamlit as st

    @st.cache_data(show_spinner="Crunching the analysis…")
    def _cached(path: str, yr: YearRange | None) -> DashboardData:
        return compute_dashboard_data(path, year_range=yr)

    return _cached(str(db_path), year_range)
