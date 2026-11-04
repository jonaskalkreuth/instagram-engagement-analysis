"""Join predictors to targets and write the analysis tables to DuckDB.

- ``analysis_posts``: one row per post, predictors + relative engagement.
- ``analysis_reach``: posts with a ``reach`` value, plus era-relative reach.

Both are rebuilt each run (CREATE OR REPLACE), not upserted.
"""

from __future__ import annotations

import argparse

import duckdb
import numpy as np
import pandas as pd

from ..config import Settings, get_settings
from ..storage import db
from . import features, relative_engagement

# Insight columns carried on the reach subset (latest snapshot per post).
_REACH_COLUMNS: tuple[str, ...] = (
    "reach",
    "saved",
    "shares",
    "views",
    "total_interactions",
)

# The target columns joined onto the predictors.
_TARGET_COLUMNS: tuple[str, ...] = (
    "engagement",
    "era_baseline",
    "rel_engagement",
    "rel_engagement_log2",
)

PRIMARY_TABLE = "analysis_posts"
REACH_TABLE = "analysis_reach"


def load_reach(
    *,
    con: duckdb.DuckDBPyConnection | None = None,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """One row per post with its latest-snapshot insight columns."""
    with db.connection(con, settings) as con:
        return con.execute(
            """
            SELECT post_id,
                   arg_max(reach, collected_at)              AS reach,
                   arg_max(saved, collected_at)              AS saved,
                   arg_max(shares, collected_at)             AS shares,
                   arg_max(views, collected_at)              AS views,
                   arg_max(total_interactions, collected_at) AS total_interactions
            FROM post_metrics
            GROUP BY post_id
            """
        ).df()


def build_analysis_table(features_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    """Join predictors to the engagement target; raises if any post lacks a target."""
    target = target_df[["post_id", *_TARGET_COLUMNS]]
    out = features_df.merge(target, on="post_id", how="inner")
    if len(out) != len(features_df):
        raise ValueError(
            f"target join dropped rows: {len(features_df)} predictors → {len(out)} joined "
            "(a post is missing its engagement target)."
        )
    return out.sort_values("timestamp", kind="stable").reset_index(drop=True)


def build_reach_subset(
    primary_df: pd.DataFrame,
    reach_df: pd.DataFrame,
    *,
    window: int = relative_engagement.DEFAULT_WINDOW,
    min_periods: int = relative_engagement.DEFAULT_MIN_PERIODS,
) -> pd.DataFrame:
    """Keep posts with reach and add ``reach_baseline``, ``rel_reach``, ``rel_reach_log2``.

    A NaN or zero baseline gives NaN relative values.
    """
    merged = primary_df.merge(reach_df[["post_id", *_REACH_COLUMNS]], on="post_id", how="left")
    subset = merged[merged["reach"].notna()].copy()
    subset = subset.sort_values("timestamp", kind="stable").reset_index(drop=True)

    baseline = relative_engagement.era_baseline(
        subset["reach"].to_numpy(dtype=float), window=window, min_periods=min_periods
    )
    subset["reach_baseline"] = baseline
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = subset["reach"] / subset["reach_baseline"]
    ratio = ratio.where(subset["reach_baseline"] > 0)  # 0/NaN baseline → NaN
    subset["rel_reach"] = ratio
    subset["rel_reach_log2"] = np.log2(ratio.where(ratio > 0))
    return subset


def write_table(con: duckdb.DuckDBPyConnection, name: str, df: pd.DataFrame) -> None:
    """Replace table ``name`` with ``df``."""
    con.register("_write_df", df)
    try:
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _write_df")
    finally:
        con.unregister("_write_df")


def assemble(
    *,
    require_complete: bool = True,
    window: int = relative_engagement.DEFAULT_WINDOW,
    min_periods: int = relative_engagement.DEFAULT_MIN_PERIODS,
    con: duckdb.DuckDBPyConnection | None = None,
    settings: Settings | None = None,
) -> dict[str, pd.DataFrame]:
    """Build both analysis tables, write them to the DB and return them by name."""
    settings = settings or get_settings()
    with db.connection(con, settings) as con:
        feats = features.compute(require_complete=require_complete, con=con, settings=settings)
        target = relative_engagement.compute(
            window=window, min_periods=min_periods, con=con, settings=settings
        )
        reach = load_reach(con=con, settings=settings)

        primary = build_analysis_table(feats, target)
        reach_subset = build_reach_subset(
            primary, reach, window=window, min_periods=min_periods
        )

        write_table(con, PRIMARY_TABLE, primary)
        write_table(con, REACH_TABLE, reach_subset)
    return {PRIMARY_TABLE: primary, REACH_TABLE: reach_subset}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analysis-table assembly: build + write analysis_posts / analysis_reach."
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Build over only the labeled posts instead of raising on holes.",
    )
    parser.add_argument("--window", type=int, default=relative_engagement.DEFAULT_WINDOW)
    parser.add_argument(
        "--min-periods", type=int, default=relative_engagement.DEFAULT_MIN_PERIODS
    )
    args = parser.parse_args()

    tables = assemble(
        require_complete=not args.allow_incomplete,
        window=args.window,
        min_periods=args.min_periods,
    )
    primary, reach = tables[PRIMARY_TABLE], tables[REACH_TABLE]
    print(f"{PRIMARY_TABLE}: {primary.shape[0]} rows × {primary.shape[1]} cols")
    print(f"  rel_engagement valid: {int(primary['rel_engagement'].notna().sum())}/{len(primary)}")
    print(f"{REACH_TABLE}: {reach.shape[0]} rows × {reach.shape[1]} cols "
          f"(reach-available subset of {len(primary)})")
    print(f"  rel_reach valid:      {int(reach['rel_reach'].notna().sum())}/{len(reach)}")
    print("\nengagement target (analysis_posts):")
    print(primary["rel_engagement"].describe().to_string())
    print("\nreach target (analysis_reach):")
    print(reach["rel_reach"].describe().to_string())


if __name__ == "__main__":
    main()
