"""Engagement relative to an era baseline (see ADR 0001).

- Engagement is likes + comments.
- The baseline is a centred, leave-one-out rolling median over a fixed number of
  posts, not days, because posting cadence varied about 20× across years.
- The output is the ratio to the baseline, plus its log2 for the model.
"""

from __future__ import annotations

import argparse

import duckdb
import numpy as np
import pandas as pd

from ..config import Settings
from ..storage import db

DEFAULT_WINDOW = 21  # neighborhood size in posts (anchor + 10 each side)
DEFAULT_MIN_PERIODS = 5  # min neighbors (self excluded) for a valid baseline
SENSITIVITY_WINDOWS = (11, 21, 31, 51)


def load_engagement(
    *,
    con: duckdb.DuckDBPyConnection | None = None,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """One row per post with latest-snapshot likes + comments, ordered by time."""
    with db.connection(con, settings) as con:
        return con.execute(
            """
            SELECT p.post_id,
                   p.timestamp,
                   m.like_count + m.comments_count AS engagement
            FROM posts p
            JOIN (
                SELECT post_id,
                       arg_max(like_count, collected_at)     AS like_count,
                       arg_max(comments_count, collected_at) AS comments_count
                FROM post_metrics
                GROUP BY post_id
            ) m USING (post_id)
            ORDER BY p.timestamp
            """
        ).df()


def era_baseline(
    values, *, window: int = DEFAULT_WINDOW, min_periods: int = DEFAULT_MIN_PERIODS
) -> np.ndarray:
    """Centred rolling median of time-ordered ``values``, excluding each point itself.

    Positions with fewer than ``min_periods`` non-NaN neighbours get NaN.
    """
    if window < 2:
        raise ValueError("window must span at least 2 posts")
    k = window // 2
    vals = np.asarray(values, dtype=float)
    n = len(vals)
    out = np.full(n, np.nan)
    for i in range(n):
        lo, hi = max(0, i - k), min(n, i + k + 1)
        neigh = np.concatenate([vals[lo:i], vals[i + 1 : hi]])
        neigh = neigh[~np.isnan(neigh)]
        if len(neigh) >= min_periods:
            out[i] = np.median(neigh)
    return out


def relative_engagement(
    df: pd.DataFrame,
    *,
    window: int = DEFAULT_WINDOW,
    min_periods: int = DEFAULT_MIN_PERIODS,
) -> pd.DataFrame:
    """Add ``era_baseline``, ``rel_engagement`` and ``rel_engagement_log2``.

    A NaN or zero baseline gives NaN relative values.
    """
    out = df.sort_values("timestamp", kind="stable").reset_index(drop=True).copy()
    out["era_baseline"] = era_baseline(
        out["engagement"].to_numpy(dtype=float), window=window, min_periods=min_periods
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = out["engagement"] / out["era_baseline"]
    ratio = ratio.where(out["era_baseline"] > 0)  # 0/NaN baseline → NaN
    out["rel_engagement"] = ratio
    out["rel_engagement_log2"] = np.log2(ratio.where(ratio > 0))
    return out


def window_sensitivity(
    df: pd.DataFrame,
    *,
    windows=SENSITIVITY_WINDOWS,
    min_periods: int = DEFAULT_MIN_PERIODS,
    reference: int = DEFAULT_WINDOW,
) -> pd.DataFrame:
    """Spearman correlation of each window's ``rel_engagement`` with the reference window."""
    ref = relative_engagement(df, window=reference, min_periods=min_periods)["rel_engagement"]
    rows = []
    for w in windows:
        r = relative_engagement(df, window=w, min_periods=min_periods)["rel_engagement"]
        mask = ref.notna() & r.notna()
        rho = ref[mask].corr(r[mask], method="spearman") if mask.sum() > 1 else float("nan")
        rows.append({"window": w, "n_valid": int(r.notna().sum()), "spearman_vs_ref": rho})
    return pd.DataFrame(rows)


def compute(
    *,
    window: int = DEFAULT_WINDOW,
    min_periods: int = DEFAULT_MIN_PERIODS,
    con: duckdb.DuckDBPyConnection | None = None,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Load engagement from the DB and attach the relative-engagement columns."""
    return relative_engagement(
        load_engagement(con=con, settings=settings),
        window=window,
        min_periods=min_periods,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Era-baseline relative engagement: summary + window sensitivity."
    )
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--min-periods", type=int, default=DEFAULT_MIN_PERIODS)
    args = parser.parse_args()

    df = compute(window=args.window, min_periods=args.min_periods)
    rel = df["rel_engagement"]
    print(f"posts: {len(df)}  |  rel_engagement valid: {int(rel.notna().sum())}")
    print(rel.describe().to_string())
    print("\ntop 5 over-performers (rel × era):")
    print(df.nlargest(5, "rel_engagement")[["timestamp", "engagement", "era_baseline", "rel_engagement"]].to_string(index=False))
    print("\nwindow sensitivity (Spearman vs default):")
    print(window_sensitivity(df, min_periods=args.min_periods).to_string(index=False))


if __name__ == "__main__":
    main()
