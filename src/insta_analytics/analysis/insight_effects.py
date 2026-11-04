"""Saves and shares by theme and format.

Two metric families: era-relative saves and shares (via :func:`eda.group_effect`),
and raw rates per reach plus ``comment_share`` (via :func:`eda.group_median`).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ..config import Settings, get_settings
from ..storage import db
from ..transform import relative_engagement as re
from . import eda

REL_TARGETS: dict[str, str] = {
    "rel_saved_log2": "saves",
    "rel_shares_log2": "shares",
}
RATE_COLS: tuple[str, ...] = (
    "save_rate",
    "share_rate",
    "interaction_rate",
    "comment_share",
)
CUTS: tuple[str, ...] = ("theme", "format")


def load_like_comment_split(
    *,
    con: duckdb.DuckDBPyConnection | None = None,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Latest-snapshot likes and comments per post, kept separate for ``comment_share``."""
    with db.connection(con, settings) as con:
        return con.execute(
            """
            SELECT post_id,
                   arg_max(like_count, collected_at)     AS like_count,
                   arg_max(comments_count, collected_at) AS comments_count
            FROM post_metrics
            GROUP BY post_id
            """
        ).df()


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """``num / den``, NaN where ``den`` is not positive."""
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    return out.where(den > 0)


def _era_relative(
    frame: pd.DataFrame, col: str, *, window: int, min_periods: int
) -> tuple[pd.Series, pd.Series]:
    """(ratio, log2) of ``col`` against its era baseline; ``frame`` must be time-ordered."""
    vals = frame[col].to_numpy(dtype=float)  # float path avoids masked-array log2 warnings
    base = re.era_baseline(vals, window=window, min_periods=min_periods)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(base > 0, vals / base, np.nan)
        log2 = np.log2(np.where(ratio > 0, ratio, np.nan))
    return pd.Series(ratio, index=frame.index), pd.Series(log2, index=frame.index)


def derive_insight_metrics(
    reach_df: pd.DataFrame,
    split_df: pd.DataFrame,
    *,
    window: int = re.DEFAULT_WINDOW,
    min_periods: int = re.DEFAULT_MIN_PERIODS,
) -> pd.DataFrame:
    """Add era-relative saves and shares, and per-reach rates, to the reach frame."""
    out = reach_df.merge(
        split_df[["post_id", "like_count", "comments_count"]], on="post_id", how="left"
    )
    out = out.sort_values("timestamp", kind="stable").reset_index(drop=True)

    out["rel_saved"], out["rel_saved_log2"] = _era_relative(
        out, "saved", window=window, min_periods=min_periods
    )
    out["rel_shares"], out["rel_shares_log2"] = _era_relative(
        out, "shares", window=window, min_periods=min_periods
    )

    out["save_rate"] = _safe_div(out["saved"], out["reach"])
    out["share_rate"] = _safe_div(out["shares"], out["reach"])
    out["interaction_rate"] = _safe_div(out["total_interactions"], out["reach"])
    out["comment_share"] = _safe_div(
        out["comments_count"], out["like_count"] + out["comments_count"]
    )
    return out


def plot_rate(med: pd.DataFrame, by: str, title: str) -> plt.Figure:
    """Horizontal bars of median rate per group with CI whiskers."""
    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.5 * len(med) + 1)))
    e = med.iloc[::-1].reset_index(drop=True)  # strongest at top
    ypos = np.arange(len(e))
    err = np.vstack([e["median"] - e["median_lo"], e["median_hi"] - e["median"]])
    err = np.where(np.isnan(err), 0.0, err)  # small groups: no whisker
    ax.barh(ypos, e["median"], color="#5851db", alpha=0.85)
    ax.errorbar(e["median"], ypos, xerr=err, fmt="none", ecolor="#333", capsize=3)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{lvl}  (n={n})" for lvl, n in zip(e[by], e["n"])])
    ax.set_xlabel("median rate")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def save_figures(
    df: pd.DataFrame,
    *,
    figures_dir: Path,
    **effect_kw,
) -> list[Path]:
    """Write the saves and shares figures to ``figures_dir``."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    specs: list[tuple[str, plt.Figure]] = []
    for target, label in REL_TARGETS.items():
        for cut in CUTS:
            eff = eda.group_effect(df, cut, target=target, **effect_kw)
            specs.append(
                (f"insight_{label}_{cut}.png",
                 eda.plot_group_effect(eff, cut, f"Relative {label} by {cut}"))
            )
    for rate in ("save_rate", "share_rate"):
        med = eda.group_median(df, "theme", target=rate, **effect_kw)
        specs.append((f"insight_{rate}_theme.png", plot_rate(med, "theme", f"{rate} by theme")))

    paths = []
    for name, fig in specs:
        p = figures_dir / name
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(p)
    return paths


def report(
    *,
    figures: bool = True,
    n_boot: int = eda.DEFAULT_N_BOOT,
    seed: int = eda.DEFAULT_SEED,
    con: duckdb.DuckDBPyConnection | None = None,
    settings: Settings | None = None,
) -> None:
    """Print the saves and shares tables and write figures."""
    settings = settings or get_settings()
    with db.connection(con, settings) as con:
        reach = eda.load_analysis_reach(con=con, settings=settings)
        split = load_like_comment_split(con=con, settings=settings)

    df = derive_insight_metrics(reach, split)
    eff_kw = dict(n_boot=n_boot, seed=seed)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")

    print(f"analysis_reach: {len(df)} posts with reach\n")
    print("=== Saves & shares (era-relative, ratio vs baseline) ===")
    for target, label in REL_TARGETS.items():
        for cut in CUTS:
            eff = eda.group_effect(df, cut, target=target, **eff_kw)
            print(f"\n-- {label} by {cut} --")
            print(eff.to_string(index=False))
            sig = eda.clears_ci(eff)
            if not sig.empty:
                print(f"   CI clears 1.0: {', '.join(str(v) for v in sig[cut])}")

    print("\n=== Rates (raw median) ===")
    for rate in RATE_COLS:
        for cut in CUTS:
            med = eda.group_median(df, cut, target=rate, **eff_kw)
            print(f"\n-- {rate} by {cut} --")
            print(med.to_string(index=False))

    if figures:
        paths = save_figures(df, figures_dir=settings.figures_dir, **eff_kw)
        print("\nfigures:")
        for p in paths:
            print(f"  → {p}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Saves & shares deep-dive: era-relative + rates by theme/format."
    )
    parser.add_argument("--no-figures", action="store_true", help="Skip writing PNGs.")
    parser.add_argument("--n-boot", type=int, default=eda.DEFAULT_N_BOOT)
    parser.add_argument("--seed", type=int, default=eda.DEFAULT_SEED)
    args = parser.parse_args()
    report(figures=not args.no_figures, n_boot=args.n_boot, seed=args.seed)


if __name__ == "__main__":
    main()
