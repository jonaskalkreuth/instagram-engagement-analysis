"""Seasonality, theme-mix drift, early-vs-late theme effects and posts per year.

Relative engagement has a yearly median near 1 by construction, so drift shows
in the theme mix rather than the overall level.
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
from ..transform.features import LOCAL_TZ
from . import eda

MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def with_year(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of ``df`` with a ``year`` column in :data:`LOCAL_TZ`, matching ``month``."""
    out = df.copy()
    out["year"] = pd.to_datetime(out["timestamp"]).dt.tz_convert(LOCAL_TZ).dt.year
    return out


def seasonality(
    posts: pd.DataFrame,
    *,
    target: str = eda.ENGAGEMENT_TARGET,
    min_n: int = eda.DEFAULT_MIN_N,
    n_boot: int = eda.DEFAULT_N_BOOT,
    seed: int = eda.DEFAULT_SEED,
) -> pd.DataFrame:
    """:func:`eda.group_effect` by month, in Jan→Dec order, with a ``month_name`` column."""
    eff = eda.group_effect(posts, "month", target=target, min_n=min_n,
                           n_boot=n_boot, seed=seed)
    eff = eff.sort_values("month", kind="stable").reset_index(drop=True)
    eff["month_name"] = [MONTH_ABBR[m - 1] for m in eff["month"]]
    return eff


def theme_mix(posts: pd.DataFrame) -> pd.DataFrame:
    """Year × theme share matrix; each row sums to 1.0."""
    d = with_year(posts)
    counts = pd.crosstab(d["year"], d["theme"])
    share = counts.div(counts.sum(axis=1), axis=0)
    return share.sort_index()


def cadence_per_year(posts: pd.DataFrame) -> pd.DataFrame:
    """Posts-per-year counts (``year``, ``posts``), ascending by year."""
    d = with_year(posts)
    out = d.groupby("year", observed=True).size().reset_index(name="posts")
    return out.sort_values("year", kind="stable").reset_index(drop=True)


def theme_era_effect(
    posts: pd.DataFrame,
    *,
    target: str = eda.ENGAGEMENT_TARGET,
    min_n: int = eda.DEFAULT_MIN_N,
    n_boot: int = eda.DEFAULT_N_BOOT,
    seed: int = eda.DEFAULT_SEED,
) -> pd.DataFrame:
    """Theme ratios before and after the median timestamp, with ``delta`` = late − early."""
    d = posts.dropna(subset=[target]).copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    cut = d["timestamp"].median()
    early = d[d["timestamp"] <= cut]
    late = d[d["timestamp"] > cut]

    def _by_theme(sub: pd.DataFrame, tag: str) -> pd.DataFrame:
        eff = eda.group_effect(sub, "theme", target=target, min_n=min_n,
                               n_boot=n_boot, seed=seed)
        return eff[["theme", "n", "ratio"]].rename(
            columns={"n": f"n_{tag}", "ratio": f"{tag}_ratio"})

    merged = _by_theme(early, "early").merge(
        _by_theme(late, "late"), on="theme", how="outer")
    merged["delta"] = merged["late_ratio"] - merged["early_ratio"]
    return merged.sort_values("delta", ascending=False, kind="stable").reset_index(drop=True)


def plot_seasonality(eff: pd.DataFrame, title: str = "Seasonality (ratio vs era)") -> plt.Figure:
    """Month bars of median ratio with bootstrap-CI whiskers; 1.0 = era baseline."""
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(eff))
    err = np.vstack([eff["ratio"] - eff["ratio_lo"], eff["ratio_hi"] - eff["ratio"]])
    err = np.where(np.isnan(err), 0.0, err)  # small months: no whisker
    ax.bar(x, eff["ratio"], color="#c13584", alpha=0.85)
    ax.errorbar(x, eff["ratio"], yerr=err, fmt="none", ecolor="#333", capsize=3)
    ax.axhline(1.0, color="#666", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n(n={n})" for m, n in zip(eff["month_name"], eff["n"])])
    ax.set_ylabel("median engagement ÷ era baseline")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_theme_mix(share: pd.DataFrame, title: str = "Theme mix per year (share)") -> plt.Figure:
    """Stacked area of each theme's posting share over the years."""
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(share.shape[1])]
    ax.stackplot(share.index, *[share[c] for c in share.columns],
                 labels=list(share.columns), colors=colors, alpha=0.9)
    ax.set_xlim(share.index.min(), share.index.max())
    ax.set_ylim(0, 1)
    ax.set_xlabel("year")
    ax.set_ylabel("share of posts")
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    fig.tight_layout()
    return fig


def plot_cadence(cadence: pd.DataFrame, title: str = "Posts per year") -> plt.Figure:
    """Line of posting volume per year."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(cadence["year"], cadence["posts"], marker="o", color="#c13584")
    for _, r in cadence.iterrows():
        ax.annotate(int(r["posts"]), (r["year"], r["posts"]),
                    textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    ax.set_xlabel("year")
    ax.set_ylabel("posts")
    ax.set_title(title)
    ax.set_xticks(cadence["year"])
    fig.tight_layout()
    return fig


def plot_theme_era(era: pd.DataFrame, title: str = "Theme effect: early vs late era") -> plt.Figure:
    """Grouped horizontal bars of each theme's ratio in the early vs late half."""
    e = era.dropna(subset=["early_ratio", "late_ratio"], how="all").iloc[::-1]
    y = np.arange(len(e))
    h = 0.38
    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.6 * len(e) + 1)))
    ax.barh(y + h / 2, e["early_ratio"].fillna(0), height=h, color="#9b59b6",
            alpha=0.85, label="early")
    ax.barh(y - h / 2, e["late_ratio"].fillna(0), height=h, color="#c13584",
            alpha=0.85, label="late")
    ax.axvline(1.0, color="#666", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(e["theme"])
    ax.set_xlabel("median engagement ÷ era baseline")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def save_figures(
    posts: pd.DataFrame,
    *,
    figures_dir: Path,
    **effect_kw,
) -> list[Path]:
    """Write the four drift figures to ``figures_dir``."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    specs: list[tuple[str, plt.Figure]] = [
        ("trends_seasonality.png", plot_seasonality(seasonality(posts, **effect_kw))),
        ("trends_theme_mix.png", plot_theme_mix(theme_mix(posts))),
        ("trends_cadence.png", plot_cadence(cadence_per_year(posts))),
        ("trends_theme_era.png", plot_theme_era(theme_era_effect(posts, **effect_kw))),
    ]
    paths = []
    for name, fig in specs:
        p = figures_dir / name
        fig.savefig(p, dpi=120, bbox_inches="tight")
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
    """Print the seasonality, theme-mix, cadence and early/late tables; write figures."""
    settings = settings or get_settings()
    with db.connection(con, settings) as con:
        posts = eda.load_analysis_posts(con=con, settings=settings)

    eff_kw = dict(n_boot=n_boot, seed=seed)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")

    print(f"analysis_posts: {len(posts)}\n")
    print("=== Seasonality — median engagement ÷ era by month ===")
    seas = seasonality(posts, **eff_kw)
    print(seas[["month_name", "n", "ratio", "ratio_lo", "ratio_hi"]].to_string(index=False))
    sig = eda.clears_ci(seas)
    if not sig.empty:
        print(f"   CI clears 1.0: {', '.join(sig['month_name'])}")

    print("\n=== Theme mix per year (share) ===")
    print(theme_mix(posts).to_string(float_format=lambda v: f"{v:.2f}"))

    print("\n=== Posts per year ===")
    print(cadence_per_year(posts).to_string(index=False))

    print("\n=== Theme effect: early vs late era (ratio vs era, delta = late − early) ===")
    print(theme_era_effect(posts, **eff_kw).to_string(index=False))

    if figures:
        paths = save_figures(posts, figures_dir=settings.figures_dir, **eff_kw)
        print("\nfigures:")
        for p in paths:
            print(f"  → {p}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seasonality & drift.")
    parser.add_argument("--no-figures", action="store_true", help="Skip writing PNGs.")
    parser.add_argument("--n-boot", type=int, default=eda.DEFAULT_N_BOOT)
    parser.add_argument("--seed", type=int, default=eda.DEFAULT_SEED)
    args = parser.parse_args()
    report(figures=not args.no_figures, n_boot=args.n_boot, seed=args.seed)


if __name__ == "__main__":
    main()
