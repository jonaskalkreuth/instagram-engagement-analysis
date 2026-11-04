"""Which post characteristics go with higher relative engagement.

- Categorical cuts: median of the log2 target with a percentile bootstrap CI,
  reported as a ratio to the era baseline. A CI that excludes 1.0 is a signal.
- Continuous predictors: Spearman ρ with a bootstrap CI.
- Theme × format cells, flagged when they beat both marginals, and quartile
  "sweet spots" for caption length and hashtag count.
- Reach gets the same treatment as a secondary, descriptive target.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")  # headless: render to file, never open a window
import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend switch)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from scipy import stats  # noqa: E402

from ..config import Settings, get_settings  # noqa: E402
from ..storage import db  # noqa: E402
from ..transform.analysis_table import PRIMARY_TABLE, REACH_TABLE  # noqa: E402

ENGAGEMENT_TARGET = "rel_engagement_log2"
REACH_TARGET = "rel_reach_log2"

CATEGORICAL_CUTS: tuple[str, ...] = (
    "format",
    "theme",
    "has_cta",
    "has_question",
    "is_weekend",
    "dow",
)
CONTINUOUS_PREDICTORS: tuple[str, ...] = (
    "caption_chars",
    "caption_words",
    "hashtag_count",
    "days_since_prev",
    "posts_prev_30d",
)

DEFAULT_N_BOOT = 10_000
DEFAULT_SEED = 0
DEFAULT_CI = 0.95
DEFAULT_MIN_N = 5  # groups smaller than this get a point estimate but no CI
DEFAULT_MIN_CELL_N = 5
SWEET_SPOT_PREDICTORS: tuple[str, ...] = ("caption_chars", "hashtag_count")


def _load_table(
    name: str,
    *,
    con: duckdb.DuckDBPyConnection | None = None,
    settings: Settings | None = None,
) -> pd.DataFrame:
    with db.connection(con, settings) as con:
        return con.execute(f"SELECT * FROM {name}").df()


def load_analysis_posts(**kw) -> pd.DataFrame:
    """Read ``analysis_posts``."""
    return _load_table(PRIMARY_TABLE, **kw)


def load_analysis_reach(**kw) -> pd.DataFrame:
    """Read ``analysis_reach``."""
    return _load_table(REACH_TABLE, **kw)


def _boot_median_ci(
    values: np.ndarray, *, n_boot: int, seed: int, ci: float
) -> tuple[float, float]:
    """Percentile bootstrap CI of the median of ``values`` (seeded, vectorized)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    meds = np.median(values[idx], axis=1)
    lo, hi = (1 - ci) / 2, 1 - (1 - ci) / 2
    return float(np.quantile(meds, lo)), float(np.quantile(meds, hi))


def group_effect(
    df: pd.DataFrame,
    by: str,
    *,
    target: str = ENGAGEMENT_TARGET,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    ci: float = DEFAULT_CI,
    min_n: int = DEFAULT_MIN_N,
) -> pd.DataFrame:
    """Median ratio to the era baseline per level of ``by``, with a bootstrap CI.

    Groups under ``min_n`` get no CI. The seed is offset per group, so each
    bootstrap is independent but the table is reproducible. Sorted by ratio.
    """
    valid = df[[by, target]].dropna()
    rows = []
    for i, (level, sub) in enumerate(valid.groupby(by, observed=True, sort=False)):
        vals = sub[target].to_numpy(dtype=float)
        n = len(vals)
        med = float(np.median(vals))
        if n >= min_n:
            lo, hi = _boot_median_ci(vals, n_boot=n_boot, seed=seed + i, ci=ci)
        else:
            lo = hi = np.nan
        rows.append(
            {
                by: level,
                "n": n,
                "ratio": 2.0**med,
                "ratio_lo": 2.0**lo,
                "ratio_hi": 2.0**hi,
                "median_log2": med,
            }
        )
    out = pd.DataFrame(rows, columns=[by, "n", "ratio", "ratio_lo", "ratio_hi", "median_log2"])
    return out.sort_values("ratio", ascending=False, kind="stable").reset_index(drop=True)


def group_median(
    df: pd.DataFrame,
    by: str,
    *,
    target: str,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    ci: float = DEFAULT_CI,
    min_n: int = DEFAULT_MIN_N,
) -> pd.DataFrame:
    """Like :func:`group_effect`, but on the raw scale for targets that are already rates."""
    valid = df[[by, target]].dropna()
    rows = []
    for i, (level, sub) in enumerate(valid.groupby(by, observed=True, sort=False)):
        vals = sub[target].to_numpy(dtype=float)
        n = len(vals)
        med = float(np.median(vals))
        if n >= min_n:
            lo, hi = _boot_median_ci(vals, n_boot=n_boot, seed=seed + i, ci=ci)
        else:
            lo = hi = np.nan
        rows.append({by: level, "n": n, "median": med, "median_lo": lo, "median_hi": hi})
    out = pd.DataFrame(rows, columns=[by, "n", "median", "median_lo", "median_hi"])
    return out.sort_values("median", ascending=False, kind="stable").reset_index(drop=True)


def quantile_bucket(
    series: pd.Series, q: int = 4, *, labels: list[str] | None = None
) -> pd.Series:
    """Bin ``series`` into ``q`` quantiles.

    Duplicate edges collapse into fewer bins, and an all-NaN series returns all-NaN.
    """
    valid = series.dropna()
    if valid.empty:
        return pd.Series(pd.NA, index=series.index, dtype="object")
    return pd.qcut(series, q, labels=labels, duplicates="drop")


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman ρ as the Pearson correlation of ranks."""
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    if rx.std() == 0 or ry.std() == 0:  # a constant predictor has no rank order
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def spearman_table(
    df: pd.DataFrame,
    predictors: tuple[str, ...] = CONTINUOUS_PREDICTORS,
    *,
    target: str = ENGAGEMENT_TARGET,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    ci: float = DEFAULT_CI,
) -> pd.DataFrame:
    """Spearman ρ of each predictor with ``target``, with a paired bootstrap CI."""
    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    rows = []
    for i, pred in enumerate(predictors):
        pair = df[[pred, target]].dropna()
        x = pair[pred].to_numpy(dtype=float)
        y = pair[target].to_numpy(dtype=float)
        rho = _spearman(x, y)
        rng = np.random.default_rng(seed + i)
        idx = rng.integers(0, len(x), size=(n_boot, len(x)))
        boot = np.array([_spearman(x[j], y[j]) for j in idx])
        rows.append(
            {
                "predictor": pred,
                "n": len(x),
                "rho": rho,
                "rho_lo": float(np.nanquantile(boot, lo_q)),
                "rho_hi": float(np.nanquantile(boot, hi_q)),
            }
        )
    out = pd.DataFrame(rows)
    return out.reindex(
        out["rho"].abs().sort_values(ascending=False, kind="stable").index
    ).reset_index(drop=True)


def cell_medians(
    df: pd.DataFrame,
    *,
    row: str = "theme",
    col: str = "format",
    target: str = ENGAGEMENT_TARGET,
    min_n: int = DEFAULT_MIN_CELL_N,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(ratio, n) matrices per ``row`` × ``col`` cell; ratio is NaN where ``n < min_n``."""
    valid = df[[row, col, target]].dropna()
    med = valid.pivot_table(index=row, columns=col, values=target, aggfunc="median")
    n = valid.pivot_table(index=row, columns=col, values=target, aggfunc="count")
    n = n.reindex_like(med).fillna(0).astype(int)
    ratio = (2.0**med).where(n >= min_n)
    return ratio, n


def beats_both_marginals(
    df: pd.DataFrame,
    *,
    row: str = "theme",
    col: str = "format",
    target: str = ENGAGEMENT_TARGET,
    min_n: int = DEFAULT_MIN_CELL_N,
) -> pd.DataFrame:
    """True where a cell's ratio beats both its row and column marginal ratio.

    Blanked small-``n`` cells are False.
    """
    ratio, _ = cell_medians(df, row=row, col=col, target=target, min_n=min_n)
    row_marg = group_effect(df, row, target=target, min_n=1).set_index(row)["ratio"]
    col_marg = group_effect(df, col, target=target, min_n=1).set_index(col)["ratio"]
    beats_row = ratio.gt(row_marg.reindex(ratio.index), axis=0)
    beats_col = ratio.gt(col_marg.reindex(ratio.columns), axis=1)
    return (beats_row & beats_col).fillna(False)


def sweet_spot(
    df: pd.DataFrame,
    predictor: str,
    *,
    q: int = 4,
    target: str = ENGAGEMENT_TARGET,
    min_n: int = DEFAULT_MIN_N,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """:func:`group_effect` over quantile buckets of ``predictor``, in bucket order."""
    d = df[[predictor, target]].dropna().copy()
    d["_bucket"] = quantile_bucket(d[predictor], q)
    eff = group_effect(d, "_bucket", target=target, min_n=min_n, n_boot=n_boot, seed=seed)
    order = {c: i for i, c in enumerate(d["_bucket"].cat.categories)}
    eff = (
        eff.assign(_o=eff["_bucket"].map(order))
        .sort_values("_o", kind="stable")
        .drop(columns="_o")
        .reset_index(drop=True)
        .rename(columns={"_bucket": predictor})
    )
    eff[predictor] = eff[predictor].astype(str)  # Intervals → labels for plotting
    return eff


def clears_ci(effect: pd.DataFrame) -> pd.DataFrame:
    """Rows whose CI excludes 1.0."""
    return effect[(effect["ratio_lo"] > 1.0) | (effect["ratio_hi"] < 1.0)]


def plot_group_effect(effect: pd.DataFrame, by: str, title: str) -> plt.Figure:
    """Horizontal bars of median ratio per group with CI whiskers and a 1.0 line."""
    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.5 * len(effect) + 1)))
    e = effect.iloc[::-1].reset_index(drop=True)  # strongest at top
    ypos = np.arange(len(e))
    err = np.vstack([e["ratio"] - e["ratio_lo"], e["ratio_hi"] - e["ratio"]])
    err = np.where(np.isnan(err), 0.0, err)  # small groups: no whisker
    ax.barh(ypos, e["ratio"], color="#c13584", alpha=0.85)
    ax.errorbar(e["ratio"], ypos, xerr=err, fmt="none", ecolor="#333", capsize=3)
    ax.axvline(1.0, color="#666", linestyle="--", linewidth=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{lvl}  (n={n})" for lvl, n in zip(e[by], e["n"])])
    ax.set_xlabel("median engagement ÷ era baseline  (1.0 = typical)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_timing_heatmap(df: pd.DataFrame, target: str = ENGAGEMENT_TARGET) -> plt.Figure:
    """Day-of-week × hour median relative engagement (ratio), blank where no posts."""
    piv = (
        df.dropna(subset=[target])
        .assign(ratio=lambda d: 2.0 ** d[target])
        .pivot_table(index="dow", columns="hour", values="ratio", aggfunc="median")
        .reindex(index=range(7))
    )
    fig, ax = plt.subplots(figsize=(11, 4))
    im = ax.imshow(piv.to_numpy(), aspect="auto", cmap="magma", origin="lower")
    ax.set_yticks(range(7))
    ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns)
    ax.set_xlabel("hour of day (local)")
    ax.set_title("Median relative engagement by day × hour (ratio vs era)")
    fig.colorbar(im, ax=ax, label="ratio")
    fig.tight_layout()
    return fig


def plot_cell_heatmap(
    ratio: pd.DataFrame, n: pd.DataFrame, flags: pd.DataFrame, title: str
) -> plt.Figure:
    """Heatmap of the ratio matrix; flagged cells get a ``*`` and a border."""
    data = ratio.to_numpy(dtype=float)
    fig, ax = plt.subplots(
        figsize=(2.0 + 1.3 * data.shape[1], 1.2 + 0.6 * data.shape[0])
    )
    im = ax.imshow(data, aspect="auto", cmap="magma")
    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels(ratio.columns, rotation=30, ha="right")
    ax.set_yticks(range(data.shape[0]))
    ax.set_yticklabels(ratio.index)
    mid = np.nanmedian(data) if np.isfinite(data).any() else 1.0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = ratio.iloc[i, j]
            if pd.isna(v):
                ax.text(j, i, "·", ha="center", va="center", color="#888")
                continue
            mark = "*" if bool(flags.iloc[i, j]) else ""
            colour = "black" if v >= mid else "white"
            ax.text(j, i, f"{v:.2f}{mark}\nn={int(n.iloc[i, j])}",
                    ha="center", va="center", fontsize=8, color=colour)
            if mark:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor="#00e0c0", linewidth=2.5))
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="ratio vs era (blank = n<min)")
    fig.tight_layout()
    return fig


def save_figures(
    posts: pd.DataFrame,
    reach: pd.DataFrame,
    *,
    figures_dir: Path,
    **effect_kw,
) -> list[Path]:
    """Write the EDA figures to ``figures_dir``."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("eda_format.png", plot_group_effect(
            group_effect(posts, "format", **effect_kw), "format", "Engagement by format")),
        ("eda_theme.png", plot_group_effect(
            group_effect(posts, "theme", **effect_kw), "theme", "Engagement by theme")),
        ("eda_timing.png", plot_timing_heatmap(posts)),
        ("eda_reach_format.png", plot_group_effect(
            group_effect(reach, "format", target=REACH_TARGET, **effect_kw),
            "format", "Reach by format (secondary/descriptive)")),
        ("eda_reach_theme.png", plot_group_effect(
            group_effect(reach, "theme", target=REACH_TARGET, **effect_kw),
            "theme", "Reach by theme (secondary/descriptive)")),
    ]
    ratio, n = cell_medians(posts)
    specs.append(("eda_theme_format.png", plot_cell_heatmap(
        ratio, n, beats_both_marginals(posts), "Theme × format: relative engagement")))
    for pred in SWEET_SPOT_PREDICTORS:
        eff = sweet_spot(posts, pred, **effect_kw)
        specs.append((f"eda_{pred}_sweetspot.png",
                      plot_group_effect(eff, pred, f"{pred} sweet spot (by quartile)")))
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
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    con: duckdb.DuckDBPyConnection | None = None,
    settings: Settings | None = None,
) -> None:
    """Print the effect-size and correlation tables and write figures."""
    settings = settings or get_settings()
    with db.connection(con, settings) as con:
        posts = load_analysis_posts(con=con, settings=settings)
        reach = load_analysis_reach(con=con, settings=settings)

    eff_kw = dict(n_boot=n_boot, seed=seed)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")

    print(f"analysis_posts: {len(posts)} · analysis_reach: {len(reach)}\n")
    print("=== Engagement (relative, full history) ===")
    for cut in CATEGORICAL_CUTS:
        eff = group_effect(posts, cut, **eff_kw)
        print(f"\n-- by {cut} --")
        print(eff.to_string(index=False))
        sig = clears_ci(eff)
        if not sig.empty:
            print(f"   CI clears 1.0: {', '.join(str(v) for v in sig[cut])}")

    print("\n-- continuous (Spearman ρ vs rel_engagement_log2) --")
    print(spearman_table(posts, **eff_kw).to_string(index=False))

    print("\n=== Theme × format — median engagement ÷ era (blank = n<min) ===")
    ratio, n = cell_medians(posts)
    flags = beats_both_marginals(posts)
    print(ratio.to_string(float_format=lambda v: f"{v:.2f}"))
    print("\n-- cell n --")
    print(n.to_string())
    winners = [f"{r}×{c}" for r in flags.index for c in flags.columns if flags.loc[r, c]]
    print(f"\ncells beating BOTH marginals: {', '.join(winners) if winners else 'none'}")

    print("\n=== Non-linear sweet spots (quartile buckets, bootstrap CI) ===")
    for pred in SWEET_SPOT_PREDICTORS:
        print(f"\n-- {pred} --")
        print(sweet_spot(posts, pred, **eff_kw).to_string(index=False))

    print("\n=== Reach (relative, descriptive — reach-available subset) ===")
    for cut in ("format", "theme"):
        print(f"\n-- by {cut} --")
        print(group_effect(reach, cut, target=REACH_TARGET, **eff_kw).to_string(index=False))

    if figures:
        paths = save_figures(posts, reach, figures_dir=settings.figures_dir, **eff_kw)
        print("\nfigures:")
        for p in paths:
            print(f"  → {p}")


def main() -> None:
    parser = argparse.ArgumentParser(description="EDA: relative engagement by characteristic.")
    parser.add_argument("--no-figures", action="store_true", help="Skip writing PNGs.")
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    report(figures=not args.no_figures, n_boot=args.n_boot, seed=args.seed)


if __name__ == "__main__":
    main()
