"""Audience shares by age, gender, country and city from the latest snapshot.

Outputs are percentages only; country and city collapse to top-N plus "Other".
Instagram withholds demographics below 100 followers, so the table can be empty.
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

TABLE = "follower_demographics"

# Fixed orders for ordered breakdowns; country and city are sorted by share.
AGE_ORDER = ["13-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
GENDER_ORDER = ["F", "M", "U"]
GENDER_LABEL = {"F": "Female", "M": "Male", "U": "Unknown"}

DEFAULT_TOP_N = 8
BRAND = "#c13584"


def load_demographics(
    *,
    con: duckdb.DuckDBPyConnection | None = None,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Read ``follower_demographics`` and keep only the latest snapshot."""
    with db.connection(con, settings) as con:
        df = con.execute(f"SELECT * FROM {TABLE}").df()
    return latest_snapshot(df)


def latest_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """Rows from the maximum ``snapshot_date`` only (empty in → empty out)."""
    if df.empty:
        return df
    latest = df["snapshot_date"].max()
    return df[df["snapshot_date"] == latest].reset_index(drop=True)


def shares(
    df: pd.DataFrame,
    breakdown: str,
    *,
    order: list[str] | None = None,
) -> pd.DataFrame:
    """``[dimension, share]`` for one breakdown, in ``order`` or by share descending."""
    sub = df[df["breakdown"] == breakdown]
    if sub.empty:
        return pd.DataFrame(columns=["dimension", "share"])
    total = sub["value"].sum()
    out = pd.DataFrame(
        {"dimension": sub["dimension"].to_numpy(), "share": sub["value"] / total}
    )
    if order is not None:
        out["dimension"] = pd.Categorical(out["dimension"], categories=order, ordered=True)
        out = out.sort_values("dimension", kind="stable")
        out["dimension"] = out["dimension"].astype(str)
    else:
        out = out.sort_values("share", ascending=False, kind="stable")
    return out.reset_index(drop=True)


def top_n_shares(df: pd.DataFrame, breakdown: str, *, n: int = DEFAULT_TOP_N) -> pd.DataFrame:
    """Top-``n`` dimensions by share plus one ``"Other (k)"`` row for the rest."""
    s = shares(df, breakdown)  # already share-desc
    if s.empty or len(s) <= n:
        return s
    head = s.iloc[:n].copy()
    tail = s.iloc[n:]
    other = pd.DataFrame(
        {"dimension": [f"Other ({len(tail)})"], "share": [tail["share"].sum()]}
    )
    return pd.concat([head, other], ignore_index=True)


def plot_shares(
    s: pd.DataFrame,
    title: str,
    *,
    horizontal: bool = False,
    labels: dict[str, str] | None = None,
) -> plt.Figure:
    """Bar chart of a ``[dimension, share]`` frame, y-axis in percent."""
    dims = [labels.get(d, d) if labels else d for d in s["dimension"]]
    pct = s["share"] * 100
    if horizontal:
        d, p = dims[::-1], pct.to_numpy()[::-1]  # biggest on top
        fig, ax = plt.subplots(figsize=(8, max(2.5, 0.5 * len(d) + 1)))
        ax.barh(np.arange(len(d)), p, color=BRAND, alpha=0.85)
        ax.set_yticks(np.arange(len(d)))
        ax.set_yticklabels(d)
        ax.set_xlabel("share of followers (%)")
    else:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(np.arange(len(dims)), pct, color=BRAND, alpha=0.85)
        ax.set_xticks(np.arange(len(dims)))
        ax.set_xticklabels(dims)
        ax.set_ylabel("share of followers (%)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def save_figures(
    df: pd.DataFrame,
    *,
    figures_dir: Path,
    top_n: int = DEFAULT_TOP_N,
) -> list[Path]:
    """Write the four demographic share figures to ``figures_dir`` (git-ignored)."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    specs: list[tuple[str, plt.Figure]] = [
        ("demographics_gender.png",
         plot_shares(shares(df, "gender", order=GENDER_ORDER),
                     "Audience by gender", labels=GENDER_LABEL)),
        ("demographics_age.png",
         plot_shares(shares(df, "age", order=AGE_ORDER), "Audience by age")),
        ("demographics_country.png",
         plot_shares(top_n_shares(df, "country", n=top_n),
                     f"Audience by country (top {top_n})", horizontal=True)),
        ("demographics_city.png",
         plot_shares(top_n_shares(df, "city", n=top_n),
                     f"Audience by city (top {top_n})", horizontal=True)),
    ]
    paths = []
    for name, fig in specs:
        p = figures_dir / name
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)
    return paths


def _fmt(s: pd.DataFrame, labels: dict[str, str] | None = None) -> str:
    if s.empty:
        return "  (none)"
    rows = [
        f"  {labels.get(d, d) if labels else d:<26} {sh * 100:5.1f}%"
        for d, sh in zip(s["dimension"], s["share"])
    ]
    return "\n".join(rows)


def report(
    *,
    figures: bool = True,
    top_n: int = DEFAULT_TOP_N,
    con: duckdb.DuckDBPyConnection | None = None,
    settings: Settings | None = None,
) -> None:
    """Print audience shares per breakdown and write the share figures."""
    settings = settings or get_settings()
    df = load_demographics(con=con, settings=settings)
    if df.empty:
        print(f"{TABLE}: EMPTY — run collect.demographics first, or account has <100 followers.")
        return

    snap = df["snapshot_date"].iloc[0]
    print(f"{TABLE}: snapshot {snap}\n")

    print("=== Gender (share) ===")
    print(_fmt(shares(df, "gender", order=GENDER_ORDER), labels=GENDER_LABEL))
    print("\n=== Age (share) ===")
    print(_fmt(shares(df, "age", order=AGE_ORDER)))
    print(f"\n=== Country (top {top_n} share) ===")
    print(_fmt(top_n_shares(df, "country", n=top_n)))
    print(f"\n=== City (top {top_n} share) ===")
    print(_fmt(top_n_shares(df, "city", n=top_n)))

    if figures:
        paths = save_figures(df, figures_dir=settings.figures_dir, top_n=top_n)
        print("\nfigures:")
        for p in paths:
            print(f"  → {p}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audience demographics.")
    parser.add_argument("--no-figures", action="store_true", help="Skip writing PNGs.")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N,
                        help="Top-N countries/cities before the Other bucket.")
    args = parser.parse_args()
    report(figures=not args.no_figures, top_n=args.top_n)


if __name__ == "__main__":
    main()
