"""Validate ``posts.theme`` and load labeled posts for the features step.

    uv run python -m insta_analytics.transform.themes validate
"""

from __future__ import annotations

import argparse
import sys

import duckdb
import pandas as pd

from ..config import Settings, get_settings
from ..storage import db
from .labels import THEMES

_LOAD_COLUMNS = (
    "post_id",
    "timestamp",
    "media_type",
    "media_product_type",
    "caption",
    "permalink",
    "theme",
)

# Max ids listed in a report.
_SAMPLE = 10


def validate_labels(
    *,
    con: duckdb.DuckDBPyConnection | None = None,
    settings: Settings | None = None,
) -> dict:
    """Report coverage and validity of ``posts.theme``; NULL and blank count as unlabeled."""
    with db.connection(con, settings) as con:
        rows = con.execute("SELECT post_id, theme FROM posts").fetchall()

    allowed = set(THEMES)
    unlabeled_ids: list[str] = []
    invalid_ids: list[str] = []
    invalid_values: set[str] = set()
    by_theme: dict[str, int] = {t: 0 for t in THEMES}

    for post_id, theme in rows:
        theme = (theme or "").strip()
        if not theme:
            unlabeled_ids.append(post_id)
        elif theme in allowed:
            by_theme[theme] += 1
        else:
            invalid_ids.append(post_id)
            invalid_values.add(theme)

    total = len(rows)
    unlabeled = len(unlabeled_ids)
    invalid = len(invalid_ids)
    labeled = total - unlabeled - invalid
    return {
        "total": total,
        "labeled": labeled,
        "unlabeled": unlabeled,
        "unlabeled_ids": unlabeled_ids[:_SAMPLE],
        "invalid": invalid,
        "invalid_ids": invalid_ids[:_SAMPLE],
        "invalid_values": sorted(invalid_values),
        "by_theme": {t: n for t, n in by_theme.items() if n},
        "complete": total > 0 and unlabeled == 0 and invalid == 0,
    }


def load_labeled_posts(
    *,
    require_complete: bool = True,
    con: duckdb.DuckDBPyConnection | None = None,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Load posts with ``theme`` as an ordered Categorical.

    Invalid themes always raise. Unlabeled posts raise when ``require_complete``,
    and are dropped otherwise.
    """
    settings = settings or get_settings()
    with db.connection(con, settings) as con:
        report = validate_labels(con=con, settings=settings)
        if report["invalid"]:
            raise ValueError(
                f"posts.theme has {report['invalid']} invalid value(s) "
                f"{report['invalid_values']} (ids e.g. {report['invalid_ids']}) — "
                "re-run the label import to fix before loading."
            )
        if require_complete and report["unlabeled"]:
            raise ValueError(
                f"posts.theme is incomplete: {report['unlabeled']}/{report['total']} "
                f"unlabeled (ids e.g. {report['unlabeled_ids']}). Finish labeling "
                "or pass require_complete=False to load only labeled posts."
            )
        df = con.execute(
            f"SELECT {', '.join(_LOAD_COLUMNS)} FROM posts ORDER BY timestamp"
        ).df()

    df["theme"] = df["theme"].where(df["theme"].astype("string").str.strip() != "")
    if not require_complete:
        df = df[df["theme"].notna()].reset_index(drop=True)
    df["theme"] = pd.Categorical(df["theme"], categories=list(THEMES), ordered=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Theme validation gate: check posts.theme coverage/validity."
    )
    parser.add_argument("action", choices=("validate",), help="What to do.")
    args = parser.parse_args()

    if args.action == "validate":
        r = validate_labels()
        status = "OK" if r["complete"] else "INCOMPLETE"
        print(
            f"[{status}] {r['labeled']}/{r['total']} labeled "
            f"({r['unlabeled']} unlabeled, {r['invalid']} invalid)."
        )
        if r["unlabeled"]:
            print(f"  · unlabeled ids (up to {_SAMPLE}): {r['unlabeled_ids']}")
        if r["invalid_values"]:
            print(f"  · invalid values: {r['invalid_values']} (ids {r['invalid_ids']})")
        for theme, n in r["by_theme"].items():
            print(f"  · {theme}: {n}")
        sys.exit(0 if r["complete"] else 1)


if __name__ == "__main__":
    main()
