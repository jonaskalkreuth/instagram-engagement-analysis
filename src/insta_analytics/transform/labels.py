"""Theme labeling via a CSV round-trip (taxonomy: ``docs/theme-taxonomy.md``).

    uv run python -m insta_analytics.transform.labels export   # write themes.csv
    uv run python -m insta_analytics.transform.labels import   # land into DuckDB

Export keeps themes already filled in the CSV or the DB. Import writes only valid
themes, with ``UPDATE`` rather than ``db.upsert``: an upsert of just
``{post_id, theme}`` would insert rows with a NULL ``ig_user_id``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import duckdb
import pandas as pd

from ..config import Settings
from ..storage import db

# Single source of truth for the taxonomy and its order.
THEMES: tuple[str, ...] = (
    "Release/Promo",
    "Show/Tour",
    "Live/Performance",
    "Broadcast/Mix",
    "Studio/BTS",
    "Personal/Lifestyle",
    "Other",
)

# Theme sits before the long caption, for easier tagging.
_CSV_COLUMNS = (
    "post_id",
    "timestamp",
    "media_type",
    "media_product_type",
    "theme",
    "caption",
    "permalink",
)

# data/labels/ is git-ignored: real captions are private.
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CSV = _REPO_ROOT / "data" / "labels" / "themes.csv"


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a label CSV as all-strings with blanks as ``""`` (never NaN)."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def build_export_frame(posts: pd.DataFrame, existing: pd.DataFrame | None) -> pd.DataFrame:
    """Merge DB posts with an existing CSV; a non-blank CSV theme wins over the DB."""
    out = posts.copy()
    out["theme"] = out["theme"].fillna("")
    if existing is not None and not existing.empty:
        prior = existing.set_index("post_id")["theme"]
        mapped = out["post_id"].map(prior)
        # CSV theme wins where it's non-blank; otherwise keep the DB theme.
        out["theme"] = mapped.where(mapped.notna() & (mapped != ""), out["theme"])
    out = out.sort_values("timestamp", kind="stable", na_position="last")
    return out[list(_CSV_COLUMNS)]


def export_labels(
    csv_path: str | Path | None = None,
    *,
    con: duckdb.DuckDBPyConnection | None = None,
    settings: Settings | None = None,
) -> dict[str, int]:
    """Write/refresh the label CSV from ``posts``. Returns total/labeled/blank."""
    csv_path = Path(csv_path) if csv_path is not None else DEFAULT_CSV

    with db.connection(con, settings) as con:
        posts = con.execute(
            "SELECT post_id, timestamp, media_type, media_product_type, "
            "caption, permalink, theme FROM posts"
        ).df()

    existing = _read_csv(csv_path) if csv_path.exists() else None
    frame = build_export_frame(posts, existing)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)

    labeled = int((frame["theme"] != "").sum())
    return {"total": len(frame), "labeled": labeled, "blank": len(frame) - labeled}


def import_labels(
    csv_path: str | Path | None = None,
    *,
    con: duckdb.DuckDBPyConnection | None = None,
    settings: Settings | None = None,
) -> dict:
    """Write valid CSV themes into ``posts.theme``; skip blanks, report invalid values."""
    csv_path = Path(csv_path) if csv_path is not None else DEFAULT_CSV
    df = _read_csv(csv_path)

    valid: list[tuple[str, str]] = []  # (theme, post_id) for the UPDATE
    blank = 0
    invalid: Counter[str] = Counter()
    for post_id, theme in zip(df["post_id"], df["theme"]):
        theme = (theme or "").strip()
        if not theme:
            blank += 1
        elif theme in THEMES:
            valid.append((theme, post_id))
        else:
            invalid[theme] += 1

    with db.connection(con, settings) as con:
        if valid:
            con.executemany("UPDATE posts SET theme = ? WHERE post_id = ?", valid)

    by_theme = Counter(theme for theme, _ in valid)
    return {
        "labeled": len(valid),
        "blank": blank,
        "invalid": int(sum(invalid.values())),
        "invalid_values": sorted(invalid),
        "by_theme": {t: by_theme.get(t, 0) for t in THEMES if by_theme.get(t)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Theme labeling: export/import.")
    parser.add_argument("action", choices=("export", "import"), help="What to do.")
    parser.add_argument("--csv", default=None, help=f"CSV path (default {DEFAULT_CSV}).")
    args = parser.parse_args()

    if args.action == "export":
        r = export_labels(args.csv)
        print(f"→ Exported {r['total']} posts ({r['labeled']} labeled, {r['blank']} blank).")
    else:
        r = import_labels(args.csv)
        print(f"→ Imported {r['labeled']} labels ({r['blank']} blank, {r['invalid']} invalid).")
        if r["invalid_values"]:
            print(f"  · invalid theme values (not written): {r['invalid_values']}")
        for theme, n in r["by_theme"].items():
            print(f"  · {theme}: {n}")


if __name__ == "__main__":
    main()
