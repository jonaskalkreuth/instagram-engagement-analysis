"""Build the synthetic DuckDB: generate → land → run the real transform pipeline.

    uv run python -m insta_analytics.synthetic.build            # → data/synthetic/insta.duckdb
    uv run python -m insta_analytics.synthetic.build --seed 7   # different draw

Uses dummy ``Settings``, so it needs no ``.env``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..config import Settings
from ..storage import db
from ..transform import analysis_table
from . import generate
from . import signals as S

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = _REPO_ROOT / "data" / "synthetic" / "insta.duckdb"


def _synthetic_settings(path: Path) -> Settings:
    """A creds-free Settings for the transform layer (con is passed, so path is unused)."""
    return Settings(
        meta_app_id="synthetic",
        meta_app_secret="synthetic",
        ig_access_token="synthetic",
        ig_user_id=S.IG_USER_ID,
        data_source="synthetic",
        duckdb_path=path,
    )


def build(out: str | Path | None = None, *, seed: int = S.SEED, fresh: bool = True) -> dict:
    """Build a synthetic DB at ``out`` (replacing it if ``fresh``) and return row counts."""
    out = Path(out) if out is not None else DEFAULT_OUT
    if fresh and out.exists():
        out.unlink()

    tables = generate.generate_all(seed)
    settings = _synthetic_settings(out)

    con = db.connect(out)
    try:
        db.bootstrap(con)
        counts = {name: db.upsert(con, name, rows) for name, rows in tables.items()}
        result = analysis_table.assemble(con=con, settings=settings)
    finally:
        con.close()

    counts["analysis_posts"] = len(result["analysis_posts"])
    counts["analysis_reach"] = len(result["analysis_reach"])
    return counts


def _recovery_summary(out: Path) -> None:
    """Print the recovered planted signal so a build shows the story landed."""
    from ..analysis import eda

    con = db.connect(out)
    try:
        posts = con.execute("SELECT * FROM analysis_posts").df()
    finally:
        con.close()
    med = posts["rel_engagement"].median()
    print(f"  median rel_engagement = {med:.3f} (target ~1.0)")
    for by in ("theme", "format"):
        eff = eda.group_effect(posts, by)
        pairs = ", ".join(f"{r[by]} {r['ratio']:.2f}×" for _, r in eff.iterrows())
        print(f"  {by}: {pairs}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the synthetic DuckDB.")
    parser.add_argument("--seed", type=int, default=S.SEED)
    parser.add_argument("--out", default=None, help=f"Output DB path (default {DEFAULT_OUT}).")
    args = parser.parse_args()

    counts = build(args.out, seed=args.seed)
    out = Path(args.out) if args.out else DEFAULT_OUT
    print(f"→ Built {out}")
    for name, n in counts.items():
        print(f"  {name}: {n}")
    _recovery_summary(out)


if __name__ == "__main__":
    main()
