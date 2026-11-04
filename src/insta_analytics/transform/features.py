"""Labeled posts → one predictor row per post: format, theme, caption, timing, cadence.

Timing is read in Europe/Berlin, converted explicitly because DuckDB returns
``TIMESTAMPTZ`` in the host's zone.
"""

from __future__ import annotations

import argparse
import re

import duckdb
import numpy as np
import pandas as pd

from ..config import Settings, get_settings
from .labels import THEMES
from .themes import load_labeled_posts

FORMATS: tuple[str, ...] = ("Reel", "Video", "Carousel", "Image")

LOCAL_TZ = "Europe/Berlin"

_HASHTAG_RE = re.compile(r"#\w+")
_MENTION_RE = re.compile(r"@[\w.]+")
# Emoji-ish codepoints: pictographs, symbols, transport, dingbats, flags. Broad by
# design — we want a "how decorated is this caption" count, not exact grapheme
# segmentation, so a few symbol chars counting as emoji is acceptable.
_EMOJI_RE = re.compile(
    "["
    "\U0001f300-\U0001faff"  # symbols & pictographs, supplemental, extended-A
    "\U00002600-\U000027bf"  # misc symbols + dingbats
    "\U0001f1e6-\U0001f1ff"  # regional indicators (flags)
    "\U00002190-\U000021ff"  # arrows
    "\U00002b00-\U00002bff"  # misc symbols & arrows (stars, etc.)
    "\U0000fe0f"             # variation selector-16 (emoji presentation)
    "]"
)

# Call-to-action phrases, matched as lowercase substrings.
_CTA_PHRASES: tuple[str, ...] = (
    "link in bio",
    "out now",
    "out today",
    "tickets",
    "ticket link",
    "listen",
    "stream",
    "streaming",
    "pre-save",
    "presave",
    "pre save",
    "download",
    "watch",
    "swipe",
    "sign up",
    "grab your",
    "get your",
    "check it out",
    "check out",
)


def format_of(media_type: str | None, media_product_type: str | None) -> str:
    """Map the raw media type pair to Reel, Video, Carousel or Image."""
    if (media_product_type or "").upper() == "REELS":
        return "Reel"
    mt = (media_type or "").upper()
    if mt == "VIDEO":
        return "Video"
    if mt == "CAROUSEL_ALBUM":
        return "Carousel"
    return "Image"


def caption_features(caption: str | None) -> dict:
    """Pure per-caption predictors. ``None``/blank → all-zero / False counts."""
    text = caption or ""
    stripped = text.strip()
    words = stripped.split()
    lower = text.lower()
    return {
        "caption_chars": len(stripped),
        "caption_words": len(words),
        "line_count": (stripped.count("\n") + 1) if stripped else 0,
        "hashtag_count": len(_HASHTAG_RE.findall(text)),
        "mention_count": len(_MENTION_RE.findall(text)),
        "emoji_count": len(_EMOJI_RE.findall(text)),
        "has_question": "?" in text,
        "has_cta": any(p in lower for p in _CTA_PHRASES),
    }


def timing_features(ts: pd.Timestamp) -> dict:
    """Hour, day of week (Monday=0), weekend flag and month in :data:`LOCAL_TZ`."""
    ts = ts.tz_convert(LOCAL_TZ)
    return {
        "hour": int(ts.hour),
        "dow": int(ts.dayofweek),
        "is_weekend": bool(ts.dayofweek >= 5),
        "month": int(ts.month),
    }


def cadence_features(timestamps: pd.Series) -> pd.DataFrame:
    """``days_since_prev`` and ``posts_prev_30d`` for timestamp-sorted publish times."""
    ts = pd.to_datetime(timestamps)
    days_since_prev = ts.diff().dt.total_seconds() / 86_400.0

    vals = ts.to_numpy()
    prev_30d = np.empty(len(vals), dtype=float)
    window = np.timedelta64(30, "D")
    for i in range(len(vals)):
        lo = vals[i] - window
        prev_30d[i] = int(np.count_nonzero((vals[:i] > lo) & (vals[:i] <= vals[i])))
    return pd.DataFrame(
        {"days_since_prev": days_since_prev.to_numpy(), "posts_prev_30d": prev_30d},
        index=timestamps.index,
    )


def build_features(posts: pd.DataFrame) -> pd.DataFrame:
    """Build the predictor frame from :func:`themes.load_labeled_posts` output."""
    out = posts.sort_values("timestamp", kind="stable").reset_index(drop=True).copy()

    out["format"] = pd.Categorical(
        [format_of(mt, mpt) for mt, mpt in zip(out["media_type"], out["media_product_type"])],
        categories=list(FORMATS),
        ordered=True,
    )
    out["theme"] = pd.Categorical(out["theme"], categories=list(THEMES), ordered=True)

    cap = pd.DataFrame([caption_features(c) for c in out["caption"]], index=out.index)
    tim = pd.DataFrame([timing_features(t) for t in out["timestamp"]], index=out.index)
    cad = cadence_features(out["timestamp"])

    feats = pd.concat([out[["post_id", "timestamp", "theme", "format"]], cap, tim, cad], axis=1)
    return feats


def compute(
    *,
    require_complete: bool = True,
    con: duckdb.DuckDBPyConnection | None = None,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Load labeled posts from the DB and build the predictor frame."""
    posts = load_labeled_posts(require_complete=require_complete, con=con, settings=settings)
    return build_features(posts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Feature engineering: build + summarize the predictor frame."
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Build over only the labeled posts instead of raising on holes.",
    )
    args = parser.parse_args()

    feats = compute(require_complete=not args.allow_incomplete)
    print(f"posts: {len(feats)}  |  features: {feats.shape[1] - 2}")  # minus post_id, timestamp
    print("\nformat:")
    print(feats["format"].value_counts().to_string())
    print("\ntheme:")
    print(feats["theme"].value_counts().to_string())
    num = feats.select_dtypes("number")
    print("\nnumeric features (describe):")
    print(num.describe().T.to_string())
    print("\nflags:")
    print(f"  has_cta:      {int(feats['has_cta'].sum())}/{len(feats)}")
    print(f"  has_question: {int(feats['has_question'].sum())}/{len(feats)}")
    print(f"  is_weekend:   {int(feats['is_weekend'].sum())}/{len(feats)}")


if __name__ == "__main__":
    main()
