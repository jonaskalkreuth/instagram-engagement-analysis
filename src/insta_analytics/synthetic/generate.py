"""Pure, seeded row builders for the four schema tables."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

import numpy as np

from . import signals as S

_COLLECTED_AT = datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc)  # single snapshot


def _choice(rng: np.random.Generator, mapping: dict[str, float]) -> str:
    """Weighted pick of a key; weights are normalized so they need not sum to 1."""
    keys = list(mapping)
    p = np.array([mapping[k] for k in keys], dtype=float)
    return str(rng.choice(keys, p=p / p.sum()))


def _followers(ts: datetime) -> float:
    """Exponential follower growth over the timeline; the era baseline removes it."""
    span = (S.TIMELINE_END - S.TIMELINE_START).days
    frac = np.clip((ts.date() - S.TIMELINE_START).days / span, 0.0, 1.0)
    return S.FOLLOWER_START * (S.FOLLOWER_BASE / S.FOLLOWER_START) ** frac


def _make_caption(rng: np.random.Generator) -> str:
    """Filler caption with a realistic length + hashtags. NO engagement signal."""
    n_chars = int(np.exp(rng.normal(S.CAPTION_CHARS_LOG_MEAN, S.CAPTION_CHARS_LOG_SD)))
    n_chars = int(np.clip(n_chars, 20, 1500))
    words: list[str] = []
    while len(" ".join(words)) < n_chars:
        words.append(str(rng.choice(S._CAPTION_WORDS)))
    body = " ".join(words)
    if rng.random() < 0.25:
        body += "?"
    if rng.random() < 0.30:
        body += " " + str(rng.choice(S.CTA_PHRASES))
    n_tags = int(np.clip(rng.poisson(S.HASHTAG_LAMBDA), 0, S.HASHTAG_MAX))
    tags = " ".join("#" + str(t) for t in rng.choice(S._HASHTAGS, size=n_tags))
    return (body + "\n\n" + tags).strip()


def _year_timestamps(rng: np.random.Generator, year: int, n: int) -> list[datetime]:
    """``n`` jittered publish datetimes within ``year`` (clipped to the timeline)."""
    lo = max(date(year, 1, 1), S.TIMELINE_START)
    hi = min(date(year, 12, 31), S.TIMELINE_END)
    span = (hi - lo).days or 1
    offsets = np.sort(rng.integers(0, span + 1, size=n))
    out = []
    for off in offsets:
        d = lo + timedelta(days=int(off))
        t = time(hour=int(rng.integers(9, 23)), minute=int(rng.integers(0, 60)))
        # tz-aware UTC; timing carries no planted signal so the offset is cosmetic.
        out.append(datetime.combine(d, t, tzinfo=timezone.utc))
    return out


def generate_posts(rng: np.random.Generator) -> list[dict]:
    """440 ``posts`` rows in timestamp order, with planted theme + format."""
    rows: list[dict] = []
    for year, n in S.POSTS_PER_YEAR.items():
        mix = S.THEME_MIX_BY_YEAR[year]
        for ts in _year_timestamps(rng, year, n):
            theme = _choice(rng, mix)
            fmt = _choice(rng, S.FORMAT_BY_THEME[theme])
            media_type, product_type = S.FORMAT_MEDIA[fmt]
            rows.append(
                {
                    "post_id": "",  # assigned after the global timestamp sort
                    "ig_user_id": S.IG_USER_ID,
                    "media_type": media_type,
                    "media_product_type": product_type,
                    "caption": _make_caption(rng),
                    "permalink": "",
                    "timestamp": ts,
                    "theme": theme,
                    "ingested_at": _COLLECTED_AT,
                    "_format": fmt,  # scratch, dropped before landing
                }
            )
    rows.sort(key=lambda r: r["timestamp"])
    for i, r in enumerate(rows):
        r["post_id"] = f"syn_{i:04d}"
        r["permalink"] = f"https://example.com/p/syn_{i:04d}"
    return rows


def generate_metrics(rng: np.random.Generator, posts: list[dict]) -> list[dict]:
    """One ``post_metrics`` row per post; ``posts`` must be sorted (earliest get NULL reach)."""
    rows: list[dict] = []
    for i, p in enumerate(posts):
        fmt, theme = p["_format"], p["theme"]
        followers = _followers(p["timestamp"])

        # engagement = likes + comments (the relative-engagement base)
        eng = (
            followers * S.ENG_RATE
            * S.F_FORMAT[fmt] * S.F_THEME[theme]
            * S.F_CELL.get((theme, fmt), 1.0)
            * rng.lognormal(0.0, S.SIGMA_ENG)
        )
        cs = float(np.clip(rng.normal(S.COMMENT_SHARE_MEAN, S.COMMENT_SHARE_SD), 0.01, 0.15))
        likes = max(0, round(eng * (1 - cs)))
        comments = max(0, round(eng * cs))

        # reach — milder echo; NULL for the earliest N_NO_REACH posts
        if i < S.N_NO_REACH:
            reach = None
        else:
            base_reach = followers * S.REACH_RATE
            reach = max(
                1,
                round(
                    base_reach
                    * S.F_THEME[theme] ** S.REACH_COMPRESS
                    * S.F_FORMAT[fmt] ** S.REACH_COMPRESS
                    * rng.lognormal(0.0, S.SIGMA_REACH)
                ),
            )

        saved = max(
            0,
            round(
                followers * S.SAVE_RATE * S.S_THEME[theme] * S.S_FORMAT[fmt]
                * rng.lognormal(0.0, S.SIGMA_SAVE)
            ),
        )
        shares = max(
            0,
            round(
                followers * S.SHARE_RATE * S.SH_THEME[theme] * S.SH_FORMAT[fmt]
                * rng.lognormal(0.0, S.SIGMA_SHARE)
            ),
        )
        views = (
            round(reach * rng.uniform(S.VIEWS_REACH_LO, S.VIEWS_REACH_HI))
            if reach is not None and fmt in ("Reel", "Video")
            else None
        )
        rows.append(
            {
                "post_id": p["post_id"],
                "collected_at": _COLLECTED_AT,
                "like_count": likes,
                "comments_count": comments,
                "reach": reach,
                "saved": saved,
                "shares": shares,
                "views": views,
                "total_interactions": likes + comments + saved + shares,
            }
        )
    return rows


def generate_account_snapshots(rng: np.random.Generator) -> list[dict]:
    """30 daily rows each of ``follower_count`` + ``reach`` ending at the timeline end."""
    rows: list[dict] = []
    for back in range(S.SNAPSHOT_DAYS):
        d = S.TIMELINE_END - timedelta(days=back)
        followers = round(_followers(datetime.combine(d, time(12), tzinfo=timezone.utc)))
        reach = max(0, round(rng.normal(S.DAILY_REACH_MEAN, S.DAILY_REACH_SD)))
        for metric, value in (("follower_count", followers), ("reach", reach)):
            rows.append(
                {
                    "ig_user_id": S.IG_USER_ID,
                    "date": d,
                    "metric": metric,
                    "value": value,
                    "collected_at": _COLLECTED_AT,
                }
            )
    return rows


def _breakdown_rows(breakdown: str, shares: dict[str, float]) -> list[dict]:
    counts = {dim: round(sh * S.FOLLOWER_BASE) for dim, sh in shares.items()}
    return [
        {
            "ig_user_id": S.IG_USER_ID,
            "snapshot_date": S.DEMO_SNAPSHOT,
            "breakdown": breakdown,
            "dimension": dim,
            "value": value,
            "collected_at": _COLLECTED_AT,
        }
        for dim, value in counts.items()
    ]


def generate_demographics() -> list[dict]:
    """One ``follower_demographics`` snapshot: gender/age/country/city breakdowns."""
    rows: list[dict] = []
    rows += _breakdown_rows("gender", S.GENDER_SHARES)
    rows += _breakdown_rows("age", S.AGE_SHARES)
    rows += _breakdown_rows("country", S.COUNTRY_SHARES)
    rows += _breakdown_rows("city", S.CITY_SHARES)
    return rows


def generate_all(seed: int = S.SEED) -> dict[str, list[dict]]:
    """All four tables, ready to land, with the scratch ``_format`` key removed."""
    rng = np.random.default_rng(seed)
    posts = generate_posts(rng)
    metrics = generate_metrics(rng, posts)
    account = generate_account_snapshots(rng)
    demographics = generate_demographics()
    for p in posts:
        p.pop("_format", None)
    return {
        "posts": posts,
        "post_metrics": metrics,
        "account_snapshots": account,
        "follower_demographics": demographics,
    }
