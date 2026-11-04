"""Feature engineering — pure helpers + assembler on known fixtures."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import duckdb
import numpy as np
import pandas as pd
import pytest

from insta_analytics.transform import features as ft

BASE_TS = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)


# --- format_of ------------------------------------------------------------


def test_format_maps_all_four_real_shapes():
    assert ft.format_of("VIDEO", "REELS") == "Reel"
    assert ft.format_of("VIDEO", "FEED") == "Video"  # non-reel feed video
    assert ft.format_of("CAROUSEL_ALBUM", "FEED") == "Carousel"
    assert ft.format_of("IMAGE", "FEED") == "Image"


def test_format_reels_wins_over_media_type_and_is_case_insensitive():
    # product_type REELS decides even if media_type were odd; casing ignored.
    assert ft.format_of("video", "reels") == "Reel"
    assert ft.format_of(None, None) == "Image"  # nothing → default Image


# --- caption_features -----------------------------------------------------


def test_caption_counts_tokens_and_flags():
    cap = "🔥 Out now! Grab your #tickets @venue @some.label\nlink in bio?"
    f = ft.caption_features(cap)
    assert f["hashtag_count"] == 1
    assert f["mention_count"] == 2  # @venue and @some.label (dot allowed)
    assert f["emoji_count"] == 1
    assert f["has_question"] is True
    assert f["has_cta"] is True  # "out now" / "link in bio" / "tickets"
    assert f["line_count"] == 2
    assert f["caption_words"] == 11


def test_caption_blank_is_all_zero():
    f = ft.caption_features("")
    assert f["caption_chars"] == 0 and f["caption_words"] == 0
    assert f["line_count"] == 0 and f["has_cta"] is False and f["has_question"] is False
    assert ft.caption_features(None) == f


# --- timing_features (local tz) -------------------------------------------


def test_timing_weekday_not_weekend():
    ts = pd.Timestamp("2024-01-03 09:00", tz="Europe/Berlin")  # Wednesday
    f = ft.timing_features(ts)
    assert f["dow"] == 2 and f["is_weekend"] is False


# Same instant, read on hosts in different zones: 22:30 UTC on Sat 2024-01-06 is
# 23:30 in Berlin. Features must follow Berlin, not the host.
@pytest.mark.parametrize("zone", ["UTC", "America/New_York"])
def test_timing_converts_to_berlin_whatever_the_input_zone(zone):
    ts = pd.Timestamp("2024-01-06 22:30", tz="UTC").tz_convert(zone)
    f = ft.timing_features(ts)
    assert f["hour"] == 23
    assert f["dow"] == 5 and f["is_weekend"] is True
    assert f["month"] == 1


def test_timing_month_rolls_over_in_berlin():
    ts = pd.Timestamp("2023-12-31 23:30", tz="UTC")  # 00:30 on 1 Jan in Berlin
    f = ft.timing_features(ts)
    assert (f["hour"], f["month"]) == (0, 1)


@pytest.mark.parametrize("zone", ["UTC", "America/New_York"])
def test_timing_is_independent_of_duckdb_session_timezone(zone):
    con = duckdb.connect(":memory:")
    con.execute(f"SET TimeZone = '{zone}'")
    con.execute("CREATE TABLE p (timestamp TIMESTAMPTZ)")
    con.execute("INSERT INTO p VALUES ('2024-01-06 22:30:00+00')")
    ts = con.execute("SELECT timestamp FROM p").df()["timestamp"].iloc[0]
    con.close()
    f = ft.timing_features(ts)
    assert (f["hour"], f["dow"], f["month"]) == (23, 5, 1)


# --- cadence_features -----------------------------------------------------


def test_cadence_gap_and_trailing_30d():
    ts = pd.Series(
        pd.to_datetime(
            [BASE_TS, BASE_TS + timedelta(days=2), BASE_TS + timedelta(days=5),
             BASE_TS + timedelta(days=40)]
        )
    )
    cad = ft.cadence_features(ts)
    assert np.isnan(cad["days_since_prev"].iloc[0])  # first post has no prior
    assert cad["days_since_prev"].tolist()[1:] == [2.0, 3.0, 35.0]
    # trailing-30d (strictly-earlier posts within 30d): 0, 1, 2, then 0 (40d gap)
    assert cad["posts_prev_30d"].tolist() == [0.0, 1.0, 2.0, 0.0]


# --- build_features (assembler) -------------------------------------------


def _posts_frame():
    return pd.DataFrame(
        {
            "post_id": ["b", "a"],  # deliberately out of timestamp order
            "timestamp": [BASE_TS + timedelta(days=1), BASE_TS],
            "media_type": ["VIDEO", "IMAGE"],
            "media_product_type": ["REELS", "FEED"],
            "caption": ["Out now #new @x", "quiet day"],
            "theme": ["Release/Promo", "Personal/Lifestyle"],
        }
    )


def test_build_features_sorts_and_shapes():
    feats = ft.build_features(_posts_frame())
    assert list(feats["post_id"]) == ["a", "b"]  # resorted by timestamp
    assert list(feats["format"]) == ["Image", "Reel"]
    assert list(feats["theme"].astype(str)) == ["Personal/Lifestyle", "Release/Promo"]
    assert feats.loc[feats["post_id"] == "b", "has_cta"].iloc[0]  # "Out now"
    assert not feats.loc[feats["post_id"] == "a", "has_cta"].iloc[0]


def test_build_features_categoricals_ordered():
    feats = ft.build_features(_posts_frame())
    assert list(feats["theme"].cat.categories) == list(ft.THEMES)
    assert list(feats["format"].cat.categories) == list(ft.FORMATS)
    assert feats["theme"].cat.ordered and feats["format"].cat.ordered


def test_build_features_no_target_column():
    feats = ft.build_features(_posts_frame())
    assert not any(c.startswith("rel_engagement") for c in feats.columns)
    assert "engagement" not in feats.columns
