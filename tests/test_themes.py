"""Theme validation gate + tidy read surface for the features path."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from insta_analytics.config import Settings
from insta_analytics.storage import db
from insta_analytics.transform import themes
from insta_analytics.transform.labels import THEMES

SETTINGS = Settings(
    meta_app_id="111", meta_app_secret="shh", ig_access_token="LONG", ig_user_id="u1"
)
INGESTED = datetime(2024, 6, 2, 10, 0, tzinfo=timezone.utc)


@pytest.fixture
def con():
    c = db.connect(":memory:")
    db.bootstrap(c)
    yield c
    c.close()


def _post(con, post_id, ts, *, caption="hi", theme=None):
    con.execute(
        "INSERT INTO posts (post_id, ig_user_id, media_type, media_product_type, "
        "caption, permalink, timestamp, theme, ingested_at) "
        "VALUES (?, 'u1', 'IMAGE', 'FEED', ?, 'http://x', ?, ?, ?)",
        [post_id, caption, ts, theme, INGESTED],
    )


def _ts(day):
    return datetime(2024, 1, day, tzinfo=timezone.utc)


# --- validate_labels ------------------------------------------------------


def test_validate_complete_when_all_labeled_and_valid(con):
    _post(con, "a", _ts(1), theme="Show/Tour")
    _post(con, "b", _ts(2), theme="Studio/BTS")
    r = themes.validate_labels(con=con, settings=SETTINGS)
    assert r["complete"] is True
    assert r["total"] == 2 and r["labeled"] == 2
    assert r["unlabeled"] == 0 and r["invalid"] == 0
    assert r["by_theme"] == {"Show/Tour": 1, "Studio/BTS": 1}


def test_validate_flags_null_and_blank_as_unlabeled(con):
    _post(con, "a", _ts(1), theme="Show/Tour")
    _post(con, "b", _ts(2), theme=None)  # NULL
    _post(con, "c", _ts(3), theme="   ")  # blank/whitespace
    r = themes.validate_labels(con=con, settings=SETTINGS)
    assert r["complete"] is False
    assert r["unlabeled"] == 2
    assert set(r["unlabeled_ids"]) == {"b", "c"}
    assert r["labeled"] == 1


def test_validate_catches_invalid_hand_edited_value(con):
    _post(con, "a", _ts(1), theme="Show/Tour")
    _post(con, "c", _ts(2), theme="Banger")  # not in taxonomy
    r = themes.validate_labels(con=con, settings=SETTINGS)
    assert r["complete"] is False
    assert r["invalid"] == 1 and r["invalid_values"] == ["Banger"]
    assert r["invalid_ids"] == ["c"]


def test_validate_empty_db_is_not_complete(con):
    r = themes.validate_labels(con=con, settings=SETTINGS)
    assert r["total"] == 0 and r["complete"] is False


# --- load_labeled_posts ---------------------------------------------------


def test_load_returns_ordered_categorical_in_timestamp_order(con):
    _post(con, "b", _ts(2), theme="Studio/BTS")
    _post(con, "a", _ts(1), theme="Show/Tour")
    df = themes.load_labeled_posts(con=con, settings=SETTINGS)
    assert list(df["post_id"]) == ["a", "b"]  # timestamp order
    assert isinstance(df["theme"].dtype, pd.CategoricalDtype)
    assert df["theme"].cat.ordered
    assert list(df["theme"].cat.categories) == list(THEMES)


def test_load_raises_when_incomplete_by_default(con):
    _post(con, "a", _ts(1), theme="Show/Tour")
    _post(con, "b", _ts(2), theme=None)
    with pytest.raises(ValueError, match="incomplete"):
        themes.load_labeled_posts(con=con, settings=SETTINGS)


def test_load_drops_unlabeled_when_not_requiring_complete(con):
    _post(con, "a", _ts(1), theme="Show/Tour")
    _post(con, "b", _ts(2), theme=None)
    df = themes.load_labeled_posts(con=con, settings=SETTINGS, require_complete=False)
    assert list(df["post_id"]) == ["a"]
    assert df["theme"].notna().all()


def test_load_raises_on_invalid_even_when_not_requiring_complete(con):
    _post(con, "a", _ts(1), theme="Show/Tour")
    _post(con, "c", _ts(2), theme="Banger")
    with pytest.raises(ValueError, match="invalid"):
        themes.load_labeled_posts(con=con, settings=SETTINGS, require_complete=False)
