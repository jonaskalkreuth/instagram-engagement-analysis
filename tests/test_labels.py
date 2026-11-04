"""Theme labeling workflow — export CSV, hand-tag, re-import to DuckDB."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from insta_analytics.config import Settings
from insta_analytics.storage import db
from insta_analytics.transform import labels

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


def _theme(con, post_id):
    return con.execute("SELECT theme FROM posts WHERE post_id = ?", [post_id]).fetchone()[0]


# --- export ---------------------------------------------------------------


def test_export_writes_expected_columns_ordered_by_timestamp(con, tmp_path):
    _post(con, "b", datetime(2024, 2, 1, tzinfo=timezone.utc))
    _post(con, "a", datetime(2024, 1, 1, tzinfo=timezone.utc))
    csv = tmp_path / "themes.csv"

    r = labels.export_labels(csv, con=con, settings=SETTINGS)
    assert r == {"total": 2, "labeled": 0, "blank": 2}

    df = pd.read_csv(csv, dtype=str, keep_default_na=False)
    assert list(df.columns) == list(labels._CSV_COLUMNS)
    assert list(df["post_id"]) == ["a", "b"]  # timestamp order
    assert (df["theme"] == "").all()


def test_export_counts_db_labels_as_labeled(con, tmp_path):
    _post(con, "a", datetime(2024, 1, 1, tzinfo=timezone.utc), theme="Show/Tour")
    r = labels.export_labels(tmp_path / "themes.csv", con=con, settings=SETTINGS)
    assert r == {"total": 1, "labeled": 1, "blank": 0}


def test_export_preserves_existing_csv_labels_and_appends_new_blank(con, tmp_path):
    _post(con, "a", datetime(2024, 1, 1, tzinfo=timezone.utc))
    csv = tmp_path / "themes.csv"
    labels.export_labels(csv, con=con, settings=SETTINGS)

    # Hand-tag "a", then a new post arrives and we re-export.
    df = pd.read_csv(csv, dtype=str, keep_default_na=False)
    df.loc[df["post_id"] == "a", "theme"] = "Studio/BTS"
    df.to_csv(csv, index=False)
    _post(con, "b", datetime(2024, 2, 1, tzinfo=timezone.utc))

    r = labels.export_labels(csv, con=con, settings=SETTINGS)
    assert r == {"total": 2, "labeled": 1, "blank": 1}
    out = pd.read_csv(csv, dtype=str, keep_default_na=False).set_index("post_id")["theme"]
    assert out["a"] == "Studio/BTS"  # preserved
    assert out["b"] == ""  # new post, still blank


# --- import ---------------------------------------------------------------


def _csv(tmp_path, rows):
    csv = tmp_path / "themes.csv"
    pd.DataFrame(rows, columns=list(labels._CSV_COLUMNS)).to_csv(csv, index=False)
    return csv


def test_import_skips_blank_and_rejects_invalid(con, tmp_path):
    _post(con, "a", datetime(2024, 1, 1, tzinfo=timezone.utc))
    _post(con, "b", datetime(2024, 2, 1, tzinfo=timezone.utc))
    _post(con, "c", datetime(2024, 3, 1, tzinfo=timezone.utc))
    csv = _csv(
        tmp_path,
        [
            ["a", "", "", "", "Studio/BTS", "", ""],
            ["b", "", "", "", "", "", ""],  # blank → skipped
            ["c", "", "", "", "Banger", "", ""],  # invalid → not written
        ],
    )
    r = labels.import_labels(csv, con=con, settings=SETTINGS)
    assert r["labeled"] == 1 and r["blank"] == 1 and r["invalid"] == 1
    assert r["invalid_values"] == ["Banger"]
    assert r["by_theme"] == {"Studio/BTS": 1}
    assert _theme(con, "a") == "Studio/BTS"
    assert _theme(con, "b") is None
    assert _theme(con, "c") is None  # invalid never landed


def test_import_is_idempotent_and_updates_on_rerun(con, tmp_path):
    _post(con, "a", datetime(2024, 1, 1, tzinfo=timezone.utc))
    csv = _csv(tmp_path, [["a", "", "", "", "Show/Tour", "", ""]])
    labels.import_labels(csv, con=con, settings=SETTINGS)
    labels.import_labels(csv, con=con, settings=SETTINGS)
    assert _theme(con, "a") == "Show/Tour"

    # Re-tagging and re-importing overwrites in place.
    _csv(tmp_path, [["a", "", "", "", "Studio/BTS", "", ""]])
    labels.import_labels(csv, con=con, settings=SETTINGS)
    assert _theme(con, "a") == "Studio/BTS"


def test_import_whitespace_theme_treated_as_blank(con, tmp_path):
    _post(con, "a", datetime(2024, 1, 1, tzinfo=timezone.utc))
    csv = _csv(tmp_path, [["a", "", "", "", "   ", "", ""]])
    r = labels.import_labels(csv, con=con, settings=SETTINGS)
    assert r["blank"] == 1 and r["labeled"] == 0
    assert _theme(con, "a") is None


def test_themes_constant_matches_taxonomy_doc():
    """The seven strings must match docs/theme-taxonomy.md exactly."""
    assert set(labels.THEMES) == {
        "Release/Promo", "Show/Tour", "Live/Performance", "Broadcast/Mix",
        "Studio/BTS", "Personal/Lifestyle", "Other",
    }
