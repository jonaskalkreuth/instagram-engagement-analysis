"""Land fetched media into DuckDB — mapping + end-to-end upsert."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from insta_analytics.collect import land
from insta_analytics.config import Settings
from insta_analytics.storage import db

COLLECTED_AT = datetime(2024, 6, 2, 10, 0, tzinfo=timezone.utc)

SETTINGS = Settings(
    meta_app_id="111",
    meta_app_secret="shh",
    ig_access_token="LONG",
    ig_user_id="u1",
    graph_api_version="v21.0",
    data_source="real",  # collectors refuse the synthetic source
)


def _media(post_id="p1", **over):
    m = {
        "id": post_id,
        "caption": "hello",
        "media_type": "IMAGE",
        "media_product_type": "FEED",
        "permalink": "http://x",
        "timestamp": "2024-06-01T12:34:56+0000",
        "like_count": 10,
        "comments_count": 2,
    }
    m.update(over)
    return m


@pytest.fixture
def con():
    c = db.connect(":memory:")
    db.bootstrap(c)
    yield c
    c.close()


def _count(con, table):
    return con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


# --- pure mapping ---------------------------------------------------------


def test_media_to_rows_maps_post_and_metric():
    posts, metrics = land.media_to_rows(
        [_media("p1")], ig_user_id="u1", collected_at=COLLECTED_AT
    )
    assert posts == [
        {
            "post_id": "p1",
            "ig_user_id": "u1",
            "media_type": "IMAGE",
            "media_product_type": "FEED",
            "caption": "hello",
            "permalink": "http://x",
            "timestamp": datetime(2024, 6, 1, 12, 34, 56, tzinfo=timezone.utc),
            "theme": None,
            "ingested_at": COLLECTED_AT,
        }
    ]
    assert metrics == [
        {
            "post_id": "p1",
            "collected_at": COLLECTED_AT,
            "like_count": 10,
            "comments_count": 2,
            # insight columns are present-but-NULL; the backfill fills them.
            "reach": None,
            "saved": None,
            "shares": None,
            "views": None,
            "total_interactions": None,
        }
    ]


def test_media_to_rows_missing_optionals_become_none():
    posts, metrics = land.media_to_rows(
        [{"id": "p1"}], ig_user_id="u1", collected_at=COLLECTED_AT
    )
    assert posts[0]["caption"] is None
    assert posts[0]["timestamp"] is None
    assert metrics[0]["like_count"] is None


# --- end-to-end land into DuckDB -----------------------------------------


def test_land_media_populates_both_tables(con):
    counts = land.land_media(
        [_media("a"), _media("b")], con=con, settings=SETTINGS, land_raw=False
    )
    assert counts == {"posts": 2, "post_metrics": 2}
    assert _count(con, "posts") == 2
    assert _count(con, "post_metrics") == 2
    # ig_user_id comes from settings, insights columns stay NULL.
    row = con.execute("SELECT ig_user_id FROM posts WHERE post_id='a'").fetchone()
    assert row[0] == "u1"
    assert con.execute("SELECT reach FROM post_metrics WHERE post_id='a'").fetchone()[0] is None


def test_land_media_timestamp_stored_as_timestamptz(con):
    land.land_media([_media("a")], con=con, settings=SETTINGS, land_raw=False)
    ts = con.execute("SELECT timestamp FROM posts WHERE post_id='a'").fetchone()[0]
    assert ts == datetime(2024, 6, 1, 12, 34, 56, tzinfo=timezone.utc)


def test_land_media_lands_raw_json(con, tmp_path):
    land.land_media(
        [_media("a")], con=con, settings=SETTINGS, raw_dir=str(tmp_path)
    )
    files = list(tmp_path.glob("media_*.json"))
    assert len(files) == 1


def test_land_media_rerun_keeps_theme_and_ingested_at(con):
    """A re-run must not erase a hand-labelled theme or the first-landed time."""
    land.land_media([_media("a")], con=con, settings=SETTINGS, land_raw=False)
    con.execute("UPDATE posts SET theme = 'Show/Tour' WHERE post_id = 'a'")
    first = con.execute("SELECT ingested_at FROM posts WHERE post_id='a'").fetchone()[0]

    land.land_media([_media("a", caption="edited")], con=con, settings=SETTINGS, land_raw=False)

    theme, ingested_at, caption = con.execute(
        "SELECT theme, ingested_at, caption FROM posts WHERE post_id='a'"
    ).fetchone()
    assert _count(con, "posts") == 1
    assert theme == "Show/Tour"
    assert ingested_at == first
    assert caption == "edited"
