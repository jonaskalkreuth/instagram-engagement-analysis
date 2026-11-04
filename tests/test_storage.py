"""Storage layer: bootstrap, idempotent upsert, connect, raw-JSON landing."""

import json
from datetime import datetime, timezone

import duckdb
import pytest

from insta_analytics.config import Settings
from insta_analytics.storage import db, schema


@pytest.fixture
def con():
    c = db.connect(":memory:")
    db.bootstrap(c)
    yield c
    c.close()


def _post(post_id="p1", **over):
    row = {
        "post_id": post_id,
        "ig_user_id": "u1",
        "media_type": "IMAGE",
        "media_product_type": "FEED",
        "caption": "hi",
        "permalink": "http://x",
        "timestamp": datetime(2024, 1, 1, tzinfo=timezone.utc),
        "theme": None,
        "ingested_at": datetime(2024, 1, 2, tzinfo=timezone.utc),
    }
    row.update(over)
    return row


def _count(con, table):
    return con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


def test_bootstrap_creates_all_tables(con):
    names = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    assert names == set(schema.TABLES)


def test_upsert_inserts_rows(con):
    n = db.upsert(con, "posts", [_post("a"), _post("b")])
    assert n == 2
    assert _count(con, "posts") == 2


def test_upsert_updates_non_key_column_in_place(con):
    db.upsert(con, "posts", [_post("a", caption="first")])
    db.upsert(con, "posts", [_post("a", caption="second")])
    assert _count(con, "posts") == 1
    assert con.execute("SELECT caption FROM posts").fetchone()[0] == "second"


def test_upsert_do_nothing_when_all_columns_are_keys(con):
    # account_snapshots row written with only its dedup keys -> conflict = DO NOTHING.
    keys_only = {"ig_user_id": "u1", "date": "2024-01-01", "metric": "reach"}
    db.upsert(con, "account_snapshots", [keys_only])
    db.upsert(con, "account_snapshots", [keys_only])
    assert _count(con, "account_snapshots") == 1


def test_upsert_empty_is_noop(con):
    assert db.upsert(con, "posts", []) == 0


def test_upsert_unknown_table_raises(con):
    with pytest.raises(KeyError):
        db.upsert(con, "nope", [{"x": 1}])


def test_upsert_missing_dedup_key_raises(con):
    with pytest.raises(ValueError):
        db.upsert(con, "posts", [{"ig_user_id": "u1"}])  # no post_id


def test_connect_creates_parent_dir(tmp_path):
    target = tmp_path / "nested" / "insta.duckdb"
    c = db.connect(target)
    c.close()
    assert target.parent.is_dir()
    assert target.exists()


def test_connection_leaves_passed_con_open(con):
    with db.connection(con) as c:
        assert c is con
    assert _count(con, "posts") == 0


def test_connection_closes_the_con_it_opens(tmp_path):
    settings = Settings(
        meta_app_id="1", meta_app_secret="s", ig_access_token="t", ig_user_id="u1",
        duckdb_path=tmp_path / "insta.duckdb",
    )
    with db.connection(settings=settings) as c:
        c.execute("SELECT 1")
    with pytest.raises(duckdb.ConnectionException):
        c.execute("SELECT 1")


def test_land_raw_writes_readable_json(tmp_path):
    payload = {"data": [{"id": "1", "café": "ok"}]}
    out = db.land_raw(payload, "media", raw_dir=tmp_path)
    assert out.exists()
    assert out.name.startswith("media_") and out.suffix == ".json"
    assert json.loads(out.read_text(encoding="utf-8")) == payload
