"""Schema DDL is valid and the dedup manifest matches the real tables."""

import duckdb
import pytest

from insta_analytics.storage import schema

_EXPECTED_COLUMNS = {
    "posts": {
        "post_id",
        "ig_user_id",
        "media_type",
        "media_product_type",
        "caption",
        "permalink",
        "timestamp",
        "theme",
        "ingested_at",
    },
    "post_metrics": {
        "post_id",
        "collected_at",
        "like_count",
        "comments_count",
        "reach",
        "saved",
        "shares",
        "views",
        "total_interactions",
    },
    "account_snapshots": {
        "ig_user_id",
        "date",
        "metric",
        "value",
        "collected_at",
    },
    "follower_demographics": {
        "ig_user_id",
        "snapshot_date",
        "breakdown",
        "dimension",
        "value",
        "collected_at",
    },
}


@pytest.fixture
def conn():
    con = duckdb.connect(":memory:")
    for ddl in schema.ALL_DDL:
        con.execute(ddl)
    yield con
    con.close()


@pytest.mark.parametrize("table", _EXPECTED_COLUMNS)
def test_columns_present(conn, table):
    cols = {r[1] for r in conn.execute(f"PRAGMA table_info('{table}')").fetchall()}
    assert cols == _EXPECTED_COLUMNS[table]


def test_ddl_is_idempotent(conn):
    # CREATE TABLE IF NOT EXISTS — re-running the DDL must not raise.
    for ddl in schema.ALL_DDL:
        conn.execute(ddl)


@pytest.mark.parametrize("table", schema.TABLES)
def test_dedup_key_enforced_as_primary_key(conn, table):
    # Inserting two rows with the same dedup key must violate the PK.
    info = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
    pk_cols = {r[1] for r in info if r[5]}  # r[5] = pk flag
    assert pk_cols == set(schema.TABLES[table]["dedup_keys"])
