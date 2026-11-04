"""Collectors refuse to write unless the data source is real.

The synthetic DB is tracked by git, so real API data must never land in it.
"""

from __future__ import annotations

import httpx
import pytest

from insta_analytics.collect import account, backfill, demographics, land
from insta_analytics.config import Settings, db_path_for_source
from insta_analytics.storage import db


def _settings(**over) -> Settings:
    kw = dict(
        meta_app_id="111",
        meta_app_secret="shh",
        ig_access_token="LONG",
        ig_user_id="u1",
        data_source="synthetic",
    )
    kw.update(over)
    return Settings(**kw)


def _no_network() -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"collector made a request: {request.url}")

    return httpx.Client(base_url=backfill.client.GRAPH_BASE, transport=httpx.MockTransport(handler))


@pytest.fixture
def con():
    c = db.connect(":memory:")
    yield c
    c.close()


def _tables(con) -> int:
    return con.execute("SELECT count(*) FROM information_schema.tables").fetchone()[0]


COLLECTORS = {
    "land_media": lambda s, c: land.land_media([{"id": "a"}], con=c, settings=s, land_raw=False),
    "backfill": lambda s, c: backfill.backfill(
        con=c, settings=s, client_=_no_network(), sleep=0, land_raw=False
    ),
    "snapshot_account": lambda s, c: account.snapshot_account([], con=c, settings=s, land_raw=False),
    "snapshot_demographics": lambda s, c: demographics.snapshot_demographics(
        [], con=c, settings=s, land_raw=False
    ),
}


@pytest.mark.parametrize("name", COLLECTORS)
def test_collector_refuses_synthetic_source(name, con):
    with pytest.raises(RuntimeError, match="INSTA_DATA_SOURCE=real"):
        COLLECTORS[name](_settings(data_source="synthetic"), con)
    assert _tables(con) == 0  # refused before any DB write


@pytest.mark.parametrize("name", COLLECTORS)
def test_collector_refuses_real_source_pointed_at_synthetic_db(name, con):
    s = _settings(data_source="real", duckdb_path=db_path_for_source("synthetic"))
    with pytest.raises(RuntimeError, match="synthetic DB"):
        COLLECTORS[name](s, con)
    assert _tables(con) == 0
