"""Connect, create tables, upsert rows and land raw JSON."""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import duckdb

from ..config import Settings, get_settings
from . import schema

Row = Mapping[str, Any]


def connect(path: str | Path | None = None) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection (default: settings path), creating the parent dir."""
    if path is None:
        path = get_settings().duckdb_path
    if path != ":memory:":
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path = str(path)
    return duckdb.connect(path)


@contextmanager
def connection(
    con: duckdb.DuckDBPyConnection | None = None, settings: Settings | None = None
) -> Iterator[duckdb.DuckDBPyConnection]:
    """Yield ``con`` as is, or open one from settings and close it afterwards."""
    if con is not None:
        yield con
        return
    con = connect((settings or get_settings()).duckdb_path)
    try:
        yield con
    finally:
        con.close()


def bootstrap(con: duckdb.DuckDBPyConnection) -> None:
    """Create every table if absent. Idempotent — safe to call on each run."""
    for ddl in schema.ALL_DDL:
        con.execute(ddl)


def upsert(con: duckdb.DuckDBPyConnection, table: str, rows: Sequence[Row]) -> int:
    """Upsert ``rows`` on the table's dedup keys and return how many were submitted.

    Existing rows keep their ``keep_on_conflict`` columns.
    """
    if table not in schema.TABLES:
        raise KeyError(f"unknown table {table!r}; expected one of {list(schema.TABLES)}")
    rows = list(rows)
    if not rows:
        return 0

    keys: tuple[str, ...] = schema.TABLES[table]["dedup_keys"]  # type: ignore[assignment]
    cols = list(rows[0].keys())
    if any(k not in cols for k in keys):
        raise ValueError(f"rows for {table!r} miss dedup key(s) {set(keys) - set(cols)}")

    placeholders = ", ".join("?" for _ in cols)
    col_list = ", ".join(cols)
    keep: tuple[str, ...] = schema.TABLES[table].get("keep_on_conflict", ())  # type: ignore[assignment]
    non_key = [c for c in cols if c not in keys and c not in keep]
    conflict_target = ", ".join(keys)
    action = (
        "DO NOTHING"
        if not non_key
        else "DO UPDATE SET " + ", ".join(f"{c} = excluded.{c}" for c in non_key)
    )
    sql = (
        f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) "
        f"ON CONFLICT ({conflict_target}) {action}"
    )
    con.executemany(sql, [[r[c] for c in cols] for r in rows])
    return len(rows)


def land_raw(payload: Any, label: str, raw_dir: str | Path | None = None) -> Path:
    """Write an API ``payload`` to ``{raw_dir}/{label}_{UTC-timestamp}.json``."""
    if raw_dir is None:
        raw_dir = get_settings().raw_dir
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = raw_dir / f"{label}_{stamp}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
