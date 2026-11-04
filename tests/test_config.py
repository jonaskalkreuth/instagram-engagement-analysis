"""Config loading: fields parse from env, sensible defaults, no real secrets."""

from pathlib import Path

import pytest

from insta_analytics.config import Settings, db_path_for_source, get_settings

_SECRETS = {
    "INSTA_META_APP_ID": "app123",
    "INSTA_META_APP_SECRET": "secret123",
    "INSTA_IG_ACCESS_TOKEN": "token123",
    "INSTA_IG_USER_ID": "user123",
}


def _set_env(monkeypatch, **extra):
    for k, v in {**_SECRETS, **extra}.items():
        monkeypatch.setenv(k, v)


def test_loads_secrets_from_env(monkeypatch):
    _set_env(monkeypatch)
    s = Settings(_env_file=None)  # ignore any real .env on disk
    assert s.ig_access_token == "token123"
    assert s.meta_app_id == "app123"


def test_defaults(monkeypatch):
    _set_env(monkeypatch)
    s = Settings(_env_file=None)
    assert s.data_source == "synthetic"
    assert s.graph_api_version == "v21.0"
    assert s.duckdb_path.parent.name == "synthetic"
    assert s.duckdb_path.name == "insta.duckdb"
    assert isinstance(s.raw_dir, Path)


def test_db_path_follows_real_source(monkeypatch):
    _set_env(monkeypatch, INSTA_DATA_SOURCE="real")
    s = Settings(_env_file=None)
    assert s.duckdb_path.parent.name == "processed"


def test_explicit_db_path_overrides_source(monkeypatch):
    _set_env(monkeypatch, INSTA_DATA_SOURCE="synthetic", INSTA_DUCKDB_PATH="/tmp/custom.duckdb")
    s = Settings(_env_file=None)
    assert s.duckdb_path == Path("/tmp/custom.duckdb")


def test_db_path_for_source_selects_by_env(monkeypatch):
    """The secret-free resolver picks the path from INSTA_DATA_SOURCE, no secrets."""
    monkeypatch.delenv("INSTA_DUCKDB_PATH", raising=False)
    monkeypatch.setenv("INSTA_DATA_SOURCE", "real")
    assert db_path_for_source().parent.name == "processed"
    monkeypatch.setenv("INSTA_DATA_SOURCE", "synthetic")
    assert db_path_for_source().parent.name == "synthetic"


def test_db_path_for_source_arg_beats_env(monkeypatch):
    monkeypatch.delenv("INSTA_DUCKDB_PATH", raising=False)
    monkeypatch.setenv("INSTA_DATA_SOURCE", "synthetic")
    assert db_path_for_source("real").parent.name == "processed"


def test_db_path_for_source_override_wins(monkeypatch):
    monkeypatch.setenv("INSTA_DATA_SOURCE", "real")
    monkeypatch.setenv("INSTA_DUCKDB_PATH", "/tmp/custom.duckdb")
    assert db_path_for_source() == Path("/tmp/custom.duckdb")


def test_data_source_rejects_bad_value(monkeypatch):
    _set_env(monkeypatch, INSTA_DATA_SOURCE="bogus")
    with pytest.raises(ValueError):
        Settings(_env_file=None)


def test_get_settings_is_cached(monkeypatch):
    _set_env(monkeypatch)
    get_settings.cache_clear()
    assert get_settings() is get_settings()
