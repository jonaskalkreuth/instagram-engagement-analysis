"""Typed application config, loaded from a git-ignored ``.env``.

Secrets (Meta app creds, IG token) live only in ``.env``; ``.env.example`` is the
committed template. Import config via :func:`get_settings`.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[2]

# DB path per data source. Real (private, git-ignored) vs synthetic (fake, tracked).
_DB_PATHS = {
    "real": _REPO_ROOT / "data" / "processed" / "insta.duckdb",
    "synthetic": _REPO_ROOT / "data" / "synthetic" / "insta.duckdb",
}
_DEFAULT_SOURCE = "synthetic"


def _env(name: str) -> str | None:
    """Read ``INSTA_<name>`` from the env or ``.env``, without validating ``Settings``."""
    key = f"INSTA_{name}"
    val = os.environ.get(key)
    if val is None:
        val = dotenv_values(_REPO_ROOT / ".env").get(key)
    return val or None


def db_path_for_source(source: str | None = None) -> Path:
    """Resolve the DuckDB path without loading secrets, defaulting to synthetic.

    ``INSTA_DUCKDB_PATH`` wins over ``source`` and ``INSTA_DATA_SOURCE``.
    """
    override = _env("DUCKDB_PATH")
    if override:
        return Path(override)
    source = source or _env("DATA_SOURCE") or _DEFAULT_SOURCE
    return _DB_PATHS[source]


class Settings(BaseSettings):
    """Runtime config from ``INSTA_*`` env vars or ``.env``."""

    model_config = SettingsConfigDict(
        env_prefix="INSTA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Secrets (required at runtime; see docs/setup-meta-auth.md) ---
    meta_app_id: str
    meta_app_secret: str
    ig_access_token: str
    ig_user_id: str

    # --- Non-secret knobs (safe defaults) ---
    graph_api_version: str = "v21.0"
    data_source: Literal["synthetic", "real"] = "synthetic"
    # ``None`` → resolved from ``data_source`` below. Set explicitly (arg or
    # ``INSTA_DUCKDB_PATH``) to override the source-based path.
    duckdb_path: Path | None = None
    raw_dir: Path = _REPO_ROOT / "data" / "raw"
    figures_dir: Path = _REPO_ROOT / "reports" / "figures"

    @model_validator(mode="after")
    def _resolve_db_path(self) -> "Settings":
        """Point ``duckdb_path`` at the source's DB unless one was set explicitly."""
        if "duckdb_path" not in self.model_fields_set or self.duckdb_path is None:
            self.duckdb_path = _DB_PATHS[self.data_source]
        return self


def require_real_source(settings: Settings) -> None:
    """Raise unless ``settings`` points at the real DB; git tracks the synthetic one."""
    if settings.data_source != "real":
        raise RuntimeError(
            "Collectors write real account data: set INSTA_DATA_SOURCE=real "
            f"(got {settings.data_source!r})."
        )
    if Path(settings.duckdb_path).resolve() == _DB_PATHS["synthetic"].resolve():
        raise RuntimeError(f"Collectors must not write to the synthetic DB: {settings.duckdb_path}")


@lru_cache
def get_settings() -> Settings:
    """Return cached :class:`Settings`, parsed once per process."""
    return Settings()  # type: ignore[call-arg]  # values come from env/.env
