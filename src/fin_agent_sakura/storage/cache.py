"""Small file-based cache for provider responses and GUI services."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True, slots=True)
class CacheInfo:
    """Metadata for one cache entry."""

    key: str
    path: Path
    exists: bool
    age_seconds: float | None = None
    expired: bool | None = None


class CacheStore:
    """Simple JSON/CSV cache under data/processed/cache."""

    def __init__(self, root: str | Path = "data/processed/cache") -> None:
        self.root = Path(root)

    def get_json(self, key: str, ttl_seconds: int | float | None) -> Any | None:
        path = self._path(key, "json")
        if not self._is_fresh(path, ttl_seconds):
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def set_json(self, key: str, payload: Any) -> Path:
        path = self._path(key, "json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return path

    def get_dataframe(self, key: str, ttl_seconds: int | float | None) -> pd.DataFrame | None:
        path = self._path(key, "csv")
        if not self._is_fresh(path, ttl_seconds):
            return None
        return pd.read_csv(path)

    def set_dataframe(self, key: str, frame: pd.DataFrame) -> Path:
        path = self._path(key, "csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        return path

    def describe(self, key: str, suffix: str = "json", ttl_seconds: int | float | None = None) -> CacheInfo:
        path = self._path(key, suffix)
        if not path.exists():
            return CacheInfo(key=key, path=path, exists=False)
        age = max(0.0, time.time() - path.stat().st_mtime)
        expired = False if ttl_seconds is None else age > ttl_seconds
        return CacheInfo(key=key, path=path, exists=True, age_seconds=age, expired=expired)

    def _path(self, key: str, suffix: str) -> Path:
        safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", key).strip("_")[:180]
        return self.root / f"{safe_key}.{suffix}"

    def _is_fresh(self, path: Path, ttl_seconds: int | float | None) -> bool:
        if not path.exists():
            return False
        if ttl_seconds is None:
            return True
        return (time.time() - path.stat().st_mtime) <= ttl_seconds
