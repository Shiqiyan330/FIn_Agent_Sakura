"""SQLite persistence for local run records and audit summaries."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_SQLITE_PATH = Path("data/processed/sakura_state.sqlite")


@dataclass(frozen=True, slots=True)
class SQLiteStore:
    path: Path = DEFAULT_SQLITE_PATH

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS investment_runs (
                    run_id TEXT PRIMARY KEY,
                    generated_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    artifact_type TEXT NOT NULL,
                    path TEXT NOT NULL,
                    source_step TEXT NOT NULL,
                    generated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    feature TEXT NOT NULL,
                    model TEXT NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    estimated_cost_usd REAL NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS state_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    record_type TEXT NOT NULL,
                    record_key TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )

    def save_investment_run(self, run: Any) -> None:
        self.initialize()
        payload = run.to_dict()
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO investment_runs(run_id, generated_at, status, payload_json)
                VALUES (?, ?, ?, ?)
                """,
                (run.run_id, run.generated_at, run.status, json.dumps(payload, ensure_ascii=False, default=str)),
            )
            conn.execute("DELETE FROM artifacts WHERE run_id = ?", (run.run_id,))
            conn.executemany(
                """
                INSERT INTO artifacts(run_id, name, artifact_type, path, source_step, generated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        run.run_id,
                        artifact.name,
                        artifact.artifact_type,
                        artifact.path,
                        artifact.source_step,
                        artifact.generated_at,
                    )
                    for artifact in run.artifacts
                ],
            )

    def save_llm_usage_record(self, record: Any) -> None:
        self.initialize()
        payload = record.to_dict()
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                INSERT INTO llm_usage(created_at, feature, model, total_tokens, estimated_cost_usd, payload_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record.created_at,
                    record.feature,
                    record.model,
                    record.total_tokens,
                    record.estimated_cost_usd,
                    json.dumps(payload, ensure_ascii=False, default=str),
                ),
            )

    def table_counts(self) -> dict[str, int]:
        self.initialize()
        with sqlite3.connect(self.path) as conn:
            return {
                table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                for table in ["investment_runs", "artifacts", "llm_usage", "state_records"]
            }

    def save_state_record(self, record_type: str, record_key: str, payload: dict[str, Any]) -> None:
        self.initialize()
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                INSERT INTO state_records(record_type, record_key, created_at, payload_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    record_type,
                    record_key,
                    str(payload.get("generated_at") or payload.get("created_at") or pd.Timestamp.now().isoformat()),
                    json.dumps(payload, ensure_ascii=False, default=str),
                ),
            )

    def load_runs(self, limit: int = 50) -> pd.DataFrame:
        self.initialize()
        with sqlite3.connect(self.path) as conn:
            return pd.read_sql_query(
                "SELECT run_id, generated_at, status FROM investment_runs ORDER BY generated_at DESC LIMIT ?",
                conn,
                params=(limit,),
            )
