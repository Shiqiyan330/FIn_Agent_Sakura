"""CSV persistence for rebalance drift and timing-rule events."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_REBALANCE_LOG_PATH = Path("data/processed/rebalance_events.csv")


@dataclass(frozen=True, slots=True)
class RebalanceEventLog:
    """Append-only local CSV log for paper rebalance decisions."""

    path: Path = DEFAULT_REBALANCE_LOG_PATH

    def append_records(self, records: list[dict[str, Any]]) -> Path:
        """Append normalized records to the CSV log and return the file path."""

        if not records:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if not self.path.exists():
                pd.DataFrame(columns=_LOG_COLUMNS).to_csv(self.path, index=False, encoding="utf-8-sig")
            return self.path

        self.path.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame([_normalize_record(record) for record in records])
        frame = frame.reindex(columns=_LOG_COLUMNS)
        if self.path.exists():
            existing = pd.read_csv(self.path)
            frame = pd.concat([existing, frame], ignore_index=True)
        frame.to_csv(self.path, index=False, encoding="utf-8-sig")
        return self.path

    def load(self) -> pd.DataFrame:
        """Load the local rebalance event log."""

        if not self.path.exists():
            return pd.DataFrame(columns=_LOG_COLUMNS)
        return pd.read_csv(self.path)


def append_rebalance_analysis_events(
    *,
    generated_at: str,
    drift_alerts: list[dict[str, Any]],
    trade_orders: list[dict[str, Any]],
    risk_gate: dict[str, Any] | None,
    client_id: str = "paper_client",
    path: str | Path = DEFAULT_REBALANCE_LOG_PATH,
) -> Path:
    """Persist drift alerts, deferred orders, executed paper orders, and risk decisions."""

    orders_by_ticker = {str(order.get("ticker", "")).upper(): order for order in trade_orders}
    risk_decision = str((risk_gate or {}).get("decision") or "not_run")
    records: list[dict[str, Any]] = []

    for alert in drift_alerts:
        ticker = str(alert.get("ticker", "")).upper()
        order = orders_by_ticker.get(ticker)
        records.append(
            {
                **alert,
                "generated_at": generated_at,
                "client_id": client_id,
                "event_type": "order" if order else "drift_alert",
                "action": (order or alert).get("action"),
                "target_weight_delta": (order or {}).get("target_weight_delta"),
                "suggested_batches": (order or {}).get("suggested_batches"),
                "batch_weight_delta": (order or {}).get("batch_weight_delta"),
                "risk_gate_decision": risk_decision,
                "execution_label": (order or alert).get("execution_label"),
                "reason": (order or alert).get("reason"),
            }
        )

    for ticker, order in orders_by_ticker.items():
        if any(str(alert.get("ticker", "")).upper() == ticker for alert in drift_alerts):
            continue
        records.append(
            {
                **order,
                "generated_at": generated_at,
                "client_id": client_id,
                "event_type": "order",
                "risk_gate_decision": risk_decision,
            }
        )

    return RebalanceEventLog(Path(path)).append_records(records)


def load_rebalance_event_log(path: str | Path = DEFAULT_REBALANCE_LOG_PATH) -> pd.DataFrame:
    """Load the latest local rebalance event CSV."""

    return RebalanceEventLog(Path(path)).load()


_LOG_COLUMNS = [
    "generated_at",
    "client_id",
    "event_type",
    "ticker",
    "action",
    "execution_label",
    "decision",
    "current_weight",
    "target_weight",
    "target_weight_delta",
    "drift",
    "threshold",
    "risk_gate_decision",
    "technical_signal",
    "technical_rsi_14",
    "technical_momentum",
    "price_above_ma_50",
    "price_above_ma_200",
    "sentiment_score",
    "sentiment_confidence",
    "suggested_batches",
    "batch_weight_delta",
    "reason",
    "reasons_json",
]


def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized = {column: record.get(column) for column in _LOG_COLUMNS}
    if "reasons" in record and not normalized.get("reasons_json"):
        normalized["reasons_json"] = json.dumps(record["reasons"], ensure_ascii=False, default=str)
    return normalized
