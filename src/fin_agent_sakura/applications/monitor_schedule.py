"""Local daily drift-monitor schedule and one-shot check service."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fin_agent_sakura.applications.china_investment_assistant import ChinaInvestmentAssistant
from fin_agent_sakura.applications.rebalance_log import RebalanceEventLog
from fin_agent_sakura.monitoring import InMemoryPortfolioRepository, PortfolioMonitor
from fin_agent_sakura.storage import PositionMemory, SQLiteStore


DEFAULT_DAILY_MONITOR_SCHEDULE_PATH = Path("data/processed/daily_monitor_schedule.json")
DEFAULT_DAILY_MONITOR_RESULT_PATH = Path("data/processed/daily_monitor_latest.json")


@dataclass(frozen=True, slots=True)
class DailyMonitorSchedule:
    """User-configured local schedule for daily drift checks."""

    enabled: bool
    client_id: str = "paper_client"
    portfolio_id: str = "default"
    run_time: str = "09:30"
    drift_threshold: float = 0.05
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    last_run_at: str | None = None
    last_status: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "client_id": self.client_id,
            "portfolio_id": self.portfolio_id,
            "run_time": self.run_time,
            "drift_threshold": self.drift_threshold,
            "updated_at": self.updated_at,
            "last_run_at": self.last_run_at,
            "last_status": self.last_status,
            "warnings": self.warnings,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DailyMonitorSchedule":
        return cls(
            enabled=bool(payload.get("enabled", False)),
            client_id=str(payload.get("client_id") or "paper_client"),
            portfolio_id=str(payload.get("portfolio_id") or "default"),
            run_time=str(payload.get("run_time") or "09:30"),
            drift_threshold=float(payload.get("drift_threshold") or 0.05),
            updated_at=str(payload.get("updated_at") or datetime.now(UTC).isoformat()),
            last_run_at=payload.get("last_run_at"),
            last_status=payload.get("last_status"),
            warnings=list(payload.get("warnings") or []),
        )


@dataclass(frozen=True, slots=True)
class DailyMonitorCheckResult:
    """Result of one local daily drift-monitor run."""

    generated_at: str
    client_id: str
    portfolio_id: str
    status: str
    drift_threshold: float
    current_weights: dict[str, float]
    target_weights: dict[str, float]
    events: list[dict[str, Any]]
    warnings: list[str]
    log_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "client_id": self.client_id,
            "portfolio_id": self.portfolio_id,
            "status": self.status,
            "drift_threshold": self.drift_threshold,
            "current_weights": self.current_weights,
            "target_weights": self.target_weights,
            "events": self.events,
            "warnings": self.warnings,
            "log_path": self.log_path,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DailyMonitorCheckResult":
        return cls(
            generated_at=str(payload.get("generated_at") or ""),
            client_id=str(payload.get("client_id") or "paper_client"),
            portfolio_id=str(payload.get("portfolio_id") or "default"),
            status=str(payload.get("status") or "unknown"),
            drift_threshold=float(payload.get("drift_threshold") or 0.05),
            current_weights={str(k): float(v) for k, v in dict(payload.get("current_weights") or {}).items()},
            target_weights={str(k): float(v) for k, v in dict(payload.get("target_weights") or {}).items()},
            events=list(payload.get("events") or []),
            warnings=list(payload.get("warnings") or []),
            log_path=str(payload.get("log_path") or ""),
        )


def save_daily_monitor_schedule(
    *,
    enabled: bool,
    client_id: str = "paper_client",
    portfolio_id: str = "default",
    run_time: str = "09:30",
    drift_threshold: float = 0.05,
    warnings: list[str] | None = None,
    path: str | Path = DEFAULT_DAILY_MONITOR_SCHEDULE_PATH,
) -> DailyMonitorSchedule:
    """Persist the local daily monitor schedule."""

    schedule = DailyMonitorSchedule(
        enabled=enabled,
        client_id=client_id.strip() or "paper_client",
        portfolio_id=portfolio_id.strip() or "default",
        run_time=run_time.strip() or "09:30",
        drift_threshold=_validate_threshold(drift_threshold),
        warnings=warnings or [],
    )
    _write_json(Path(path), schedule.to_dict())
    SQLiteStore().save_state_record("daily_monitor_schedule", schedule.client_id, schedule.to_dict())
    return schedule


def load_daily_monitor_schedule(
    path: str | Path = DEFAULT_DAILY_MONITOR_SCHEDULE_PATH,
) -> DailyMonitorSchedule | None:
    """Load the latest local daily monitor schedule."""

    payload = _read_json(Path(path))
    if payload is None:
        return None
    return DailyMonitorSchedule.from_dict(payload)


def load_latest_daily_monitor_result(
    path: str | Path = DEFAULT_DAILY_MONITOR_RESULT_PATH,
) -> DailyMonitorCheckResult | None:
    """Load the latest one-shot monitor run result."""

    payload = _read_json(Path(path))
    if payload is None:
        return None
    return DailyMonitorCheckResult.from_dict(payload)


def run_daily_monitor_check_once(
    *,
    client_id: str = "paper_client",
    portfolio_id: str = "default",
    drift_threshold: float = 0.05,
    schedule_path: str | Path = DEFAULT_DAILY_MONITOR_SCHEDULE_PATH,
    result_path: str | Path = DEFAULT_DAILY_MONITOR_RESULT_PATH,
) -> DailyMonitorCheckResult:
    """Run one drift check using saved local positions and latest target weights."""

    result = asyncio.run(
        _run_daily_monitor_check_once_async(
            client_id=client_id,
            portfolio_id=portfolio_id,
            drift_threshold=drift_threshold,
        )
    )
    _write_json(Path(result_path), result.to_dict())
    _touch_schedule_after_run(Path(schedule_path), result)
    SQLiteStore().save_state_record("daily_monitor_result", client_id, result.to_dict())
    return result


async def _run_daily_monitor_check_once_async(
    *,
    client_id: str,
    portfolio_id: str,
    drift_threshold: float,
) -> DailyMonitorCheckResult:
    warnings: list[str] = []
    threshold = _validate_threshold(drift_threshold)
    assistant = ChinaInvestmentAssistant()
    latest = assistant.load_latest_result()
    target_weights = dict(latest.target_weights) if latest is not None else {}
    if not target_weights:
        warnings.append("没有找到最近一次投资方案目标权重，请先运行“投资方案生成”或“一键运行全流程”。")

    current_weights = PositionMemory().load_weights()
    if not current_weights and latest is not None:
        current_weights = dict(latest.current_weights)
        warnings.append("没有找到本地仓位 CSV，本次使用最近投资方案中的演示/历史当前权重。")
    if not current_weights:
        warnings.append("没有可用当前仓位，本次只能记录空检查结果。")

    repository = InMemoryPortfolioRepository(
        current_weights={client_id: current_weights},
        target_weights={client_id: target_weights},
    )
    monitor = PortfolioMonitor(repository, drift_threshold=threshold)
    raw_events = await monitor.check_client(client_id)
    generated_at = datetime.now(UTC).isoformat()
    event_rows = [
        {
            "generated_at": generated_at,
            "client_id": event.client_id,
            "portfolio_id": portfolio_id,
            "event_type": "scheduled_drift_alert",
            "ticker": event.ticker,
            "action": "sell" if event.direction == "overweight" else "buy",
            "execution_label": "每日检查待确认",
            "current_weight": event.current_weight,
            "target_weight": event.target_weight,
            "drift": event.drift,
            "threshold": event.threshold,
            "decision": "needs_rebalance",
            "reason": f"每日漂移检查发现绝对偏离 {abs(event.drift):.2%} 超过阈值 {event.threshold:.2%}",
        }
        for event in raw_events
    ]
    log_path = RebalanceEventLog().append_records(event_rows)
    status = "warning" if warnings else "success"
    if not target_weights or not current_weights:
        status = "failed"
    return DailyMonitorCheckResult(
        generated_at=generated_at,
        client_id=client_id,
        portfolio_id=portfolio_id,
        status=status,
        drift_threshold=threshold,
        current_weights=current_weights,
        target_weights=target_weights,
        events=event_rows,
        warnings=warnings,
        log_path=str(log_path),
    )


def _touch_schedule_after_run(path: Path, result: DailyMonitorCheckResult) -> None:
    schedule = load_daily_monitor_schedule(path)
    if schedule is None:
        schedule = DailyMonitorSchedule(
            enabled=False,
            client_id=result.client_id,
            portfolio_id=result.portfolio_id,
            drift_threshold=result.drift_threshold,
        )
    updated = DailyMonitorSchedule(
        enabled=schedule.enabled,
        client_id=schedule.client_id,
        portfolio_id=schedule.portfolio_id,
        run_time=schedule.run_time,
        drift_threshold=schedule.drift_threshold,
        updated_at=datetime.now(UTC).isoformat(),
        last_run_at=result.generated_at,
        last_status=result.status,
        warnings=schedule.warnings,
    )
    _write_json(path, updated.to_dict())


def _validate_threshold(value: float) -> float:
    threshold = float(value)
    if not 0 < threshold <= 1:
        raise ValueError("drift_threshold must satisfy 0 < threshold <= 1")
    return threshold


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
