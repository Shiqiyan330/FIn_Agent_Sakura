"""Lightweight smoke-test service for GUI acceptance checks."""

from __future__ import annotations

import importlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from fin_agent_sakura.applications.full_workflow import load_latest_investment_run, run_full_advisory_workflow
from fin_agent_sakura.config import get_llm_config, get_tushare_config


SmokeStatus = Literal["success", "failed", "skipped"]
DEFAULT_SMOKE_REPORT_PATH = Path("data/processed/smoke_test_latest.json")


@dataclass(frozen=True, slots=True)
class SmokeTestItem:
    name: str
    status: SmokeStatus
    summary: str
    elapsed_seconds: float = 0.0
    error: str | None = None
    output: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SmokeTestReport:
    generated_at: str
    items: list[SmokeTestItem]

    @property
    def passed(self) -> bool:
        return not any(item.status == "failed" for item in self.items)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "passed": self.passed,
            "items": [item.to_dict() for item in self.items],
        }


def run_gui_smoke_tests(*, include_live_workflow: bool = False) -> SmokeTestReport:
    """Run low-cost checks for the Streamlit GUI and optional live workflow."""

    items = [
        _check_dashboard_import(),
        _check_llm_config(),
        _check_tushare_config(),
        _check_latest_outputs(),
    ]
    if include_live_workflow:
        items.append(_check_live_workflow())
    else:
        items.append(SmokeTestItem("一键全流程", "skipped", "默认跳过真实调用，避免消耗额度。"))

    report = SmokeTestReport(generated_at=pd.Timestamp.now().isoformat(), items=items)
    _save_report(report)
    return report


def load_latest_smoke_test_report(path: str | Path = DEFAULT_SMOKE_REPORT_PATH) -> SmokeTestReport | None:
    report_path = Path(path)
    if not report_path.exists():
        return None
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    return SmokeTestReport(
        generated_at=str(payload.get("generated_at", "")),
        items=[SmokeTestItem(**item) for item in payload.get("items", [])],
    )


def _check_dashboard_import() -> SmokeTestItem:
    start = time.perf_counter()
    try:
        importlib.import_module("fin_agent_sakura.dashboard.app")
        return SmokeTestItem("GUI import", "success", "Dashboard 模块可以正常导入。", time.perf_counter() - start)
    except Exception as exc:
        return _failed_item("GUI import", start, exc, "Dashboard 模块导入失败。")


def _check_llm_config() -> SmokeTestItem:
    start = time.perf_counter()
    cfg = get_llm_config()
    if not cfg.api_key:
        return SmokeTestItem("LLM配置", "failed", "未配置 OPENAI_API_KEY。", time.perf_counter() - start)
    return SmokeTestItem(
        "LLM配置",
        "success",
        f"已配置模型 {cfg.chat_model}。",
        time.perf_counter() - start,
        output={"chat_model": cfg.chat_model, "base_url": cfg.base_url, "embedding_model": cfg.embedding_model},
    )


def _check_tushare_config() -> SmokeTestItem:
    start = time.perf_counter()
    cfg = get_tushare_config()
    if not cfg.token:
        return SmokeTestItem("TuShare配置", "failed", "未配置 TUSHARE_TOKEN。", time.perf_counter() - start)
    return SmokeTestItem(
        "TuShare配置",
        "success",
        "已配置 TuShare Token。",
        time.perf_counter() - start,
        output={"http_url": cfg.http_url},
    )


def _check_latest_outputs() -> SmokeTestItem:
    start = time.perf_counter()
    paths = [
        "data/processed/a_share_universe_latest.json",
        "data/processed/china_investment_result.json",
        "data/processed/investment_run_latest.json",
        "data/processed/rebalance_events.csv",
    ]
    existing = [path for path in paths if Path(path).exists()]
    if len(existing) < 3:
        return SmokeTestItem(
            "最新产物",
            "failed",
            "关键产物不足，请先运行股票池或一键全流程。",
            time.perf_counter() - start,
            output={"existing": existing},
        )
    latest_run = load_latest_investment_run()
    return SmokeTestItem(
        "最新产物",
        "success",
        "关键产物文件已存在。",
        time.perf_counter() - start,
        output={"existing": existing, "latest_run": latest_run.run_id if latest_run else None},
    )


def _check_live_workflow() -> SmokeTestItem:
    start = time.perf_counter()
    try:
        result = run_full_advisory_workflow(
            profile_text="我是保守型投资者，期望跑赢通胀即可",
            max_candidates=5,
            selected_count=5,
            include_backtest=False,
            expert_mode=False,
        )
        return SmokeTestItem(
            "一键全流程",
            "success" if result.status == "success" else "failed",
            f"全流程运行完成：{result.status}。",
            time.perf_counter() - start,
            output={"run_id": result.run_id, "steps": [(step.name, step.status) for step in result.steps]},
        )
    except Exception as exc:
        return _failed_item("一键全流程", start, exc, "一键全流程运行失败。")


def _failed_item(name: str, start: float, exc: Exception, summary: str) -> SmokeTestItem:
    return SmokeTestItem(
        name=name,
        status="failed",
        summary=summary,
        elapsed_seconds=time.perf_counter() - start,
        error=f"{type(exc).__name__}: {exc}",
    )


def _save_report(report: SmokeTestReport, path: str | Path = DEFAULT_SMOKE_REPORT_PATH) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
