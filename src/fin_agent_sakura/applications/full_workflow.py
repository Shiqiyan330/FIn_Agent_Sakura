"""One-click advisory workflow orchestration for the Streamlit control center."""

from __future__ import annotations

import json
import time
import uuid
import asyncio
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from fin_agent_sakura.applications.a_share_universe import build_a_share_universe, load_latest_a_share_universe
from fin_agent_sakura.applications.backtest_service import load_latest_backtest_report, run_a_share_backtest
from fin_agent_sakura.applications.china_investment_assistant import ChinaInvestmentAssistant, StockCandidate
from fin_agent_sakura.applications.client_profile import load_latest_client_profile
from fin_agent_sakura.applications.data_source_health import run_a_share_data_source_health_check
from fin_agent_sakura.applications.rebalance_log import DEFAULT_REBALANCE_LOG_PATH, load_rebalance_event_log
from fin_agent_sakura.config import get_llm_config, get_tushare_config
from fin_agent_sakura.storage import PositionMemory, SQLiteStore


RunStepStatus = Literal["pending", "running", "success", "failed", "skipped"]

DEFAULT_RUN_DIR = Path("data/processed/investment_runs")
DEFAULT_LATEST_RUN_PATH = Path("data/processed/investment_run_latest.json")


@dataclass(frozen=True, slots=True)
class InvestmentRunStep:
    """Status record for one one-click workflow step."""

    name: str
    status: RunStepStatus
    summary: str
    elapsed_seconds: float = 0.0
    error: str | None = None
    output: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class InvestmentArtifact:
    """Downloadable artifact produced by a workflow run."""

    name: str
    artifact_type: str
    path: str
    source_step: str
    generated_at: str
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class InvestmentRun:
    """Persistable one-click advisory workflow result."""

    run_id: str
    generated_at: str
    profile_text: str
    max_candidates: int
    selected_count: int
    include_backtest: bool
    expert_mode: bool
    steps: list[InvestmentRunStep]
    artifacts: list[InvestmentArtifact]
    warnings: list[str] = field(default_factory=list)

    @property
    def status(self) -> RunStepStatus:
        if any(step.status == "failed" for step in self.steps if step.name in {"投资方案", "风险断路器"}):
            return "failed"
        if any(step.status == "success" for step in self.steps):
            return "success"
        return "skipped"

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "generated_at": self.generated_at,
            "profile_text": self.profile_text,
            "max_candidates": self.max_candidates,
            "selected_count": self.selected_count,
            "include_backtest": self.include_backtest,
            "expert_mode": self.expert_mode,
            "status": self.status,
            "steps": [step.to_dict() for step in self.steps],
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "warnings": self.warnings,
        }


def run_full_advisory_workflow(
    profile_text: str | None = None,
    *,
    max_candidates: int = 30,
    selected_count: int = 10,
    include_backtest: bool = False,
    expert_mode: bool = False,
    output_dir: str | Path = DEFAULT_RUN_DIR,
) -> InvestmentRun:
    """Run the GUI one-click A-share advisory workflow and persist a run record."""

    run_id = uuid.uuid4().hex[:12]
    generated_at = pd.Timestamp.now().isoformat()
    output_path = Path(output_dir) / run_id
    output_path.mkdir(parents=True, exist_ok=True)

    steps: list[InvestmentRunStep] = []
    artifacts: list[InvestmentArtifact] = []
    warnings: list[str] = []

    latest_profile = load_latest_client_profile()
    resolved_profile = (
        profile_text
        or (latest_profile.natural_language_profile if latest_profile is not None else "")
        or "我是保守型投资者，期望跑赢通胀即可"
    )

    steps.append(_run_llm_smoke())
    data_step = _run_data_source_smoke()
    steps.append(data_step)

    universe_step, universe = _run_universe_step(max_candidates)
    steps.append(universe_step)
    if universe is not None:
        artifacts.append(
            _artifact(
                name="股票池 JSON",
                artifact_type="json",
                path="data/processed/a_share_universe_latest.json",
                source_step="股票池",
                description="最近一次 A 股股票池选择器结果。",
            )
        )

    positions = PositionMemory()
    current_weights = positions.load_weights()
    if not current_weights:
        warnings.append("没有检测到本地仓位 CSV，投资方案使用演示仓位完成漂移监控。")

    investment_step, investment_result = _run_investment_step(
        profile_text=resolved_profile,
        max_candidates=max_candidates,
        selected_count=selected_count,
        current_weights=current_weights or None,
        universe=universe,
    )
    steps.append(investment_step)
    if investment_result is not None:
        artifacts.extend(_write_investment_artifacts(investment_result, output_path, generated_at))

    risk_step = _risk_step_from_investment(investment_result)
    steps.append(risk_step)
    if investment_result is not None and investment_result.risk_gate is not None and Path("data/processed/risk_gate_latest.json").exists():
        artifacts.append(
            _artifact(
                name="风险断路器报告 JSON",
                artifact_type="json",
                path="data/processed/risk_gate_latest.json",
                source_step="风险断路器",
                description="订单最终出口的硬编码风控评估。",
            )
        )

    drift_step = _run_drift_log_step()
    steps.append(drift_step)
    if DEFAULT_REBALANCE_LOG_PATH.exists():
        artifacts.append(
            _artifact(
                name="再平衡事件 CSV",
                artifact_type="csv",
                path=str(DEFAULT_REBALANCE_LOG_PATH),
                source_step="漂移日志",
                description="历史漂移、择时规则、风控标签事件日志。",
            )
        )

    backtest_report = None
    if include_backtest:
        backtest_step, backtest_report = _run_backtest_step(universe)
        steps.append(backtest_step)
        if backtest_report is not None:
            artifacts.extend(_write_backtest_artifacts(backtest_report, output_path, generated_at))
    else:
        steps.append(
            InvestmentRunStep(
                name="回测快照",
                status="skipped",
                summary="已按默认成本控制跳过回测；可在 GUI 勾选后运行。",
            )
        )
        latest_backtest = load_latest_backtest_report()
        if latest_backtest is not None and Path("data/processed/backtest_latest.json").exists():
            artifacts.append(
                _artifact(
                    name="最近一次回测报告 JSON",
                    artifact_type="json",
                    path="data/processed/backtest_latest.json",
                    source_step="回测快照",
                    description="非本次运行生成，来自最近一次历史回测。",
                )
            )
            if Path("data/processed/backtest_latest.html").exists():
                artifacts.append(
                    _artifact(
                        name="最近一次回测报告 HTML",
                        artifact_type="html",
                        path="data/processed/backtest_latest.html",
                        source_step="回测快照",
                        description="非本次运行生成，来自最近一次历史回测。",
                    )
                )

    if include_backtest and Path("data/processed/backtest_latest.html").exists():
            artifacts.append(
                _artifact(
                    name="回测报告 HTML",
                    artifact_type="html",
                    path="data/processed/backtest_latest.html",
                    source_step="回测快照",
                    description="本次快速回测 HTML 报告。",
                )
            )

    run = InvestmentRun(
        run_id=run_id,
        generated_at=generated_at,
        profile_text=resolved_profile,
        max_candidates=max_candidates,
        selected_count=selected_count,
        include_backtest=include_backtest,
        expert_mode=expert_mode,
        steps=steps,
        artifacts=artifacts,
        warnings=warnings,
    )
    _save_run(run, output_path=output_path)
    store = SQLiteStore()
    store.save_investment_run(run)
    store.save_state_record("client_profile", run.run_id, {"generated_at": generated_at, "profile_text": resolved_profile})
    if investment_result is not None:
        result_payload = investment_result.to_dict()
        store.save_state_record("target_weights", run.run_id, {"generated_at": generated_at, "target_weights": result_payload.get("target_weights", {})})
        store.save_state_record("positions", run.run_id, {"generated_at": generated_at, "current_weights": result_payload.get("current_weights", {})})
        store.save_state_record("drift_events", run.run_id, {"generated_at": generated_at, "drift_alerts": result_payload.get("drift_alerts", [])})
        if result_payload.get("risk_gate"):
            store.save_state_record("risk_gate", run.run_id, {"generated_at": generated_at, "risk_gate": result_payload["risk_gate"]})
    if include_backtest and backtest_report is not None:
        store.save_state_record("backtest", run.run_id, backtest_report.to_dict())
    return run


def load_latest_investment_run(path: str | Path = DEFAULT_LATEST_RUN_PATH) -> InvestmentRun | None:
    """Load the latest one-click workflow run."""

    run_path = Path(path)
    if not run_path.exists():
        return None
    payload = json.loads(run_path.read_text(encoding="utf-8"))
    return _run_from_payload(payload)


def list_investment_run_artifacts(run_id: str | None = None) -> list[InvestmentArtifact]:
    """Return artifacts for a specific run or the latest run."""

    if run_id:
        path = DEFAULT_RUN_DIR / run_id / "investment_run.json"
        if not path.exists():
            return []
        run = _run_from_payload(json.loads(path.read_text(encoding="utf-8")))
    else:
        run = load_latest_investment_run()
    return [] if run is None else run.artifacts


def _run_llm_smoke() -> InvestmentRunStep:
    start = time.perf_counter()
    cfg = get_llm_config()
    if not cfg.api_key:
        return InvestmentRunStep(
            name="LLM检查",
            status="failed",
            summary="未配置 LLM API Key，后续步骤将使用本地兜底逻辑。",
            elapsed_seconds=time.perf_counter() - start,
            error="OPENAI_API_KEY is missing",
            output={"chat_model": cfg.chat_model, "base_url": cfg.base_url},
        )
    return InvestmentRunStep(
        name="LLM检查",
        status="success",
        summary=f"已检测到模型配置：{cfg.chat_model}。",
        elapsed_seconds=time.perf_counter() - start,
        output={"chat_model": cfg.chat_model, "base_url": cfg.base_url, "embedding_model": cfg.embedding_model},
    )


def _run_data_source_smoke() -> InvestmentRunStep:
    start = time.perf_counter()
    try:
        report = asyncio.run(
            run_a_share_data_source_health_check(
                price_ticker="000001.SZ",
                statement_ticker="600519.SH",
                start_date="2018-07-01",
                end_date="2018-07-18",
                mode="auto",
            )
        )
        status: RunStepStatus = "success" if report.is_healthy else "failed"
        return InvestmentRunStep(
            name="A股数据源",
            status=status,
            summary="A股行情与财报数据源检查完成。" if report.is_healthy else "A股数据源存在失败项，后续步骤会尽量兜底。",
            elapsed_seconds=time.perf_counter() - start,
            output={
                "is_healthy": report.is_healthy,
                "checks": len(report.checks),
                "tushare_http_url": report.tushare_http_url,
            },
        )
    except Exception as exc:
        return _failed_step("A股数据源", start, exc, "A股数据源检查失败，后续步骤会尝试使用缓存或离线兜底。")


def _run_universe_step(max_candidates: int) -> tuple[InvestmentRunStep, list[StockCandidate] | None]:
    start = time.perf_counter()
    try:
        latest = load_latest_a_share_universe()
        result = latest or build_a_share_universe(["沪深300"], max_count=max_candidates, force_refresh=False)
        universe = [
            StockCandidate(ticker=item.ticker, name=item.name or item.ticker, sector=item.industry or item.board or "未知")
            for item in result.items[:max_candidates]
        ]
        return (
            InvestmentRunStep(
                name="股票池",
                status="success",
                summary=f"已准备 {len(universe)} 只 A 股候选池。",
                elapsed_seconds=time.perf_counter() - start,
                output={
                    "sources": result.sources,
                    "count": len(universe),
                    "cache_hit": result.cache_hit,
                    "warnings": result.warnings,
                },
            ),
            universe,
        )
    except Exception as exc:
        return (_failed_step("股票池", start, exc, "股票池构建失败，投资方案会使用内置核心股票池。"), None)


def _run_investment_step(
    *,
    profile_text: str,
    max_candidates: int,
    selected_count: int,
    current_weights: dict[str, float] | None,
    universe: list[StockCandidate] | None,
) -> tuple[InvestmentRunStep, Any | None]:
    start = time.perf_counter()
    try:
        result = _run_async_investment(
            profile_text=profile_text,
            max_candidates=max_candidates,
            selected_count=selected_count,
            current_weights=current_weights,
            universe=universe,
        )
        return (
            InvestmentRunStep(
                name="投资方案",
                status="success",
                summary=f"已生成 {len(result.selected)} 只目标持仓、{len(result.trade_orders)} 条纸面订单。",
                elapsed_seconds=time.perf_counter() - start,
                output={
                    "mode": result.mode,
                    "selected": len(result.selected),
                    "drift_alerts": len(result.drift_alerts),
                    "trade_orders": len(result.trade_orders),
                    "warnings": result.warnings,
                },
            ),
            result,
        )
    except Exception as exc:
        return (_failed_step("投资方案", start, exc, "投资方案生成失败，请先检查 A 股数据源和股票池。"), None)


def _run_async_investment(**kwargs: Any) -> Any:
    return asyncio.run(ChinaInvestmentAssistant().run(**kwargs, use_llm_report=False))


def _risk_step_from_investment(result: Any | None) -> InvestmentRunStep:
    if result is None:
        return InvestmentRunStep(name="风险断路器", status="skipped", summary="投资方案未生成，跳过风险断路器。")
    risk_gate = result.risk_gate
    if not risk_gate:
        return InvestmentRunStep(name="风险断路器", status="skipped", summary="没有纸面订单，风险断路器无需评估。")
    approved = risk_gate.get("decision") == "approved"
    return InvestmentRunStep(
        name="风险断路器",
        status="success" if approved else "failed",
        summary="风险断路器批准纸面订单，真实交易前仍需人工确认。" if approved else "风险断路器拒绝本批纸面订单。",
        output={
            "decision": risk_gate.get("decision"),
            "portfolio_var": risk_gate.get("portfolio_var"),
            "max_drawdown": risk_gate.get("max_drawdown"),
            "alerts": len(risk_gate.get("alerts") or []),
        },
    )


def _run_drift_log_step() -> InvestmentRunStep:
    start = time.perf_counter()
    try:
        log = load_rebalance_event_log()
        return InvestmentRunStep(
            name="漂移日志",
            status="success",
            summary=f"已读取 {len(log)} 条再平衡事件日志。",
            elapsed_seconds=time.perf_counter() - start,
            output={"rows": len(log), "path": str(DEFAULT_REBALANCE_LOG_PATH)},
        )
    except Exception as exc:
        return _failed_step("漂移日志", start, exc, "漂移日志读取失败。")


def _run_backtest_step(universe: list[StockCandidate] | None) -> tuple[InvestmentRunStep, Any | None]:
    start = time.perf_counter()
    try:
        tickers = [item.ticker for item in (universe or [])][:5]
        if len(tickers) < 2:
            tickers = ["600519.SH", "000858.SZ", "000333.SZ", "300750.SZ", "600036.SH"]
        report = run_a_share_backtest(
            tickers=tickers,
            market="cn",
            strategy="momentum_top_n",
            start_date="2023-01-01",
            rebalance_frequency_days=21,
            top_n=min(3, len(tickers)),
        )
        return (
            InvestmentRunStep(
                name="回测快照",
                status="success",
                summary=f"已完成 {len(tickers)} 只股票的快速回测快照。",
                elapsed_seconds=time.perf_counter() - start,
                output={
                    "tickers": tickers,
                    "cumulative_return": report.cumulative_return,
                    "sharpe_ratio": report.sharpe_ratio,
                    "max_drawdown": report.max_drawdown,
                },
            ),
            report,
        )
    except Exception as exc:
        return (_failed_step("回测快照", start, exc, "回测快照失败，不影响投顾方案和风险报告。"), None)


def _write_investment_artifacts(result: Any, output_path: Path, generated_at: str) -> list[InvestmentArtifact]:
    artifacts: list[InvestmentArtifact] = []

    report_path = output_path / "research_report.html"
    report_path.write_text(str(result.research_report_html or ""), encoding="utf-8")
    artifacts.append(_artifact("持仓研报 HTML", "html", report_path, "投资方案", "LLM或本地模板生成的持仓研报。", generated_at))

    weights_path = output_path / "target_weights.csv"
    pd.DataFrame(
        [{"ticker": ticker, "target_weight": weight} for ticker, weight in result.target_weights.items()]
    ).to_csv(weights_path, index=False, encoding="utf-8-sig")
    artifacts.append(_artifact("目标权重 CSV", "csv", weights_path, "投资方案", "本次纸面投顾目标权重。", generated_at))

    orders_path = output_path / ("paper_trade_orders.csv" if result.risk_gate and result.risk_gate.get("decision") == "approved" else "research_only_orders.csv")
    pd.DataFrame(result.trade_orders).to_csv(orders_path, index=False, encoding="utf-8-sig")
    artifacts.append(_artifact("纸面订单 CSV" if result.risk_gate and result.risk_gate.get("decision") == "approved" else "研究记录订单 CSV", "csv", orders_path, "投资方案", "纸面订单，不连接券商下单。", generated_at))

    return artifacts


def _write_backtest_artifacts(report: Any, output_path: Path, generated_at: str) -> list[InvestmentArtifact]:
    path = output_path / "backtest_equity_curve.csv"
    strategy = pd.DataFrame(report.equity_curve)
    benchmark = pd.DataFrame(report.benchmark_curve)
    curve = strategy.merge(benchmark, on="date", how="outer") if not strategy.empty and not benchmark.empty else strategy
    curve.to_csv(path, index=False, encoding="utf-8-sig")
    return [_artifact("回测曲线 CSV", "csv", path, "回测快照", "本次快速回测净值曲线。", generated_at)]


def _artifact(
    name: str,
    artifact_type: str,
    path: str | Path,
    source_step: str,
    description: str = "",
    generated_at: str | None = None,
) -> InvestmentArtifact:
    return InvestmentArtifact(
        name=name,
        artifact_type=artifact_type,
        path=str(path),
        source_step=source_step,
        generated_at=generated_at or pd.Timestamp.now().isoformat(),
        description=description,
    )


def _failed_step(name: str, start: float, exc: Exception, summary: str) -> InvestmentRunStep:
    return InvestmentRunStep(
        name=name,
        status="failed",
        summary=summary,
        elapsed_seconds=time.perf_counter() - start,
        error=f"{type(exc).__name__}: {exc}",
    )


def _save_run(run: InvestmentRun, *, output_path: Path) -> None:
    payload = json.dumps(run.to_dict(), ensure_ascii=False, indent=2, default=str)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "investment_run.json").write_text(payload, encoding="utf-8")
    DEFAULT_LATEST_RUN_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_LATEST_RUN_PATH.write_text(payload, encoding="utf-8")


def _run_from_payload(payload: dict[str, Any]) -> InvestmentRun:
    return InvestmentRun(
        run_id=str(payload["run_id"]),
        generated_at=str(payload["generated_at"]),
        profile_text=str(payload.get("profile_text") or ""),
        max_candidates=int(payload.get("max_candidates", 30)),
        selected_count=int(payload.get("selected_count", 10)),
        include_backtest=bool(payload.get("include_backtest", False)),
        expert_mode=bool(payload.get("expert_mode", False)),
        steps=[InvestmentRunStep(**step) for step in payload.get("steps", [])],
        artifacts=[InvestmentArtifact(**artifact) for artifact in payload.get("artifacts", [])],
        warnings=list(payload.get("warnings") or []),
    )
