"""A-share data source health checks for GUI diagnostics."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import pandas as pd

from fin_agent_sakura.config import get_tushare_config
from fin_agent_sakura.data import CNMarketDataClient, DataProviderError, MarketDataClientFactory


DataSourceMode = Literal["auto", "tushare", "akshare"]

DEFAULT_HEALTH_REPORT_PATH = Path("data/processed/a_share_data_source_health_latest.json")


@dataclass(frozen=True, slots=True)
class DataSourceCheckResult:
    """One provider/interface health-check row."""

    interface: str
    ticker: str
    provider: str
    status: Literal["success", "failed", "skipped"]
    row_count: int
    elapsed_seconds: float | None = None
    error: str | None = None
    columns: list[str] = field(default_factory=list)
    preview: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface": self.interface,
            "ticker": self.ticker,
            "provider": self.provider,
            "status": self.status,
            "row_count": self.row_count,
            "elapsed_seconds": self.elapsed_seconds,
            "error": self.error,
            "columns": self.columns,
            "preview": self.preview,
        }


@dataclass(frozen=True, slots=True)
class DataSourceHealthReport:
    """Result object persisted after an A-share data source health check."""

    generated_at: str
    mode: DataSourceMode
    price_ticker: str
    statement_ticker: str
    start_date: str
    end_date: str
    tushare_configured: bool
    tushare_http_url: str | None
    checks: list[DataSourceCheckResult]
    warnings: list[str] = field(default_factory=list)

    @property
    def is_healthy(self) -> bool:
        required = [check for check in self.checks if check.status != "skipped"]
        if not required:
            return False
        if self.mode in {"tushare", "akshare"}:
            return all(check.status == "success" for check in required)

        interfaces = {check.interface for check in required}
        return all(
            any(check.interface == interface and check.status == "success" for check in required)
            for interface in interfaces
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "mode": self.mode,
            "price_ticker": self.price_ticker,
            "statement_ticker": self.statement_ticker,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "tushare_configured": self.tushare_configured,
            "tushare_http_url": self.tushare_http_url,
            "is_healthy": self.is_healthy,
            "checks": [check.to_dict() for check in self.checks],
            "warnings": self.warnings,
        }


async def run_a_share_data_source_health_check(
    *,
    price_ticker: str = "000001.SZ",
    statement_ticker: str = "600519.SH",
    start_date: str = "2018-07-01",
    end_date: str = "2018-07-18",
    mode: DataSourceMode = "auto",
    output_path: str | Path = DEFAULT_HEALTH_REPORT_PATH,
) -> DataSourceHealthReport:
    """Check A-share OHLCV and three financial statement endpoints."""

    cfg = get_tushare_config()
    client = MarketDataClientFactory.get_client("cn")
    if not isinstance(client, CNMarketDataClient):
        raise TypeError("Expected CNMarketDataClient for A-share health check")

    warnings: list[str] = []
    checks: list[DataSourceCheckResult] = []

    if mode == "tushare" and not cfg.token:
        warnings.append("TuShare Token 未配置，TuShare 专项检查会被标记为失败。")

    provider_plan = _provider_plan(mode=mode, tushare_configured=bool(cfg.token))
    for provider in provider_plan:
        checks.extend(
            await _run_provider_checks(
                client=client,
                provider=provider,
                price_ticker=price_ticker,
                statement_ticker=statement_ticker,
                start_date=start_date,
                end_date=end_date,
                tushare_configured=bool(cfg.token),
            )
        )

    report = DataSourceHealthReport(
        generated_at=pd.Timestamp.utcnow().isoformat(),
        mode=mode,
        price_ticker=price_ticker,
        statement_ticker=statement_ticker,
        start_date=start_date,
        end_date=end_date,
        tushare_configured=bool(cfg.token),
        tushare_http_url=cfg.http_url,
        checks=checks,
        warnings=warnings + _build_provider_warnings(checks, mode=mode),
    )
    _save_report(report, output_path)
    return report


def load_latest_data_source_health_report(
    path: str | Path = DEFAULT_HEALTH_REPORT_PATH,
) -> DataSourceHealthReport | None:
    """Load the latest persisted health check report if it exists."""

    report_path = Path(path)
    if not report_path.exists():
        return None

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    return DataSourceHealthReport(
        generated_at=str(payload.get("generated_at", "")),
        mode=payload.get("mode", "auto"),
        price_ticker=str(payload.get("price_ticker", "")),
        statement_ticker=str(payload.get("statement_ticker", "")),
        start_date=str(payload.get("start_date", "")),
        end_date=str(payload.get("end_date", "")),
        tushare_configured=bool(payload.get("tushare_configured", False)),
        tushare_http_url=payload.get("tushare_http_url"),
        checks=[
            DataSourceCheckResult(
                interface=str(item.get("interface", "")),
                ticker=str(item.get("ticker", "")),
                provider=str(item.get("provider", "")),
                status=item.get("status", "failed"),
                row_count=int(item.get("row_count", 0) or 0),
                elapsed_seconds=item.get("elapsed_seconds"),
                error=item.get("error"),
                columns=list(item.get("columns") or []),
                preview=list(item.get("preview") or []),
            )
            for item in payload.get("checks", [])
        ],
        warnings=list(payload.get("warnings") or []),
    )


def _provider_plan(*, mode: DataSourceMode, tushare_configured: bool) -> list[Literal["tushare", "akshare"]]:
    if mode == "tushare":
        return ["tushare"]
    if mode == "akshare":
        return ["akshare"]
    return ["tushare"]


async def _run_provider_checks(
    *,
    client: CNMarketDataClient,
    provider: Literal["tushare", "akshare"],
    price_ticker: str,
    statement_ticker: str,
    start_date: str,
    end_date: str,
    tushare_configured: bool,
) -> list[DataSourceCheckResult]:
    if provider == "tushare" and not tushare_configured:
        return [
            DataSourceCheckResult(
                interface=interface,
                ticker=price_ticker if interface == "OHLCV 日线" else statement_ticker,
                provider="TuShare",
                status="failed",
                row_count=0,
                error="TUSHARE_TOKEN 未配置。",
            )
            for interface in ["OHLCV 日线", "资产负债表", "现金流量表", "利润表"]
        ]

    tasks = [
        _run_check(
            interface="OHLCV 日线",
            ticker=price_ticker,
            provider=_provider_label(provider),
            fetcher=lambda: _fetch_provider_ohlcv(client, provider, price_ticker, start_date, end_date),
        ),
        _run_check(
            interface="资产负债表",
            ticker=statement_ticker,
            provider=_provider_label(provider),
            fetcher=lambda: _fetch_provider_statement(client, provider, statement_ticker, "balance_sheet"),
        ),
        _run_check(
            interface="现金流量表",
            ticker=statement_ticker,
            provider=_provider_label(provider),
            fetcher=lambda: _fetch_provider_statement(client, provider, statement_ticker, "cash_flow"),
        ),
        _run_check(
            interface="利润表",
            ticker=statement_ticker,
            provider=_provider_label(provider),
            fetcher=lambda: _fetch_provider_statement(client, provider, statement_ticker, "income_statement"),
        ),
    ]
    return await asyncio.gather(*tasks)


async def _run_check(
    *,
    interface: str,
    ticker: str,
    provider: str,
    fetcher: Callable[[], Any],
) -> DataSourceCheckResult:
    started = time.perf_counter()
    try:
        frame = await fetcher()
        return DataSourceCheckResult(
            interface=interface,
            ticker=ticker,
            provider=provider,
            status="success",
            row_count=len(frame),
            elapsed_seconds=round(time.perf_counter() - started, 2),
            columns=[str(column) for column in frame.columns[:20]],
            preview=_preview_frame(frame),
        )
    except Exception as exc:
        return DataSourceCheckResult(
            interface=interface,
            ticker=ticker,
            provider=provider,
            status="failed",
            row_count=0,
            elapsed_seconds=round(time.perf_counter() - started, 2),
            error=_friendly_error(exc),
        )


async def _fetch_provider_ohlcv(
    client: CNMarketDataClient,
    provider: Literal["tushare", "akshare"],
    ticker: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    if provider == "tushare":
        return await client._with_retry(  # noqa: SLF001 - GUI diagnostic needs explicit provider checks.
            f"TuShare OHLCV health check for {ticker}",
            client._fetch_tushare_ohlcv,  # noqa: SLF001
            ticker,
            start_date,
            end_date,
        )
    return await client._with_retry(  # noqa: SLF001
        f"AkShare OHLCV health check for {ticker}",
        client._fetch_akshare_ohlcv,  # noqa: SLF001
        ticker,
        start_date,
        end_date,
        "1d",
        True,
    )


async def _fetch_provider_statement(
    client: CNMarketDataClient,
    provider: Literal["tushare", "akshare"],
    ticker: str,
    statement_kind: Literal["balance_sheet", "cash_flow", "income_statement"],
) -> pd.DataFrame:
    if provider == "tushare":
        return await client._with_retry(  # noqa: SLF001
            f"TuShare {statement_kind} health check for {ticker}",
            client._fetch_tushare_statement,  # noqa: SLF001
            ticker,
            statement_kind,
            "annual",
            2,
        )
    return await client._with_retry(  # noqa: SLF001
        f"AkShare {statement_kind} health check for {ticker}",
        client._fetch_akshare_statement,  # noqa: SLF001
        ticker,
        statement_kind,
        2,
    )


def _provider_label(provider: Literal["tushare", "akshare"]) -> str:
    return "TuShare" if provider == "tushare" else "AkShare"


def _preview_frame(frame: pd.DataFrame, limit: int = 5) -> list[dict[str, Any]]:
    normalized = frame.head(limit).where(pd.notna(frame.head(limit)), None)
    return normalized.to_dict(orient="records")


def _friendly_error(exc: Exception) -> str:
    if isinstance(exc, DataProviderError):
        prefix = "数据源错误"
    else:
        prefix = type(exc).__name__
    message = str(exc).strip() or type(exc).__name__
    if "ProxyError" in message or "Unable to connect to proxy" in message:
        message = f"{message}；这通常是本机代理无法连接东方财富接口导致，自动模式会继续使用 TuShare 兜底。"
    return f"{prefix}: {message}"


def _build_provider_warnings(checks: list[DataSourceCheckResult], *, mode: DataSourceMode) -> list[str]:
    warnings: list[str] = []
    if mode != "auto":
        return warnings

    failed = [check for check in checks if check.status == "failed"]
    if not failed:
        return warnings

    successful_interfaces = {check.interface for check in checks if check.status == "success"}
    for check in failed:
        if check.interface in successful_interfaces:
            warnings.append(
                f"{check.provider} 的 {check.interface} 暂不可用，但自动模式已有其他数据源成功返回，整体流程可继续。"
            )
        else:
            warnings.append(f"{check.provider} 的 {check.interface} 暂不可用，且没有可用兜底数据源。")
    return warnings


def _save_report(report: DataSourceHealthReport, path: str | Path) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
