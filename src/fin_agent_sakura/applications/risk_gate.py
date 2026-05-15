"""Application-level risk gate for paper trade orders."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from fin_agent_sakura.data import MarketDataClientFactory
from fin_agent_sakura.monitoring import RiskLimits, RiskManager, TradeOrder


MarketName = Literal["us", "cn"]


@dataclass(frozen=True, slots=True)
class RiskGateReport:
    """Persistable risk-gate result for proposed paper orders."""

    decision: str
    portfolio_var: float
    max_drawdown: float
    proposed_weights: dict[str, float]
    alerts: list[dict[str, Any]]
    orders: list[dict[str, Any]]
    warnings: list[str]

    @property
    def approved(self) -> bool:
        return self.decision == "approved"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_paper_orders_risk(
    *,
    current_weights: dict[str, float],
    orders: list[dict[str, Any]],
    market: MarketName = "cn",
    max_var: float = 0.035,
    max_drawdown: float = 0.25,
    var_confidence: float = 0.95,
    lookback_days: int = 365,
    client_id: str = "paper_client",
    output_dir: str | Path = "data/processed",
) -> RiskGateReport:
    """Evaluate proposed paper orders through the hard-coded RiskManager."""

    warnings: list[str] = []
    trade_orders = [_dict_to_trade_order(order, client_id=client_id) for order in orders]
    limits = RiskLimits(max_var=max_var, max_drawdown=max_drawdown, var_confidence=var_confidence)
    risk_manager = RiskManager(limits)

    try:
        historical_prices = asyncio.run(
            _fetch_historical_price_matrix(
                sorted(_needed_tickers(current_weights, orders)),
                market=market,
                lookback_days=lookback_days,
            )
        )
        assessment = risk_manager.evaluate_orders(
            client_id=client_id,
            current_weights=current_weights,
            orders=trade_orders,
            historical_prices=historical_prices,
        )
        report = RiskGateReport(
            decision=assessment.decision,
            portfolio_var=assessment.portfolio_var,
            max_drawdown=assessment.max_drawdown,
            proposed_weights={ticker: float(weight) for ticker, weight in assessment.proposed_weights.items()},
            alerts=[_alert_to_dict(alert) for alert in assessment.alerts],
            orders=orders,
            warnings=warnings,
        )
    except Exception as exc:
        warnings.append(f"风险断路器评估失败，已拒绝导出可执行订单：{type(exc).__name__}: {exc}")
        report = RiskGateReport(
            decision="rejected",
            portfolio_var=0.0,
            max_drawdown=0.0,
            proposed_weights=current_weights,
            alerts=[
                {
                    "client_id": client_id,
                    "metric": "risk_gate_error",
                    "observed_value": 1.0,
                    "threshold": 0.0,
                    "message": warnings[-1],
                    "created_at": pd.Timestamp.now().isoformat(),
                }
            ],
            orders=orders,
            warnings=warnings,
        )

    _save_report(report, output_dir=output_dir)
    return report


def load_latest_risk_gate_report(output_dir: str | Path = "data/processed") -> RiskGateReport | None:
    """Load the latest saved risk-gate report."""

    path = Path(output_dir) / "risk_gate_latest.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return RiskGateReport(**payload)


async def _fetch_historical_price_matrix(
    tickers: list[str],
    *,
    market: MarketName,
    lookback_days: int,
) -> pd.DataFrame:
    if len(tickers) < 2:
        raise ValueError("风险评估至少需要两个资产的历史价格")
    client = MarketDataClientFactory.get_client(market)
    start = date.today() - timedelta(days=lookback_days)
    frames = await asyncio.gather(
        *[client.fetch_ohlcv(ticker, start=start, interval="1d", adjusted=True) for ticker in tickers]
    )
    series = []
    for ticker, frame in zip(tickers, frames, strict=True):
        if "date" not in frame.columns or "close" not in frame.columns:
            raise ValueError(f"{ticker} 历史行情缺少 date/close 字段")
        close = frame[["date", "close"]].copy()
        close["date"] = pd.to_datetime(close["date"])
        close["close"] = pd.to_numeric(close["close"], errors="coerce")
        item = close.dropna().drop_duplicates("date", keep="last").sort_values("date").set_index("date")["close"]
        item.name = ticker
        series.append(item)
    prices = pd.concat(series, axis=1).ffill().dropna(how="any")
    if prices.empty:
        raise ValueError("历史价格矩阵为空")
    return prices


def _dict_to_trade_order(order: dict[str, Any], *, client_id: str) -> TradeOrder:
    return TradeOrder(
        client_id=str(order.get("client_id") or client_id),
        ticker=str(order["ticker"]).upper().strip(),
        action=str(order["action"]).lower(),
        target_weight_delta=float(order["target_weight_delta"]),
        reason=str(order.get("reason") or ""),
    )


def _needed_tickers(current_weights: dict[str, float], orders: list[dict[str, Any]]) -> set[str]:
    tickers = {ticker.upper().strip() for ticker in current_weights}
    tickers.update(str(order["ticker"]).upper().strip() for order in orders)
    return {ticker for ticker in tickers if ticker}


def _alert_to_dict(alert: Any) -> dict[str, Any]:
    return {
        "client_id": alert.client_id,
        "metric": alert.metric,
        "observed_value": float(alert.observed_value),
        "threshold": float(alert.threshold),
        "message": alert.message,
        "created_at": alert.created_at.isoformat(),
    }


def _save_report(report: RiskGateReport, *, output_dir: str | Path) -> Path:
    path = Path(output_dir) / "risk_gate_latest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path
