"""Single-ticker technical analysis service for the Streamlit GUI."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from fin_agent_sakura.data import MarketDataClientFactory, TechnicalIndicators


DEFAULT_TECHNICAL_CACHE_DIR = Path("data/processed/technical_indicators")
MarketName = Literal["cn", "us"]


@dataclass(frozen=True, slots=True)
class TechnicalAnalysisReport:
    """Persistable single-ticker technical indicator report."""

    ticker: str
    market: MarketName
    generated_at: str
    start_date: str
    end_date: str | None
    latest: dict[str, Any]
    signals: dict[str, Any]
    history: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "market": self.market,
            "generated_at": self.generated_at,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "latest": self.latest,
            "signals": self.signals,
            "history": self.history,
        }


def run_single_ticker_technical_analysis(
    ticker: str,
    *,
    market: MarketName = "cn",
    lookback_days: int = 365,
    end_date: str | None = None,
    force_refresh: bool = False,
    cache_dir: str | Path = DEFAULT_TECHNICAL_CACHE_DIR,
) -> TechnicalAnalysisReport:
    """Fetch OHLCV, calculate indicators, and persist a compact report."""

    cache_path = _cache_path(cache_dir, ticker, market, lookback_days, end_date)
    if cache_path.exists() and not force_refresh:
        return _load_report(cache_path)

    report = asyncio.run(_build_report(ticker, market, lookback_days, end_date))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    latest_path = Path(cache_dir) / "technical_analysis_latest.json"
    latest_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return report


def load_latest_technical_analysis_report(
    cache_dir: str | Path = DEFAULT_TECHNICAL_CACHE_DIR,
) -> TechnicalAnalysisReport | None:
    """Load the latest generated technical report."""

    path = Path(cache_dir) / "technical_analysis_latest.json"
    if not path.exists():
        return None
    return _load_report(path)


async def _build_report(
    ticker: str,
    market: MarketName,
    lookback_days: int,
    end_date: str | None,
) -> TechnicalAnalysisReport:
    end = pd.Timestamp(end_date).date() if end_date else date.today()
    start = end - timedelta(days=lookback_days)
    client = MarketDataClientFactory.get_client(market)
    prices = await client.fetch_ohlcv(ticker, start=start.isoformat(), end=end.isoformat(), interval="1d", adjusted=True)
    indicators = TechnicalIndicators(engine="pandas").calculate(prices)
    indicators = indicators.where(pd.notna(indicators), None)
    latest = indicators.iloc[-1].to_dict()
    signals = _derive_signals(latest)
    history_columns = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_lower",
        "bb_middle",
        "bb_upper",
        "ma_50",
        "ma_200",
        "volume_ma_5",
        "volume_ma_20",
        "volume_ratio",
        "volume_signal",
    ]
    existing = [column for column in history_columns if column in indicators.columns]
    return TechnicalAnalysisReport(
        ticker=ticker.upper(),
        market=market,
        generated_at=pd.Timestamp.utcnow().isoformat(),
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        latest=_json_safe(latest),
        signals=signals,
        history=_json_safe(indicators[existing].tail(260).to_dict("records")),
    )


def _derive_signals(latest: dict[str, Any]) -> dict[str, Any]:
    close = _number(latest.get("close"))
    ma_50 = _number(latest.get("ma_50"))
    ma_200 = _number(latest.get("ma_200"))
    rsi = _number(latest.get("rsi_14"), 50.0)
    macd = _number(latest.get("macd"), 0.0)
    macd_signal = _number(latest.get("macd_signal"), 0.0)
    volume_signal = str(latest.get("volume_signal") or "unavailable")

    trend = "unknown"
    if close is not None and ma_50 is not None and ma_200 is not None:
        trend = "uptrend" if close >= ma_50 >= ma_200 else "downtrend" if close < ma_50 < ma_200 else "mixed"

    overbought = rsi is not None and rsi > 70
    oversold = rsi is not None and rsi < 30
    momentum = "bullish" if macd is not None and macd_signal is not None and macd > macd_signal else "bearish"
    execution = "buy_watch" if trend == "uptrend" and momentum == "bullish" and not overbought else "defer_buy" if overbought else "risk_reduce" if trend == "downtrend" else "hold"

    return {
        "trend": trend,
        "momentum": momentum,
        "overbought": overbought,
        "oversold": oversold,
        "volume_signal": volume_signal,
        "execution_signal": execution,
        "summary": f"趋势 {trend}，动量 {momentum}，RSI {rsi or 0:.1f}，成交量 {volume_signal}，建议 {execution}。",
    }


def _cache_path(
    cache_dir: str | Path,
    ticker: str,
    market: MarketName,
    lookback_days: int,
    end_date: str | None,
) -> Path:
    safe_ticker = ticker.upper().replace(".", "_")
    suffix = end_date or "latest"
    return Path(cache_dir) / f"{market}_{safe_ticker}_{lookback_days}_{suffix}.json"


def _load_report(path: Path) -> TechnicalAnalysisReport:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return TechnicalAnalysisReport(
        ticker=payload["ticker"],
        market=payload["market"],
        generated_at=payload["generated_at"],
        start_date=payload["start_date"],
        end_date=payload.get("end_date"),
        latest=payload.get("latest", {}),
        signals=payload.get("signals", {}),
        history=payload.get("history", []),
    )


def _number(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))
