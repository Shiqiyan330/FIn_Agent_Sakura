"""GUI-ready A-share event-driven backtest service."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from fin_agent_sakura.backtesting import BacktestConfig, BacktestEvent, PortfolioDecision, run_event_driven_backtest
from fin_agent_sakura.data import MarketDataClientFactory


MarketName = Literal["us", "cn"]
StrategyName = Literal["equal_weight", "momentum_top_n"]


@dataclass(frozen=True, slots=True)
class BacktestRunReport:
    """Persistable GUI report for one backtest run."""

    tickers: list[str]
    market: MarketName
    strategy: StrategyName
    start_date: str
    end_date: str
    cumulative_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    benchmark_cumulative_return: float
    benchmark_sharpe_ratio: float
    benchmark_max_drawdown: float
    index_benchmark_cumulative_return: float | None
    index_benchmark_sharpe_ratio: float | None
    index_benchmark_max_drawdown: float | None
    equity_curve: list[dict[str, Any]]
    benchmark_curve: list[dict[str, Any]]
    index_benchmark_curve: list[dict[str, Any]]
    snapshots: list[dict[str, Any]]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_a_share_backtest(
    tickers: list[str],
    *,
    market: MarketName = "cn",
    strategy: StrategyName = "momentum_top_n",
    start_date: str = "2020-01-01",
    end_date: str | None = None,
    rebalance_frequency_days: int = 21,
    transaction_cost_bps: float = 5.0,
    slippage_bps: float = 0.0,
    initial_cash: float = 1_000_000.0,
    risk_free_rate: float = 0.02,
    momentum_lookback_days: int = 60,
    top_n: int = 5,
    news_csv_path: str | Path | None = None,
    include_index_benchmark: bool = False,
    index_benchmark_ticker: str = "000300.SH",
    output_dir: str | Path = "data/processed",
) -> BacktestRunReport:
    """Run an event-driven A-share backtest and persist a GUI report."""

    clean_tickers = _normalize_tickers(tickers)
    prices = asyncio.run(_fetch_price_matrix(clean_tickers, market=market, start=start_date, end=end_date))
    strategy_fn = _build_strategy(strategy, lookback_days=momentum_lookback_days, top_n=top_n)
    config = BacktestConfig(
        initial_cash=initial_cash,
        rebalance_frequency_days=rebalance_frequency_days,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
        risk_free_rate=risk_free_rate,
    )
    news_provider = _build_news_provider(news_csv_path)
    result = asyncio.run(run_event_driven_backtest(prices, strategy_fn, news_provider=news_provider, config=config))

    benchmark_config = BacktestConfig(
        initial_cash=initial_cash,
        rebalance_frequency_days=max(len(prices) + 1, 999_999),
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
        risk_free_rate=risk_free_rate,
    )
    benchmark = asyncio.run(
        run_event_driven_backtest(prices, _equal_weight_strategy, config=benchmark_config)
    )
    warnings: list[str] = []
    if news_csv_path:
        warnings.append(f"已加载历史新闻CSV：{news_csv_path}")
    index_curve: list[dict[str, Any]] = []
    index_cumulative_return = None
    index_sharpe = None
    index_max_drawdown = None
    if include_index_benchmark:
        try:
            index_result = asyncio.run(
                _run_index_benchmark(
                    index_benchmark_ticker,
                    market=market,
                    start=start_date,
                    end=end_date,
                    initial_cash=initial_cash,
                    risk_free_rate=risk_free_rate,
                )
            )
            index_curve = _series_to_curve(index_result.equity_curve, "index_benchmark_value")
            index_cumulative_return = index_result.cumulative_return
            index_sharpe = index_result.sharpe_ratio
            index_max_drawdown = index_result.max_drawdown
        except Exception as exc:
            warnings.append(f"沪深300基准构建失败，已跳过指数基准：{type(exc).__name__}: {exc}")

    report = BacktestRunReport(
        tickers=clean_tickers,
        market=market,
        strategy=strategy,
        start_date=str(prices.index.min().date()),
        end_date=str(prices.index.max().date()),
        cumulative_return=result.cumulative_return,
        annualized_return=result.annualized_return,
        annualized_volatility=result.annualized_volatility,
        sharpe_ratio=result.sharpe_ratio,
        max_drawdown=result.max_drawdown,
        benchmark_cumulative_return=benchmark.cumulative_return,
        benchmark_sharpe_ratio=benchmark.sharpe_ratio,
        benchmark_max_drawdown=benchmark.max_drawdown,
        index_benchmark_cumulative_return=index_cumulative_return,
        index_benchmark_sharpe_ratio=index_sharpe,
        index_benchmark_max_drawdown=index_max_drawdown,
        equity_curve=_series_to_curve(result.equity_curve, "strategy_value"),
        benchmark_curve=_series_to_curve(benchmark.equity_curve, "benchmark_value"),
        index_benchmark_curve=index_curve,
        snapshots=[_snapshot_to_dict(snapshot) for snapshot in result.snapshots],
        warnings=warnings,
    )
    _save_report(report, output_dir=output_dir)
    _save_html_report(report, output_dir=output_dir)
    return report


def load_latest_backtest_report(output_dir: str | Path = "data/processed") -> BacktestRunReport | None:
    """Load the latest saved backtest report."""

    path = Path(output_dir) / "backtest_latest.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("index_benchmark_cumulative_return", None)
    payload.setdefault("index_benchmark_sharpe_ratio", None)
    payload.setdefault("index_benchmark_max_drawdown", None)
    payload.setdefault("index_benchmark_curve", [])
    return BacktestRunReport(**payload)


def save_backtest_news_csv(uploaded_file: Any, *, output_dir: str | Path = "data/processed") -> Path:
    """Persist a user-uploaded historical news CSV for backtest news_provider use."""

    filename = Path(str(uploaded_file.name)).name
    if not filename.lower().endswith(".csv"):
        raise ValueError("历史新闻文件必须是 CSV。")
    path = Path(output_dir) / "backtest_news_latest.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(uploaded_file.getbuffer())
    return path


async def _fetch_price_matrix(
    tickers: list[str],
    *,
    market: MarketName,
    start: str,
    end: str | None,
) -> pd.DataFrame:
    client = MarketDataClientFactory.get_client(market)
    frames = await asyncio.gather(
        *[client.fetch_ohlcv(ticker, start=start, end=end, interval="1d", adjusted=True) for ticker in tickers]
    )
    series = []
    for ticker, frame in zip(tickers, frames, strict=True):
        if "date" not in frame.columns or "close" not in frame.columns:
            raise ValueError(f"{ticker} historical frame must include date and close columns")
        close = frame[["date", "close"]].copy()
        close["date"] = pd.to_datetime(close["date"])
        close["close"] = pd.to_numeric(close["close"], errors="coerce")
        item = close.dropna().drop_duplicates("date", keep="last").sort_values("date").set_index("date")["close"]
        item.name = ticker
        series.append(item)
    prices = pd.concat(series, axis=1).sort_index().ffill().dropna(how="any")
    if prices.empty:
        raise ValueError("No overlapping price history available for the selected tickers")
    return prices


def _build_strategy(strategy: StrategyName, *, lookback_days: int, top_n: int) -> Any:
    if strategy == "equal_weight":
        return _equal_weight_strategy
    if strategy == "momentum_top_n":
        return lambda event: _momentum_top_n_strategy(event, lookback_days=lookback_days, top_n=top_n)
    raise ValueError(f"Unsupported strategy: {strategy}")


async def _run_index_benchmark(
    ticker: str,
    *,
    market: MarketName,
    start: str,
    end: str | None,
    initial_cash: float,
    risk_free_rate: float,
) -> Any:
    client = MarketDataClientFactory.get_client(market)
    frame = await client.fetch_ohlcv(ticker, start=start, end=end, interval="1d", adjusted=True)
    if "date" not in frame.columns or "close" not in frame.columns:
        raise ValueError(f"{ticker} index frame must include date and close columns")
    close = frame[["date", "close"]].copy()
    close["date"] = pd.to_datetime(close["date"])
    close["close"] = pd.to_numeric(close["close"], errors="coerce")
    series = close.dropna().drop_duplicates("date", keep="last").sort_values("date").set_index("date")["close"]
    equity = series / series.iloc[0] * initial_cash
    returns = equity.pct_change().fillna(0.0)
    cumulative_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
    annualized_return = float((1 + cumulative_return) ** (252 / max(len(returns), 1)) - 1)
    annualized_volatility = float(returns.std(ddof=0) * (252**0.5))
    excess_daily_return = returns.mean() - risk_free_rate / 252
    sharpe_ratio = float(excess_daily_return / returns.std(ddof=0) * (252**0.5)) if returns.std(ddof=0) > 0 else 0.0
    drawdown = equity / equity.cummax() - 1

    class _IndexResult:
        pass

    result = _IndexResult()
    result.equity_curve = equity
    result.cumulative_return = cumulative_return
    result.annualized_return = annualized_return
    result.annualized_volatility = annualized_volatility
    result.sharpe_ratio = sharpe_ratio
    result.max_drawdown = float(abs(drawdown.min()))
    return result


def _build_news_provider(news_csv_path: str | Path | None) -> Any | None:
    if not news_csv_path:
        return None
    path = Path(news_csv_path)
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if "date" not in frame.columns or "text" not in frame.columns:
        raise ValueError("历史新闻CSV必须包含 date 和 text 两列")
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date
    frame["text"] = frame["text"].fillna("").astype(str)
    grouped = frame.dropna(subset=["date"]).groupby("date")["text"].apply(list).to_dict()
    return lambda current_date: grouped.get(pd.Timestamp(current_date).date(), [])


def _equal_weight_strategy(event: BacktestEvent) -> PortfolioDecision:
    tickers = list(event.latest_prices.index)
    weight = 1 / len(tickers)
    return PortfolioDecision(target_weights={ticker: weight for ticker in tickers})


def _momentum_top_n_strategy(event: BacktestEvent, *, lookback_days: int, top_n: int) -> PortfolioDecision:
    prices = event.price_history
    if len(prices) <= max(2, lookback_days):
        return _equal_weight_strategy(event)
    lookback_prices = prices.iloc[-lookback_days]
    momentum = event.latest_prices / lookback_prices - 1
    winners = momentum.sort_values(ascending=False).head(max(1, min(top_n, len(momentum)))).index.tolist()
    weight = 1 / len(winners)
    return PortfolioDecision(
        target_weights={ticker: (weight if ticker in winners else 0.0) for ticker in prices.columns},
        metadata={"selected": winners, "lookback_days": lookback_days},
    )


def _series_to_curve(series: pd.Series, value_name: str) -> list[dict[str, Any]]:
    return [
        {"date": index.isoformat(), value_name: float(value)}
        for index, value in series.items()
    ]


def _snapshot_to_dict(snapshot: Any) -> dict[str, Any]:
    return {
        "date": snapshot.date.isoformat(),
        "portfolio_value": float(snapshot.portfolio_value),
        "weights": {ticker: float(weight) for ticker, weight in snapshot.weights.items()},
        "daily_return": float(snapshot.daily_return),
        "turnover": float(snapshot.turnover),
        "transaction_cost": float(snapshot.transaction_cost),
    }


def _normalize_tickers(tickers: list[str]) -> list[str]:
    cleaned = [ticker.upper().strip() for ticker in tickers if ticker.strip()]
    unique = list(dict.fromkeys(cleaned))
    if len(unique) < 2:
        raise ValueError("Backtest requires at least two tickers")
    return unique


def _save_report(report: BacktestRunReport, *, output_dir: str | Path) -> Path:
    path = Path(output_dir) / "backtest_latest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


def _save_html_report(report: BacktestRunReport, *, output_dir: str | Path) -> Path:
    path = Path(output_dir) / "backtest_latest.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = "".join(
        f"<tr><td>{ticker}</td></tr>"
        for ticker in report.tickers
    )
    html = f"""
    <html>
      <head><meta charset="utf-8"><title>A股回测报告</title></head>
      <body style="font-family: Arial, sans-serif; line-height: 1.55;">
        <h1>A股事件驱动回测报告</h1>
        <p><strong>区间：</strong>{report.start_date} 至 {report.end_date}</p>
        <p><strong>策略：</strong>{report.strategy}</p>
        <h2>核心指标</h2>
        <ul>
          <li>累计收益：{report.cumulative_return:.2%}</li>
          <li>年化收益：{report.annualized_return:.2%}</li>
          <li>年化波动：{report.annualized_volatility:.2%}</li>
          <li>Sharpe：{report.sharpe_ratio:.2f}</li>
          <li>最大回撤：{report.max_drawdown:.2%}</li>
          <li>等权买入持有基准收益：{report.benchmark_cumulative_return:.2%}</li>
        </ul>
        <h2>股票池</h2>
        <table border="1" cellspacing="0" cellpadding="6"><tbody>{rows}</tbody></table>
        <p>本报告由 Sakura 本地回测模块生成，仅用于研究验证。</p>
      </body>
    </html>
    """
    path.write_text(html, encoding="utf-8")
    return path
