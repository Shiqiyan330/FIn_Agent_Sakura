"""Async time-based event loop for multi-agent portfolio backtests."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import date
from typing import Any, Protocol

import pandas as pd


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Configuration for the event-driven backtest loop."""

    initial_cash: float = 1_000_000.0
    rebalance_frequency_days: int = 21
    transaction_cost_bps: float = 5.0
    risk_free_rate: float = 0.02


@dataclass(frozen=True, slots=True)
class BacktestEvent:
    """Historical slice passed to the multi-agent strategy."""

    current_date: pd.Timestamp
    price_history: pd.DataFrame
    latest_prices: pd.Series
    news: list[str]
    current_weights: pd.Series
    portfolio_value: float


@dataclass(frozen=True, slots=True)
class PortfolioDecision:
    """Strategy output for one rebalance date."""

    target_weights: dict[str, float]
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class BacktestSnapshot:
    """Daily portfolio state saved by the backtest loop."""

    date: pd.Timestamp
    portfolio_value: float
    weights: dict[str, float]
    daily_return: float
    turnover: float
    transaction_cost: float


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Backtest equity curve and performance metrics."""

    snapshots: list[BacktestSnapshot]
    equity_curve: pd.Series
    daily_returns: pd.Series
    cumulative_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float


class StrategyCallable(Protocol):
    """Async strategy interface compatible with LangGraph agent workflows."""

    def __call__(self, event: BacktestEvent) -> PortfolioDecision | Awaitable[PortfolioDecision]:
        """Return target portfolio weights for the current rebalance event."""


NewsProvider = Callable[[pd.Timestamp], list[str]]


async def run_event_driven_backtest(
    prices: pd.DataFrame,
    strategy: StrategyCallable,
    *,
    news_provider: NewsProvider | None = None,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Run a time-sliced async backtest over historical prices and news.

    Args:
        prices: Historical close price matrix indexed by date with ticker
            columns. A five-year daily matrix is the intended default input.
        strategy: Async or sync callback receiving BacktestEvent and returning
            target weights. This is where the LangGraph multi-agent system plugs
            in during each rebalance event.
        news_provider: Optional callable returning historical news texts for a
            given date.
        config: Backtest configuration including rebalance frequency and costs.

    Returns:
        BacktestResult with equity curve, daily returns, cumulative return,
        Sharpe ratio, and max drawdown.
    """

    cfg = config or BacktestConfig()
    clean_prices = _prepare_prices(prices)
    tickers = list(clean_prices.columns)
    weights = pd.Series(0.0, index=tickers, dtype="float64")
    portfolio_value = cfg.initial_cash
    previous_value = portfolio_value
    snapshots: list[BacktestSnapshot] = []

    for index, current_date in enumerate(clean_prices.index):
        latest_prices = clean_prices.loc[current_date]
        if index > 0:
            previous_prices = clean_prices.iloc[index - 1]
            asset_returns = latest_prices / previous_prices - 1
            portfolio_return = float(weights.dot(asset_returns.reindex(weights.index).fillna(0.0)))
            portfolio_value *= 1 + portfolio_return

        turnover = 0.0
        transaction_cost = 0.0
        if index == 0 or index % cfg.rebalance_frequency_days == 0:
            event = BacktestEvent(
                current_date=current_date,
                price_history=clean_prices.iloc[: index + 1],
                latest_prices=latest_prices,
                news=news_provider(current_date) if news_provider else [],
                current_weights=weights.copy(),
                portfolio_value=portfolio_value,
            )
            decision = await _maybe_await(strategy(event))
            target_weights = _normalize_target_weights(decision.target_weights, tickers)
            turnover = float((target_weights - weights).abs().sum())
            transaction_cost = portfolio_value * turnover * cfg.transaction_cost_bps / 10_000
            portfolio_value -= transaction_cost
            weights = target_weights

        daily_return = portfolio_value / previous_value - 1 if snapshots else 0.0
        snapshots.append(
            BacktestSnapshot(
                date=current_date,
                portfolio_value=portfolio_value,
                weights=weights.to_dict(),
                daily_return=float(daily_return),
                turnover=turnover,
                transaction_cost=transaction_cost,
            )
        )
        previous_value = portfolio_value

    return _build_result(snapshots, risk_free_rate=cfg.risk_free_rate)


async def async_constant_weight_strategy(event: BacktestEvent) -> PortfolioDecision:
    """Simple equal-weight baseline strategy."""

    tickers = list(event.latest_prices.index)
    weight = 1 / len(tickers)
    return PortfolioDecision(target_weights={ticker: weight for ticker in tickers})


async def _maybe_await(value: PortfolioDecision | Awaitable[PortfolioDecision]) -> PortfolioDecision:
    if inspect.isawaitable(value):
        return await value
    return value


def _prepare_prices(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        raise ValueError("prices must not be empty")
    clean = prices.copy(deep=False)
    clean.index = pd.to_datetime(clean.index)
    clean = clean.sort_index().astype("float64").ffill().dropna(how="any")
    if clean.empty:
        raise ValueError("prices has no complete rows after cleaning")
    if clean.shape[1] < 2:
        raise ValueError("prices must contain at least two assets")
    return clean


def _normalize_target_weights(target_weights: dict[str, float], tickers: list[str]) -> pd.Series:
    weights = pd.Series({ticker.upper().strip(): float(weight) for ticker, weight in target_weights.items()})
    aligned = weights.reindex(tickers).fillna(0.0).clip(lower=0.0)
    total = aligned.sum()
    if total <= 0:
        raise ValueError("target weights must sum to a positive value")
    return aligned / total


def _build_result(snapshots: list[BacktestSnapshot], *, risk_free_rate: float) -> BacktestResult:
    equity_curve = pd.Series(
        [snapshot.portfolio_value for snapshot in snapshots],
        index=[snapshot.date for snapshot in snapshots],
        dtype="float64",
    )
    daily_returns = equity_curve.pct_change().fillna(0.0)
    cumulative_return = float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)
    trading_days = max(len(daily_returns), 1)
    annualized_return = float((1 + cumulative_return) ** (252 / trading_days) - 1)
    annualized_volatility = float(daily_returns.std(ddof=0) * (252**0.5))
    excess_daily_return = daily_returns.mean() - risk_free_rate / 252
    sharpe_ratio = (
        float(excess_daily_return / daily_returns.std(ddof=0) * (252**0.5))
        if daily_returns.std(ddof=0) > 0
        else 0.0
    )
    max_drawdown = _max_drawdown(equity_curve)

    return BacktestResult(
        snapshots=snapshots,
        equity_curve=equity_curve,
        daily_returns=daily_returns,
        cumulative_return=cumulative_return,
        annualized_return=annualized_return,
        annualized_volatility=annualized_volatility,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
    )


def _max_drawdown(equity_curve: pd.Series) -> float:
    running_peak = equity_curve.cummax()
    drawdown = equity_curve / running_peak - 1
    return float(abs(drawdown.min()))

