"""Hard-coded risk circuit breaker for proposed trade orders."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

import pandas as pd

from fin_agent_sakura.monitoring.timing_rules import TradeOrder


RiskDecision = Literal["approved", "rejected"]


@dataclass(frozen=True, slots=True)
class RiskLimits:
    """Client-level hard safety thresholds."""

    max_var: float
    max_drawdown: float
    var_confidence: float = 0.95


@dataclass(frozen=True, slots=True)
class RiskAlert:
    """Alarm emitted when a proposed portfolio breaches hard risk limits."""

    client_id: str
    metric: str
    observed_value: float
    threshold: float
    message: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True, slots=True)
class RiskAssessment:
    """Risk gate result for a proposed trade set."""

    decision: RiskDecision
    portfolio_var: float
    max_drawdown: float
    proposed_weights: pd.Series
    alerts: list[RiskAlert]

    @property
    def approved(self) -> bool:
        return self.decision == "approved"


class RiskManager:
    """Independent risk circuit breaker for the final trading outlet."""

    def __init__(self, limits: RiskLimits, *, logger: logging.Logger | None = None) -> None:
        if not 0 < limits.var_confidence < 1:
            raise ValueError("var_confidence must satisfy 0 < confidence < 1")
        if limits.max_var <= 0:
            raise ValueError("max_var must be positive")
        if limits.max_drawdown <= 0:
            raise ValueError("max_drawdown must be positive")

        self.limits = limits
        self.logger = logger or logging.getLogger(__name__)
        self.alerts: list[RiskAlert] = []

    def evaluate_orders(
        self,
        *,
        client_id: str,
        current_weights: dict[str, float],
        orders: list[TradeOrder],
        historical_prices: pd.DataFrame,
    ) -> RiskAssessment:
        """Approve or reject proposed orders using VaR and max drawdown only."""

        proposed_weights = self._apply_orders(current_weights, orders)
        returns = _portfolio_returns(historical_prices, proposed_weights)
        portfolio_var = _historical_var(returns, confidence=self.limits.var_confidence)
        max_drawdown = _max_drawdown(returns)

        alerts: list[RiskAlert] = []
        if portfolio_var > self.limits.max_var:
            alerts.append(
                RiskAlert(
                    client_id=client_id,
                    metric="VaR",
                    observed_value=portfolio_var,
                    threshold=self.limits.max_var,
                    message=(
                        f"Proposed portfolio VaR {portfolio_var:.4f} exceeds "
                        f"limit {self.limits.max_var:.4f}."
                    ),
                )
            )

        if max_drawdown > self.limits.max_drawdown:
            alerts.append(
                RiskAlert(
                    client_id=client_id,
                    metric="max_drawdown",
                    observed_value=max_drawdown,
                    threshold=self.limits.max_drawdown,
                    message=(
                        f"Proposed portfolio max drawdown {max_drawdown:.4f} exceeds "
                        f"limit {self.limits.max_drawdown:.4f}."
                    ),
                )
            )

        for alert in alerts:
            self.logger.warning(alert.message)
            self.alerts.append(alert)

        decision: RiskDecision = "rejected" if alerts else "approved"
        return RiskAssessment(
            decision=decision,
            portfolio_var=portfolio_var,
            max_drawdown=max_drawdown,
            proposed_weights=proposed_weights,
            alerts=alerts,
        )

    def _apply_orders(self, current_weights: dict[str, float], orders: list[TradeOrder]) -> pd.Series:
        weights = pd.Series(
            {ticker.upper().strip(): float(weight) for ticker, weight in current_weights.items()},
            dtype="float64",
        )
        for order in orders:
            ticker = order.ticker.upper().strip()
            if ticker not in weights.index:
                weights.loc[ticker] = 0.0
            if order.action == "buy":
                weights.loc[ticker] += order.target_weight_delta
            else:
                weights.loc[ticker] -= order.target_weight_delta

        weights = weights.clip(lower=0)
        total = weights.sum()
        if total <= 0:
            raise ValueError("Proposed weights sum to zero after applying orders")
        return weights / total


def _portfolio_returns(historical_prices: pd.DataFrame, weights: pd.Series) -> pd.Series:
    if historical_prices.empty:
        raise ValueError("historical_prices must not be empty")

    prices = historical_prices.copy(deep=False)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index().astype("float64")
    common_assets = [ticker for ticker in weights.index if ticker in prices.columns]
    if len(common_assets) < 2:
        raise ValueError("At least two proposed assets must exist in historical_prices")

    aligned_weights = weights.reindex(common_assets).fillna(0.0)
    aligned_weights = aligned_weights / aligned_weights.sum()
    returns = prices[common_assets].pct_change(fill_method=None).dropna(how="any")
    if returns.empty:
        raise ValueError("Not enough price history to calculate returns")
    return returns.dot(aligned_weights)


def _historical_var(returns: pd.Series, *, confidence: float) -> float:
    percentile = 1 - confidence
    return float(max(0.0, -returns.quantile(percentile)))


def _max_drawdown(returns: pd.Series) -> float:
    cumulative = (1 + returns).cumprod()
    running_peak = cumulative.cummax()
    drawdown = cumulative / running_peak - 1
    return float(abs(drawdown.min()))

