"""Signal-fusion and timing rules for rebalancing execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

from fin_agent_sakura.monitoring.portfolio_monitor import RebalanceEvent


OrderAction = Literal["buy", "sell"]
TimingDecision = Literal["execute", "defer", "ignore"]


@dataclass(frozen=True, slots=True)
class TechnicalSignal:
    """Technical analyst output used by the timing rules."""

    ticker: str
    rsi_14: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    price_above_ma_50: bool | None = None
    price_above_ma_200: bool | None = None

    @property
    def is_overbought(self) -> bool:
        return self.rsi_14 is not None and self.rsi_14 > 70

    @property
    def is_oversold(self) -> bool:
        return self.rsi_14 is not None and self.rsi_14 < 30

    @property
    def has_bullish_momentum(self) -> bool:
        macd_bullish = (
            self.macd is not None
            and self.macd_signal is not None
            and self.macd > self.macd_signal
        )
        trend_bullish = self.price_above_ma_50 is True or self.price_above_ma_200 is True
        return macd_bullish or trend_bullish

    @property
    def has_bearish_momentum(self) -> bool:
        macd_bearish = (
            self.macd is not None
            and self.macd_signal is not None
            and self.macd < self.macd_signal
        )
        trend_bearish = self.price_above_ma_50 is False and self.price_above_ma_200 is False
        return macd_bearish or trend_bearish


@dataclass(frozen=True, slots=True)
class SentimentSignal:
    """Sentiment analyst output used by the timing rules."""

    ticker: str
    score: float
    confidence: float = 1.0

    @property
    def weighted_score(self) -> float:
        return self.score * self.confidence


@dataclass(frozen=True, slots=True)
class TradeOrder:
    """Formal execution order emitted by the timing rule engine."""

    client_id: str
    ticker: str
    action: OrderAction
    target_weight_delta: float
    reason: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True, slots=True)
class TimingRuleResult:
    """Decision for one rebalance event after signal fusion."""

    decision: TimingDecision
    event: RebalanceEvent
    order: TradeOrder | None
    reasons: list[str]


@dataclass(frozen=True, slots=True)
class TimingRuleConfig:
    """Thresholds controlling signal fusion and execution timing."""

    overbought_rsi: float = 70.0
    oversold_rsi: float = 30.0
    bullish_sentiment_threshold: float = 0.2
    bearish_sentiment_threshold: float = -0.2
    require_sentiment_for_buy: bool = True
    require_technical_confirmation_for_buy: bool = True


class TimingRuleEngine:
    """Explicit rule engine for rebalancing timing decisions."""

    def __init__(self, config: TimingRuleConfig | None = None) -> None:
        self.config = config or TimingRuleConfig()

    def evaluate(
        self,
        event: RebalanceEvent,
        technical_signal: TechnicalSignal,
        sentiment_signal: SentimentSignal | None = None,
    ) -> TimingRuleResult:
        """Fuse drift, technicals, and sentiment into an execution decision."""

        action = _action_from_drift(event.drift)
        reasons: list[str] = [
            f"{event.ticker} drift {event.drift:.4f} exceeds threshold {event.threshold:.4f}."
        ]

        if action == "buy":
            return self._evaluate_buy(event, technical_signal, sentiment_signal, reasons)
        return self._evaluate_sell(event, technical_signal, sentiment_signal, reasons)

    def _evaluate_buy(
        self,
        event: RebalanceEvent,
        technical_signal: TechnicalSignal,
        sentiment_signal: SentimentSignal | None,
        reasons: list[str],
    ) -> TimingRuleResult:
        if technical_signal.rsi_14 is not None and technical_signal.rsi_14 > self.config.overbought_rsi:
            reasons.append(f"RSI {technical_signal.rsi_14:.2f} indicates overbought conditions.")
            return TimingRuleResult("defer", event, None, reasons)

        bullish_sentiment = (
            sentiment_signal is not None
            and sentiment_signal.weighted_score >= self.config.bullish_sentiment_threshold
        )
        bullish_technicals = technical_signal.has_bullish_momentum or technical_signal.is_oversold

        if self.config.require_sentiment_for_buy and not bullish_sentiment:
            reasons.append("Buy deferred because sentiment confirmation is missing.")
            return TimingRuleResult("defer", event, None, reasons)

        if self.config.require_technical_confirmation_for_buy and not bullish_technicals:
            reasons.append("Buy deferred because technical confirmation is missing.")
            return TimingRuleResult("defer", event, None, reasons)

        reasons.append("Buy consensus confirmed by sentiment and technical signals.")
        order = TradeOrder(
            client_id=event.client_id,
            ticker=event.ticker,
            action="buy",
            target_weight_delta=abs(event.drift),
            reason=" ".join(reasons),
        )
        return TimingRuleResult("execute", event, order, reasons)

    def _evaluate_sell(
        self,
        event: RebalanceEvent,
        technical_signal: TechnicalSignal,
        sentiment_signal: SentimentSignal | None,
        reasons: list[str],
    ) -> TimingRuleResult:
        bearish_sentiment = (
            sentiment_signal is not None
            and sentiment_signal.weighted_score <= self.config.bearish_sentiment_threshold
        )
        sell_confirmed = (
            technical_signal.has_bearish_momentum
            or technical_signal.is_overbought
            or bearish_sentiment
        )

        if not sell_confirmed:
            reasons.append("Sell execution allowed to reduce drift even without bearish confirmation.")
        else:
            reasons.append("Sell confirmed by bearish, overbought, or risk-reduction signals.")

        order = TradeOrder(
            client_id=event.client_id,
            ticker=event.ticker,
            action="sell",
            target_weight_delta=abs(event.drift),
            reason=" ".join(reasons),
        )
        return TimingRuleResult("execute", event, order, reasons)


def generate_trade_orders(
    events: list[RebalanceEvent],
    technical_signals: dict[str, TechnicalSignal],
    sentiment_signals: dict[str, SentimentSignal] | None = None,
    *,
    engine: TimingRuleEngine | None = None,
) -> list[TradeOrder]:
    """Generate formal trade orders for rebalance events that pass timing rules."""

    rule_engine = engine or TimingRuleEngine()
    sentiments = sentiment_signals or {}
    orders: list[TradeOrder] = []

    for event in events:
        ticker = event.ticker.upper()
        technical = technical_signals.get(ticker)
        if technical is None:
            continue

        result = rule_engine.evaluate(event, technical, sentiments.get(ticker))
        if result.order is not None:
            orders.append(result.order)

    return orders


def _action_from_drift(drift: float) -> OrderAction:
    return "sell" if drift > 0 else "buy"

