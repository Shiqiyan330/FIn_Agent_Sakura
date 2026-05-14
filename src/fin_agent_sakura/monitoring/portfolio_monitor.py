"""Threshold-drift portfolio monitor for rebalancing events."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol


@dataclass(frozen=True, slots=True)
class RebalanceEvent:
    """Event emitted when a holding drifts beyond the configured threshold."""

    client_id: str
    ticker: str
    current_weight: float
    target_weight: float
    drift: float
    threshold: float
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def direction(self) -> str:
        """Return whether the asset is overweight or underweight."""

        return "overweight" if self.drift > 0 else "underweight"


class PortfolioRepository(Protocol):
    """Database adapter contract used by PortfolioMonitor."""

    async def get_current_weights(self, client_id: str) -> dict[str, float]:
        """Return current actual portfolio weights keyed by ticker."""

    async def get_target_weights(self, client_id: str) -> dict[str, float]:
        """Return BL target portfolio weights keyed by ticker."""

    async def record_rebalance_event(self, event: RebalanceEvent) -> None:
        """Persist one rebalance event."""


class InMemoryPortfolioRepository:
    """Small repository useful for local tests before a real DB is wired in."""

    def __init__(
        self,
        current_weights: dict[str, dict[str, float]] | None = None,
        target_weights: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self.current_weights = current_weights or {}
        self.target_weights = target_weights or {}
        self.events: list[RebalanceEvent] = []

    async def get_current_weights(self, client_id: str) -> dict[str, float]:
        return dict(self.current_weights.get(client_id, {}))

    async def get_target_weights(self, client_id: str) -> dict[str, float]:
        return dict(self.target_weights.get(client_id, {}))

    async def record_rebalance_event(self, event: RebalanceEvent) -> None:
        self.events.append(event)


class PortfolioMonitor:
    """Background task that detects weight drift against BL target weights."""

    def __init__(
        self,
        repository: PortfolioRepository,
        *,
        drift_threshold: float = 0.05,
        poll_interval_seconds: float = 300.0,
    ) -> None:
        if not 0 < drift_threshold <= 1:
            raise ValueError("drift_threshold must satisfy 0 < threshold <= 1")
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")

        self.repository = repository
        self.drift_threshold = drift_threshold
        self.poll_interval_seconds = poll_interval_seconds
        self._stop_event = asyncio.Event()

    async def check_client(self, client_id: str) -> list[RebalanceEvent]:
        """Check one client once and persist rebalance events for drift breaches."""

        current_weights = _normalize_weights(await self.repository.get_current_weights(client_id))
        target_weights = _normalize_weights(await self.repository.get_target_weights(client_id))
        tickers = sorted(set(current_weights) | set(target_weights))

        events: list[RebalanceEvent] = []
        for ticker in tickers:
            current = current_weights.get(ticker, 0.0)
            target = target_weights.get(ticker, 0.0)
            drift = current - target
            if abs(drift) <= self.drift_threshold:
                continue

            event = RebalanceEvent(
                client_id=client_id,
                ticker=ticker,
                current_weight=current,
                target_weight=target,
                drift=drift,
                threshold=self.drift_threshold,
            )
            await self.repository.record_rebalance_event(event)
            events.append(event)

        return events

    async def run_forever(self, client_ids: list[str]) -> None:
        """Poll clients until stop is requested."""

        self._stop_event.clear()
        while not self._stop_event.is_set():
            await asyncio.gather(*(self.check_client(client_id) for client_id in client_ids))
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.poll_interval_seconds,
                )
            except TimeoutError:
                continue

    def stop(self) -> None:
        """Request shutdown of run_forever."""

        self._stop_event.set()


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    return {
        ticker.upper().strip(): float(weight)
        for ticker, weight in weights.items()
        if ticker.strip()
    }

