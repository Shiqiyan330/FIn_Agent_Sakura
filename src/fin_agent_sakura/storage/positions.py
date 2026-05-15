"""Local CSV-backed memory for portfolio positions."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd


class PositionMemory:
    """Persist the user's current holdings as a local CSV file."""

    COLUMNS = ["ticker", "name", "shares", "market_value", "weight"]

    def __init__(self, path: str | Path = "data/processed/current_positions.csv") -> None:
        self.path = Path(path)

    def load(self) -> pd.DataFrame:
        """Load the saved positions, returning an empty template when missing."""

        if not self.path.exists():
            return self.empty_frame()
        frame = pd.read_csv(self.path)
        return self.normalize(frame)

    def save(self, positions: pd.DataFrame) -> Path:
        """Normalize and save positions to disk using UTF-8 BOM for Excel."""

        normalized = self.normalize(positions)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        normalized.to_csv(self.path, index=False, encoding="utf-8-sig")
        return self.path

    def load_weights(self) -> dict[str, float]:
        """Return ticker-to-weight mapping from saved positions."""

        frame = self.load()
        if frame.empty:
            return {}
        weights = pd.to_numeric(frame["weight"], errors="coerce").fillna(0.0)
        weights = weights.where(weights <= 1.0, weights / 100.0)
        valid = frame.assign(weight=weights)
        valid = valid[(valid["ticker"].astype(str) != "") & (valid["weight"] > 0)]
        total = float(valid["weight"].sum())
        if total <= 0:
            return {}
        return {
            str(row["ticker"]).upper(): float(row["weight"]) / total
            for _, row in valid.iterrows()
        }

    def save_weights(self, weights: Mapping[str, float]) -> Path:
        """Save a simple ticker-weight mapping as positions."""

        frame = pd.DataFrame(
            [
                {
                    "ticker": ticker,
                    "name": "",
                    "shares": 0.0,
                    "market_value": 0.0,
                    "weight": weight,
                }
                for ticker, weight in weights.items()
            ]
        )
        return self.save(frame)

    def empty_frame(self) -> pd.DataFrame:
        """Return a blank positions table with the expected columns."""

        return pd.DataFrame(columns=self.COLUMNS)

    def template_frame(self) -> pd.DataFrame:
        """Return a user-editable starter table."""

        return pd.DataFrame(
            [
                {
                    "ticker": "600519.SH",
                    "name": "贵州茅台",
                    "shares": 0.0,
                    "market_value": 0.0,
                    "weight": 0.0,
                },
                {
                    "ticker": "000858.SZ",
                    "name": "五粮液",
                    "shares": 0.0,
                    "market_value": 0.0,
                    "weight": 0.0,
                },
            ],
            columns=self.COLUMNS,
        )

    def normalize(self, positions: pd.DataFrame) -> pd.DataFrame:
        """Coerce uploaded or edited holdings into the standard schema."""

        frame = positions.copy()
        for column in self.COLUMNS:
            if column not in frame.columns:
                frame[column] = "" if column in {"ticker", "name"} else 0.0
        frame = frame[self.COLUMNS]
        frame["ticker"] = frame["ticker"].fillna("").astype(str).str.strip().str.upper()
        frame["name"] = frame["name"].fillna("").astype(str).str.strip()
        frame["shares"] = pd.to_numeric(frame["shares"], errors="coerce").fillna(0.0)
        frame["market_value"] = pd.to_numeric(frame["market_value"], errors="coerce").fillna(0.0)
        frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce").fillna(0.0)
        frame["weight"] = frame["weight"].where(frame["weight"] <= 1.0, frame["weight"] / 100.0)

        market_value_total = float(frame["market_value"].sum())
        weight_total = float(frame["weight"].sum())
        if market_value_total > 0 and weight_total <= 0:
            frame["weight"] = frame["market_value"] / market_value_total

        frame = frame[frame["ticker"] != ""].reset_index(drop=True)
        return frame
