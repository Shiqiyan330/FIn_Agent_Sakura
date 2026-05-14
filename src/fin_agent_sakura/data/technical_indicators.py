"""Technical indicator calculations for OHLCV market data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal, Sequence

import pandas as pd


IndicatorEngine = Literal["auto", "pandas-ta", "talib", "pandas"]


class TechnicalIndicatorError(ValueError):
    """Raised when technical indicators cannot be calculated."""


@dataclass(frozen=True, slots=True)
class TechnicalIndicators:
    """Calculate standard technical indicators from an OHLCV DataFrame.

    The class is intentionally stateless: it does not cache input frames or
    intermediate series, which keeps repeated calculations predictable in
    long-running agent workflows.
    """

    engine: IndicatorEngine = "auto"
    rsi_length: int = 14
    bollinger_length: int = 20
    bollinger_std: float = 2.0
    ma_windows: Sequence[int] = (50, 200)
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    REQUIRED_COLUMNS: ClassVar[set[str]] = {"close"}

    def calculate(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Return a new DataFrame with RSI, MACD, Bollinger Bands, and MAs."""

        self._validate_input(ohlcv)
        if ohlcv.empty:
            return ohlcv.copy(deep=False)

        if "symbol" in ohlcv.columns:
            frames = [
                self._calculate_one_symbol(frame)
                for _, frame in ohlcv.groupby("symbol", sort=False, observed=True)
            ]
            return pd.concat(frames, axis=0, copy=False)

        return self._calculate_one_symbol(ohlcv)

    def _calculate_one_symbol(self, frame: pd.DataFrame) -> pd.DataFrame:
        working = frame.copy(deep=False)
        if "date" in working.columns:
            working = working.sort_values("date", kind="mergesort")

        close = pd.to_numeric(working["close"], errors="coerce")
        indicators = self._calculate_with_selected_engine(close)

        result = pd.concat([working.reset_index(drop=True), indicators.reset_index(drop=True)], axis=1)
        return result

    def _calculate_with_selected_engine(self, close: pd.Series) -> pd.DataFrame:
        if self.engine in {"auto", "pandas-ta"}:
            try:
                return self._calculate_with_pandas_ta(close)
            except ImportError:
                if self.engine == "pandas-ta":
                    raise

        if self.engine in {"auto", "talib"}:
            try:
                return self._calculate_with_talib(close)
            except ImportError:
                if self.engine == "talib":
                    raise

        return self._calculate_with_pandas(close)

    def _calculate_with_pandas_ta(self, close: pd.Series) -> pd.DataFrame:
        import pandas_ta as ta

        indicators = pd.DataFrame(index=close.index)
        indicators[f"rsi_{self.rsi_length}"] = ta.rsi(close, length=self.rsi_length)

        macd = ta.macd(
            close,
            fast=self.macd_fast,
            slow=self.macd_slow,
            signal=self.macd_signal,
        )
        if macd is not None and not macd.empty:
            indicators["macd"] = self._first_prefixed_column(macd, "MACD_")
            indicators["macd_hist"] = self._first_prefixed_column(macd, "MACDh_")
            indicators["macd_signal"] = self._first_prefixed_column(macd, "MACDs_")

        bbands = ta.bbands(close, length=self.bollinger_length, std=self.bollinger_std)
        if bbands is not None and not bbands.empty:
            indicators["bb_lower"] = self._first_prefixed_column(bbands, "BBL_")
            indicators["bb_middle"] = self._first_prefixed_column(bbands, "BBM_")
            indicators["bb_upper"] = self._first_prefixed_column(bbands, "BBU_")

        for window in self.ma_windows:
            indicators[f"ma_{window}"] = ta.sma(close, length=window)

        return self._ensure_indicator_columns(indicators)

    def _calculate_with_talib(self, close: pd.Series) -> pd.DataFrame:
        import talib

        close_values = close.to_numpy(dtype="float64", copy=False)
        macd, macd_signal, macd_hist = talib.MACD(
            close_values,
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal,
        )
        bb_upper, bb_middle, bb_lower = talib.BBANDS(
            close_values,
            timeperiod=self.bollinger_length,
            nbdevup=self.bollinger_std,
            nbdevdn=self.bollinger_std,
            matype=0,
        )

        indicators = pd.DataFrame(index=close.index)
        indicators[f"rsi_{self.rsi_length}"] = talib.RSI(close_values, timeperiod=self.rsi_length)
        indicators["macd"] = macd
        indicators["macd_signal"] = macd_signal
        indicators["macd_hist"] = macd_hist
        indicators["bb_lower"] = bb_lower
        indicators["bb_middle"] = bb_middle
        indicators["bb_upper"] = bb_upper
        for window in self.ma_windows:
            indicators[f"ma_{window}"] = talib.SMA(close_values, timeperiod=window)

        return self._ensure_indicator_columns(indicators)

    def _calculate_with_pandas(self, close: pd.Series) -> pd.DataFrame:
        indicators = pd.DataFrame(index=close.index)

        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / self.rsi_length, min_periods=self.rsi_length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / self.rsi_length, min_periods=self.rsi_length, adjust=False).mean()
        relative_strength = avg_gain / avg_loss
        indicators[f"rsi_{self.rsi_length}"] = 100 - (100 / (1 + relative_strength))

        ema_fast = close.ewm(span=self.macd_fast, adjust=False, min_periods=self.macd_fast).mean()
        ema_slow = close.ewm(span=self.macd_slow, adjust=False, min_periods=self.macd_slow).mean()
        indicators["macd"] = ema_fast - ema_slow
        indicators["macd_signal"] = indicators["macd"].ewm(
            span=self.macd_signal,
            adjust=False,
            min_periods=self.macd_signal,
        ).mean()
        indicators["macd_hist"] = indicators["macd"] - indicators["macd_signal"]

        rolling_close = close.rolling(window=self.bollinger_length, min_periods=self.bollinger_length)
        indicators["bb_middle"] = rolling_close.mean()
        rolling_std = rolling_close.std(ddof=0)
        indicators["bb_upper"] = indicators["bb_middle"] + self.bollinger_std * rolling_std
        indicators["bb_lower"] = indicators["bb_middle"] - self.bollinger_std * rolling_std

        for window in self.ma_windows:
            indicators[f"ma_{window}"] = close.rolling(window=window, min_periods=window).mean()

        return self._ensure_indicator_columns(indicators)

    def _ensure_indicator_columns(self, indicators: pd.DataFrame) -> pd.DataFrame:
        expected_columns = [
            f"rsi_{self.rsi_length}",
            "macd",
            "macd_signal",
            "macd_hist",
            "bb_lower",
            "bb_middle",
            "bb_upper",
            *[f"ma_{window}" for window in self.ma_windows],
        ]
        for column in expected_columns:
            if column not in indicators.columns:
                indicators[column] = pd.NA
        return indicators[expected_columns]

    def _validate_input(self, ohlcv: pd.DataFrame) -> None:
        missing = self.REQUIRED_COLUMNS.difference(ohlcv.columns)
        if missing:
            columns = ", ".join(sorted(missing))
            raise TechnicalIndicatorError(f"OHLCV DataFrame missing required columns: {columns}")
        if self.rsi_length <= 0:
            raise TechnicalIndicatorError("rsi_length must be positive")
        if self.bollinger_length <= 0:
            raise TechnicalIndicatorError("bollinger_length must be positive")
        if self.bollinger_std <= 0:
            raise TechnicalIndicatorError("bollinger_std must be positive")
        if self.macd_fast <= 0 or self.macd_slow <= 0 or self.macd_signal <= 0:
            raise TechnicalIndicatorError("MACD periods must be positive")
        if self.macd_fast >= self.macd_slow:
            raise TechnicalIndicatorError("macd_fast must be smaller than macd_slow")
        if any(window <= 0 for window in self.ma_windows):
            raise TechnicalIndicatorError("Moving-average windows must be positive")

    @staticmethod
    def _first_prefixed_column(frame: pd.DataFrame, prefix: str) -> pd.Series:
        for column in frame.columns:
            if str(column).startswith(prefix):
                return frame[column]
        raise TechnicalIndicatorError(f"Indicator output missing column prefix: {prefix}")

