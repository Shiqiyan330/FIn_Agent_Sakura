"""Cross-market data access clients.

The module exposes one abstract interface for market data and financial
statements, then provides concrete clients for US equities and China A-shares.
The clients are asynchronous at the public boundary while wrapping provider
SDKs that are mostly synchronous.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Literal, ParamSpec, TypeAlias, TypeVar
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from fin_agent_sakura.config import load_dotenv
from fin_agent_sakura.storage import CacheStore

if TYPE_CHECKING:
    import pandas as pd


DateLike: TypeAlias = str | date | datetime | None
FinancialPeriod: TypeAlias = Literal["annual", "quarterly"]
OHLCVInterval: TypeAlias = Literal["1d", "1wk", "1mo"]
StatementKind: TypeAlias = Literal["balance_sheet", "cash_flow", "income_statement"]
T = TypeVar("T")
P = ParamSpec("P")


class Market(StrEnum):
    """Supported market regions."""

    US = "us"
    CN = "cn"


class DataProviderError(RuntimeError):
    """Base error raised when an upstream market-data provider fails."""


class RateLimitError(DataProviderError):
    """Raised when a provider throttles the request."""


class MissingDependencyError(DataProviderError):
    """Raised when an optional provider package is not installed."""


class ConfigurationError(DataProviderError):
    """Raised when a provider needs credentials that are not configured."""


@dataclass(frozen=True, slots=True)
class RetryConfig:
    """Retry settings for API throttling and transient provider failures."""

    max_attempts: int = 3
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0
    backoff_factor: float = 2.0
    jitter_seconds: float = 0.25


class SingletonMeta(type):
    """Simple process-local singleton metaclass."""

    _instances: dict[type[Any], Any] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class SingletonABCMeta(SingletonMeta, type(ABC)):
    """Singleton metaclass compatible with abstract base classes."""


class MarketDataClient(ABC, metaclass=SingletonABCMeta):
    """Abstract market data client used by the rest of the investment system."""

    def __init__(self, retry_config: RetryConfig | None = None) -> None:
        if getattr(self, "_initialized", False):
            return
        self.retry_config = retry_config or RetryConfig()
        self._initialized = True

    @abstractmethod
    async def fetch_ohlcv(
        self,
        symbol: str,
        start: DateLike = None,
        end: DateLike = None,
        interval: OHLCVInterval = "1d",
        adjusted: bool = True,
    ) -> "pd.DataFrame":
        """Fetch OHLCV bars for a symbol."""

    @abstractmethod
    async def fetch_balance_sheet(
        self,
        symbol: str,
        period: FinancialPeriod = "annual",
        limit: int = 5,
    ) -> "pd.DataFrame":
        """Fetch a balance sheet."""

    @abstractmethod
    async def fetch_cash_flow(
        self,
        symbol: str,
        period: FinancialPeriod = "annual",
        limit: int = 5,
    ) -> "pd.DataFrame":
        """Fetch a cash-flow statement."""

    @abstractmethod
    async def fetch_income_statement(
        self,
        symbol: str,
        period: FinancialPeriod = "annual",
        limit: int = 5,
    ) -> "pd.DataFrame":
        """Fetch an income statement."""

    async def _with_retry(
        self,
        operation_name: str,
        fn: Callable[P, T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        """Run a blocking provider call with exponential backoff."""

        config = self.retry_config
        last_error: Exception | None = None

        for attempt in range(1, config.max_attempts + 1):
            try:
                return await asyncio.to_thread(fn, *args, **kwargs)
            except (MissingDependencyError, ConfigurationError, ValueError):
                raise
            except Exception as exc:
                last_error = exc
                if attempt >= config.max_attempts:
                    break

                delay = min(
                    config.initial_delay_seconds * (config.backoff_factor ** (attempt - 1)),
                    config.max_delay_seconds,
                )
                delay += random.uniform(0, config.jitter_seconds)
                await asyncio.sleep(delay)

        raise DataProviderError(
            f"{operation_name} failed after {config.max_attempts} attempts"
        ) from last_error


class USMarketDataClient(MarketDataClient):
    """US equity data client backed by yfinance and Financial Modeling Prep."""

    _FMP_BASE_URL: ClassVar[str] = "https://financialmodelingprep.com/api/v3"
    _FMP_ENDPOINTS: ClassVar[dict[StatementKind, str]] = {
        "balance_sheet": "balance-sheet-statement",
        "cash_flow": "cash-flow-statement",
        "income_statement": "income-statement",
    }

    def __init__(
        self,
        fmp_api_key: str | None = None,
        retry_config: RetryConfig | None = None,
    ) -> None:
        if getattr(self, "_initialized", False):
            return
        load_dotenv()
        super().__init__(retry_config=retry_config)
        self.fmp_api_key = fmp_api_key or os.getenv("FMP_API_KEY")

    async def fetch_ohlcv(
        self,
        symbol: str,
        start: DateLike = None,
        end: DateLike = None,
        interval: OHLCVInterval = "1d",
        adjusted: bool = True,
    ) -> "pd.DataFrame":
        return await self._with_retry(
            f"yfinance OHLCV fetch for {symbol}",
            self._fetch_yfinance_ohlcv,
            symbol,
            start,
            end,
            interval,
            adjusted,
        )

    async def fetch_balance_sheet(
        self,
        symbol: str,
        period: FinancialPeriod = "annual",
        limit: int = 5,
    ) -> "pd.DataFrame":
        return await self._fetch_fmp_statement(symbol, "balance_sheet", period, limit)

    async def fetch_cash_flow(
        self,
        symbol: str,
        period: FinancialPeriod = "annual",
        limit: int = 5,
    ) -> "pd.DataFrame":
        return await self._fetch_fmp_statement(symbol, "cash_flow", period, limit)

    async def fetch_income_statement(
        self,
        symbol: str,
        period: FinancialPeriod = "annual",
        limit: int = 5,
    ) -> "pd.DataFrame":
        return await self._fetch_fmp_statement(symbol, "income_statement", period, limit)

    async def _fetch_fmp_statement(
        self,
        symbol: str,
        statement_kind: StatementKind,
        period: FinancialPeriod,
        limit: int,
    ) -> "pd.DataFrame":
        if self.fmp_api_key:
            return await self._with_retry(
                f"FMP {statement_kind} fetch for {symbol}",
                self._fetch_fmp_statement_sync,
                symbol,
                statement_kind,
                period,
                limit,
            )

        return await self._with_retry(
            f"yfinance fallback {statement_kind} fetch for {symbol}",
            self._fetch_yfinance_statement,
            symbol,
            statement_kind,
            period,
            limit,
        )

    def _fetch_yfinance_ohlcv(
        self,
        symbol: str,
        start: DateLike,
        end: DateLike,
        interval: OHLCVInterval,
        adjusted: bool,
    ) -> "pd.DataFrame":
        yf = _import_yfinance()
        data = yf.download(
            tickers=symbol,
            start=_format_date(start, "%Y-%m-%d"),
            end=_format_date(end, "%Y-%m-%d"),
            interval=interval,
            auto_adjust=adjusted,
            progress=False,
            threads=False,
        )
        return _normalize_ohlcv_frame(data, symbol=symbol)

    def _fetch_fmp_statement_sync(
        self,
        symbol: str,
        statement_kind: StatementKind,
        period: FinancialPeriod,
        limit: int,
    ) -> "pd.DataFrame":
        pd = _import_pandas()
        endpoint = self._FMP_ENDPOINTS[statement_kind]
        payload = self._fmp_get(
            endpoint=f"{endpoint}/{symbol.upper()}",
            params={
                "period": period,
                "limit": str(limit),
                "apikey": self.fmp_api_key or "",
            },
        )
        frame = pd.DataFrame(payload)
        if frame.empty:
            raise DataProviderError(f"FMP returned no {statement_kind} rows for {symbol}")
        return frame

    def _fetch_yfinance_statement(
        self,
        symbol: str,
        statement_kind: StatementKind,
        period: FinancialPeriod,
        limit: int,
    ) -> "pd.DataFrame":
        yf = _import_yfinance()
        ticker = yf.Ticker(symbol)
        attr_by_statement: dict[tuple[StatementKind, FinancialPeriod], str] = {
            ("balance_sheet", "annual"): "balance_sheet",
            ("balance_sheet", "quarterly"): "quarterly_balance_sheet",
            ("cash_flow", "annual"): "cashflow",
            ("cash_flow", "quarterly"): "quarterly_cashflow",
            ("income_statement", "annual"): "financials",
            ("income_statement", "quarterly"): "quarterly_financials",
        }
        statement = getattr(ticker, attr_by_statement[(statement_kind, period)])
        frame = statement.T.head(limit).reset_index(names="period")
        if frame.empty:
            raise DataProviderError(f"yfinance returned no {statement_kind} rows for {symbol}")
        return frame

    def _fmp_get(self, endpoint: str, params: dict[str, str]) -> list[dict[str, Any]]:
        query = urlencode(params)
        url = f"{self._FMP_BASE_URL}/{endpoint}?{query}"
        request = Request(url, headers={"User-Agent": "fin-agent-sakura/0.1"})

        try:
            with urlopen(request, timeout=30) as response:
                body = response.read().decode("utf-8")
                data = json.loads(body)
        except HTTPError as exc:
            if exc.code == 429:
                raise RateLimitError("FMP rate limit exceeded") from exc
            raise DataProviderError(f"FMP HTTP error {exc.code}") from exc
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise DataProviderError("FMP request failed") from exc

        if isinstance(data, dict) and "Error Message" in data:
            raise DataProviderError(str(data["Error Message"]))
        if not isinstance(data, list):
            raise DataProviderError("FMP returned an unexpected payload")
        return data


class CNMarketDataClient(MarketDataClient):
    """China A-share data client backed by AkShare and TuShare."""

    _AK_INTERVALS: ClassVar[dict[OHLCVInterval, str]] = {
        "1d": "daily",
        "1wk": "weekly",
        "1mo": "monthly",
    }
    _AK_STATEMENT_SYMBOLS: ClassVar[dict[StatementKind, str]] = {
        "balance_sheet": "资产负债表",
        "cash_flow": "现金流量表",
        "income_statement": "利润表",
    }
    _OHLCV_CACHE_TTL_SECONDS: ClassVar[int] = 24 * 60 * 60

    def __init__(
        self,
        tushare_token: str | None = None,
        tushare_http_url: str | None = None,
        retry_config: RetryConfig | None = None,
    ) -> None:
        if getattr(self, "_initialized", False):
            if tushare_token:
                self.tushare_token = tushare_token
            if tushare_http_url:
                self.tushare_http_url = tushare_http_url
            return
        load_dotenv()
        super().__init__(retry_config=retry_config)
        self.tushare_token = tushare_token or os.getenv("TUSHARE_TOKEN")
        self.tushare_http_url = tushare_http_url or os.getenv("TUSHARE_HTTP_URL")
        self.provider_mode = os.getenv("A_SHARE_DATA_PROVIDER", "tushare").strip().lower()
        self.cache = CacheStore()

    async def fetch_ohlcv(
        self,
        symbol: str,
        start: DateLike = None,
        end: DateLike = None,
        interval: OHLCVInterval = "1d",
        adjusted: bool = True,
    ) -> "pd.DataFrame":
        if self.provider_mode != "akshare":
            if not self.tushare_token:
                raise ConfigurationError("TUSHARE_TOKEN is required for A-share OHLCV data.")
            if interval != "1d":
                raise ConfigurationError("TuShare OHLCV currently supports only daily interval in this client.")
            return await self._with_retry(
                f"TuShare daily OHLCV fetch for {symbol}",
                self._fetch_tushare_ohlcv,
                symbol,
                start,
                end,
            )

        try:
            return await self._with_retry(
                f"AkShare OHLCV fetch for {symbol}",
                self._fetch_akshare_ohlcv,
                symbol,
                start,
                end,
                interval,
                adjusted,
            )
        except Exception:
            if not self.tushare_token or interval != "1d":
                raise
            return await self._with_retry(
                f"TuShare daily OHLCV fallback fetch for {symbol}",
                self._fetch_tushare_ohlcv,
                symbol,
                start,
                end,
            )

    async def fetch_balance_sheet(
        self,
        symbol: str,
        period: FinancialPeriod = "annual",
        limit: int = 5,
    ) -> "pd.DataFrame":
        return await self._fetch_cn_statement(symbol, "balance_sheet", period, limit)

    async def fetch_cash_flow(
        self,
        symbol: str,
        period: FinancialPeriod = "annual",
        limit: int = 5,
    ) -> "pd.DataFrame":
        return await self._fetch_cn_statement(symbol, "cash_flow", period, limit)

    async def fetch_income_statement(
        self,
        symbol: str,
        period: FinancialPeriod = "annual",
        limit: int = 5,
    ) -> "pd.DataFrame":
        return await self._fetch_cn_statement(symbol, "income_statement", period, limit)

    async def _fetch_cn_statement(
        self,
        symbol: str,
        statement_kind: StatementKind,
        period: FinancialPeriod,
        limit: int,
    ) -> "pd.DataFrame":
        if self.provider_mode != "akshare":
            if not self.tushare_token:
                raise ConfigurationError(f"TUSHARE_TOKEN is required for A-share {statement_kind} data.")
            return await self._with_retry(
                f"TuShare {statement_kind} fetch for {symbol}",
                self._fetch_tushare_statement,
                symbol,
                statement_kind,
                period,
                limit,
            )

        return await self._with_retry(
            f"AkShare fallback {statement_kind} fetch for {symbol}",
            self._fetch_akshare_statement,
            symbol,
            statement_kind,
            limit,
        )

    def _fetch_akshare_ohlcv(
        self,
        symbol: str,
        start: DateLike,
        end: DateLike,
        interval: OHLCVInterval,
        adjusted: bool,
    ) -> "pd.DataFrame":
        ak = _import_akshare()
        frame = ak.stock_zh_a_hist(
            symbol=_to_akshare_symbol(symbol),
            period=self._AK_INTERVALS[interval],
            start_date=_format_date(start, "%Y%m%d") or "19900101",
            end_date=_format_date(end, "%Y%m%d") or datetime.now().strftime("%Y%m%d"),
            adjust="qfq" if adjusted else "",
        )
        renamed = frame.rename(
            columns={
                "日期": "date",
                "开盘": "open",
                "最高": "high",
                "最低": "low",
                "收盘": "close",
                "成交量": "volume",
                "成交额": "amount",
                "振幅": "amplitude",
                "涨跌幅": "pct_change",
                "涨跌额": "change",
                "换手率": "turnover_rate",
            }
        )
        return _normalize_ohlcv_frame(renamed, symbol=symbol)

    def _fetch_tushare_ohlcv(
        self,
        symbol: str,
        start: DateLike,
        end: DateLike,
    ) -> "pd.DataFrame":
        cache_key = (
            f"ohlcv_tushare_{_to_tushare_code(symbol)}_"
            f"{_format_date(start, '%Y%m%d') or '19900101'}_"
            f"{_format_date(end, '%Y%m%d') or datetime.now().strftime('%Y%m%d')}"
        )
        cached = self.cache.get_dataframe(cache_key, ttl_seconds=self._OHLCV_CACHE_TTL_SECONDS)
        if cached is not None:
            return cached

        pro = self._tushare_pro()
        frame = pro.daily(
            ts_code=_to_tushare_code(symbol),
            start_date=_format_date(start, "%Y%m%d") or "19900101",
            end_date=_format_date(end, "%Y%m%d") or datetime.now().strftime("%Y%m%d"),
        )
        renamed = frame.rename(
            columns={
                "trade_date": "date",
                "vol": "volume",
            }
        )
        if "date" in renamed.columns:
            renamed = renamed.sort_values("date")
        normalized = _normalize_ohlcv_frame(renamed, symbol=symbol)
        self.cache.set_dataframe(cache_key, normalized)
        return normalized

    def _fetch_tushare_statement(
        self,
        symbol: str,
        statement_kind: StatementKind,
        period: FinancialPeriod,
        limit: int,
    ) -> "pd.DataFrame":
        pro = self._tushare_pro()
        ts_code = _to_tushare_code(symbol)
        endpoint_by_statement: dict[StatementKind, str] = {
            "balance_sheet": "balancesheet",
            "cash_flow": "cashflow",
            "income_statement": "income",
        }
        endpoint = getattr(pro, endpoint_by_statement[statement_kind])
        frame = endpoint(ts_code=ts_code)
        if period == "annual" and "report_type" in frame.columns:
            frame = frame[frame["report_type"].astype(str).isin({"1", "年报"})]
        if "end_date" in frame.columns:
            frame = frame.sort_values("end_date", ascending=False)
        frame = frame.head(limit).reset_index(drop=True)
        if frame.empty:
            raise DataProviderError(f"TuShare returned no {statement_kind} rows for {symbol}")
        return frame

    def _tushare_pro(self) -> Any:
        ts = _import_tushare()
        if not self.tushare_token:
            raise ConfigurationError("TUSHARE_TOKEN is required for TuShare")

        pro = ts.pro_api(self.tushare_token)
        pro._DataApi__token = self.tushare_token
        if self.tushare_http_url:
            pro._DataApi__http_url = self.tushare_http_url
        return pro

    def _fetch_akshare_statement(
        self,
        symbol: str,
        statement_kind: StatementKind,
        limit: int,
    ) -> "pd.DataFrame":
        ak = _import_akshare()
        stock = _to_sina_symbol(symbol)
        symbol_name = self._AK_STATEMENT_SYMBOLS[statement_kind]
        try:
            frame = ak.stock_financial_report_sina(stock=stock, symbol=symbol_name)
        except TypeError:
            frame = ak.stock_financial_report_sina(stock, symbol_name)
        if "报表日期" in frame.columns:
            frame = frame.sort_values("报表日期", ascending=False)
        frame = frame.head(limit).reset_index(drop=True)
        if frame.empty:
            raise DataProviderError(f"AkShare returned no {statement_kind} rows for {symbol}")
        return frame


class MarketDataClientFactory:
    """Factory for retrieving singleton market data clients."""

    @classmethod
    def get_client(
        cls,
        market: Market | str,
        *,
        fmp_api_key: str | None = None,
        tushare_token: str | None = None,
        tushare_http_url: str | None = None,
        retry_config: RetryConfig | None = None,
    ) -> MarketDataClient:
        market_value = Market(str(market).lower())
        if market_value is Market.US:
            return USMarketDataClient(fmp_api_key=fmp_api_key, retry_config=retry_config)
        if market_value is Market.CN:
            return CNMarketDataClient(
                tushare_token=tushare_token,
                tushare_http_url=tushare_http_url,
                retry_config=retry_config,
            )
        raise ValueError(f"Unsupported market: {market}")


def _import_pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise MissingDependencyError("Install pandas to use market data clients") from exc
    return pd


def _import_yfinance() -> Any:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise MissingDependencyError("Install yfinance to use USMarketDataClient") from exc
    return yf


def _import_akshare() -> Any:
    try:
        import akshare as ak
    except ImportError as exc:
        raise MissingDependencyError("Install akshare to use CNMarketDataClient") from exc
    return ak


def _import_tushare() -> Any:
    try:
        import tushare as ts
    except ImportError as exc:
        raise MissingDependencyError("Install tushare to use TuShare-backed statements") from exc
    return ts


def _format_date(value: DateLike, fmt: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.strftime(fmt)
    if isinstance(value, date):
        return value.strftime(fmt)
    parsed = datetime.fromisoformat(value)
    return parsed.strftime(fmt)


def _normalize_ohlcv_frame(frame: "pd.DataFrame", symbol: str) -> "pd.DataFrame":
    pd = _import_pandas()
    if frame.empty:
        raise DataProviderError(f"No OHLCV data returned for {symbol}")

    normalized = frame.copy()
    if isinstance(normalized.columns, pd.MultiIndex):
        normalized.columns = normalized.columns.get_level_values(0)

    normalized = normalized.rename(
        columns={
            "Date": "date",
            "Datetime": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    if "date" not in normalized.columns:
        normalized = normalized.reset_index().rename(columns={"index": "date"})
    else:
        normalized = normalized.reset_index(drop=True)

    normalized["symbol"] = symbol
    ordered = ["symbol", "date", "open", "high", "low", "close", "volume"]
    existing_ordered = [column for column in ordered if column in normalized.columns]
    remaining = [column for column in normalized.columns if column not in existing_ordered]
    return normalized[existing_ordered + remaining]


def _to_akshare_symbol(symbol: str) -> str:
    cleaned = symbol.strip().upper()
    if "." in cleaned:
        return cleaned.split(".", maxsplit=1)[0]
    if cleaned.startswith(("SH", "SZ", "BJ")):
        return cleaned[2:]
    return cleaned


def _to_tushare_code(symbol: str) -> str:
    cleaned = symbol.strip().upper()
    if "." in cleaned:
        code, exchange = cleaned.split(".", maxsplit=1)
        return f"{code}.{exchange}"
    if cleaned.startswith(("SH", "SZ", "BJ")):
        return f"{cleaned[2:]}.{cleaned[:2]}"
    if cleaned.startswith(("6", "9")):
        return f"{cleaned}.SH"
    if cleaned.startswith(("4", "8")):
        return f"{cleaned}.BJ"
    return f"{cleaned}.SZ"


def _to_sina_symbol(symbol: str) -> str:
    ts_code = _to_tushare_code(symbol)
    code, exchange = ts_code.split(".", maxsplit=1)
    return f"{exchange.lower()}{code}"


def monotonic_timestamp() -> float:
    """Expose a testable timestamp helper for future cache/rate-limit logic."""

    return time.monotonic()
