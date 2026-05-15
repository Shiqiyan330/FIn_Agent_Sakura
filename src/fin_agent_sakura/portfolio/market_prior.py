"""Market-equilibrium prior return estimation for Black-Litterman models."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Literal, Sequence

import pandas as pd

from fin_agent_sakura.config import get_tushare_config
from fin_agent_sakura.data import MarketDataClientFactory
from fin_agent_sakura.storage import CacheStore


MarketName = Literal["us", "cn"]


@dataclass(frozen=True, slots=True)
class MarketPriorResult:
    """Outputs needed to initialize a Black-Litterman prior."""

    tickers: list[str]
    prices: pd.DataFrame
    covariance_matrix: pd.DataFrame
    market_caps: pd.Series
    market_weights: pd.Series
    implied_prior_returns: pd.Series
    risk_aversion: float
    risk_free_rate: float


async def build_market_equilibrium_prior_async(
    tickers: Sequence[str],
    *,
    market: MarketName = "us",
    years: int = 5,
    risk_aversion: float = 2.5,
    risk_free_rate: float = 0.02,
    end_date: str | date | datetime | None = None,
) -> MarketPriorResult:
    """Build Black-Litterman market-equilibrium prior returns.

    Args:
        tickers: Stock ticker symbols to include in the investable universe.
        market: Market data backend. Use "us" for US equities and "cn" for
            China A-shares.
        years: Number of trailing years of daily price history to use.
        risk_aversion: Market risk-aversion coefficient, commonly denoted
            lambda or delta in Black-Litterman literature.
        risk_free_rate: Annualized risk-free rate as a decimal.

    Returns:
        A MarketPriorResult containing aligned historical prices, annualized
        covariance matrix, current market caps, market-cap weights, and implied
        equilibrium prior returns.
    """

    clean_tickers = _normalize_tickers(tickers)
    if years <= 0:
        raise ValueError("years must be positive")
    if risk_aversion <= 0:
        raise ValueError("risk_aversion must be positive")

    end = _coerce_date(end_date) or date.today()
    start = end - timedelta(days=365 * years + 7)
    client = MarketDataClientFactory.get_client(market)

    ohlcv_frames = await asyncio.gather(
        *[
            client.fetch_ohlcv(ticker, start=start, end=end, interval="1d", adjusted=True)
            for ticker in clean_tickers
        ]
    )
    prices = _build_price_matrix(clean_tickers, ohlcv_frames)
    covariance_matrix = _sample_covariance(prices)
    market_caps = await _fetch_market_caps(clean_tickers, market, end_date=end)
    market_weights = market_caps / market_caps.sum()
    implied_prior_returns = _market_implied_prior_returns(
        market_caps=market_caps,
        risk_aversion=risk_aversion,
        covariance_matrix=covariance_matrix,
        risk_free_rate=risk_free_rate,
    )

    return MarketPriorResult(
        tickers=clean_tickers,
        prices=prices,
        covariance_matrix=covariance_matrix,
        market_caps=market_caps,
        market_weights=market_weights,
        implied_prior_returns=implied_prior_returns,
        risk_aversion=risk_aversion,
        risk_free_rate=risk_free_rate,
    )


def build_market_equilibrium_prior(
    tickers: Sequence[str],
    *,
    market: MarketName = "us",
    years: int = 5,
    risk_aversion: float = 2.5,
    risk_free_rate: float = 0.02,
    end_date: str | date | datetime | None = None,
) -> MarketPriorResult:
    """Synchronous wrapper for build_market_equilibrium_prior_async."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            build_market_equilibrium_prior_async(
                tickers,
                market=market,
                years=years,
                risk_aversion=risk_aversion,
                risk_free_rate=risk_free_rate,
                end_date=end_date,
            )
        )
    raise RuntimeError(
        "build_market_equilibrium_prior cannot run inside an active event loop; "
        "await build_market_equilibrium_prior_async instead."
    )


def _build_price_matrix(tickers: Sequence[str], frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    price_series: list[pd.Series] = []
    for ticker, frame in zip(tickers, frames, strict=True):
        if "date" not in frame.columns or "close" not in frame.columns:
            raise ValueError(f"OHLCV frame for {ticker} must include date and close columns")

        series = frame[["date", "close"]].copy()
        series["date"] = pd.to_datetime(series["date"])
        series["close"] = pd.to_numeric(series["close"], errors="coerce")
        series = series.dropna(subset=["date", "close"]).drop_duplicates("date", keep="last")
        series = series.sort_values("date").set_index("date")["close"]
        series.name = ticker
        price_series.append(series)

    prices = pd.concat(price_series, axis=1).sort_index()
    prices = prices.ffill().dropna(how="any")
    if prices.empty:
        raise ValueError("No overlapping price history available for the requested tickers")
    return prices


def _sample_covariance(prices: pd.DataFrame) -> pd.DataFrame:
    try:
        from pypfopt import risk_models
    except ImportError as exc:
        raise RuntimeError("Install PyPortfolioOpt with `pip install -e .[portfolio]`.") from exc

    covariance = risk_models.sample_cov(prices, frequency=252)
    if not isinstance(covariance, pd.DataFrame):
        covariance = pd.DataFrame(covariance, index=prices.columns, columns=prices.columns)
    return covariance


def _market_implied_prior_returns(
    *,
    market_caps: pd.Series,
    risk_aversion: float,
    covariance_matrix: pd.DataFrame,
    risk_free_rate: float,
) -> pd.Series:
    try:
        from pypfopt.black_litterman import market_implied_prior_returns
    except ImportError as exc:
        raise RuntimeError("Install PyPortfolioOpt with `pip install -e .[portfolio]`.") from exc

    prior = market_implied_prior_returns(
        market_caps=market_caps,
        risk_aversion=risk_aversion,
        cov_matrix=covariance_matrix,
        risk_free_rate=risk_free_rate,
    )
    if not isinstance(prior, pd.Series):
        prior = pd.Series(prior, index=covariance_matrix.index)
    return prior.reindex(covariance_matrix.index)


async def _fetch_market_caps(
    tickers: Sequence[str],
    market: MarketName,
    *,
    end_date: date,
) -> pd.Series:
    if market == "cn":
        market_caps = await asyncio.gather(
            *[asyncio.to_thread(_fetch_tushare_market_cap, ticker, end_date) for ticker in tickers]
        )
        series = pd.Series(market_caps, index=list(tickers), dtype="float64")
        if series.isna().any() or (series <= 0).any():
            bad = ", ".join(series[series.isna() | (series <= 0)].index)
            raise ValueError(f"Missing or invalid A-share market cap for: {bad}")
        return series

    if market != "us":
        raise ValueError(f"Unsupported market: {market}")

    market_caps = await asyncio.gather(*[asyncio.to_thread(_fetch_yfinance_market_cap, t) for t in tickers])
    series = pd.Series(market_caps, index=list(tickers), dtype="float64")
    if series.isna().any() or (series <= 0).any():
        bad = ", ".join(series[series.isna() | (series <= 0)].index)
        raise ValueError(f"Missing or invalid market cap for: {bad}")
    return series


def _fetch_tushare_market_cap(ticker: str, end_date: date) -> float:
    try:
        import tushare as ts
    except ImportError as exc:
        raise RuntimeError("Install tushare with `pip install -e .[market-data-cn]`.") from exc

    cfg = get_tushare_config()
    if not cfg.token:
        raise RuntimeError("TUSHARE_TOKEN is required to fetch A-share market caps.")

    cache = CacheStore()
    cache_key = f"daily_basic_market_cap_{_to_tushare_code(ticker)}_{end_date.strftime('%Y%m%d')}"
    cached = cache.get_json(cache_key, ttl_seconds=24 * 60 * 60)
    if cached is not None and cached.get("market_cap"):
        return float(cached["market_cap"])

    pro = ts.pro_api(cfg.token)
    pro._DataApi__token = cfg.token
    if cfg.http_url:
        pro._DataApi__http_url = cfg.http_url

    lookup_end = end_date.strftime("%Y%m%d")
    lookup_start = (end_date - timedelta(days=45)).strftime("%Y%m%d")
    frame = pro.daily_basic(
        ts_code=_to_tushare_code(ticker),
        start_date=lookup_start,
        end_date=lookup_end,
        fields="ts_code,trade_date,total_mv,circ_mv",
    )
    if frame.empty:
        raise ValueError(f"TuShare daily_basic returned no market-cap rows for {ticker}")
    frame = frame.sort_values("trade_date", ascending=False)
    row = frame.iloc[0]
    market_cap = row.get("total_mv")
    if pd.isna(market_cap) or float(market_cap) <= 0:
        market_cap = row.get("circ_mv")
    if pd.isna(market_cap) or float(market_cap) <= 0:
        raise ValueError(f"TuShare daily_basic returned invalid market cap for {ticker}")
    value = float(market_cap)
    cache.set_json(cache_key, {"ticker": ticker, "end_date": end_date.isoformat(), "market_cap": value})
    return value


def _fetch_yfinance_market_cap(ticker: str) -> float:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Install yfinance with `pip install -e .[market-data-us]`.") from exc

    info = yf.Ticker(ticker).fast_info
    market_cap = getattr(info, "market_cap", None)
    if market_cap is None:
        try:
            market_cap = info["market_cap"]
        except (KeyError, TypeError):
            market_cap = None

    if market_cap is None:
        market_cap = yf.Ticker(ticker).info.get("marketCap")

    return float(market_cap) if market_cap is not None else float("nan")


def _normalize_tickers(tickers: Sequence[str]) -> list[str]:
    cleaned = [ticker.upper().strip() for ticker in tickers if ticker.strip()]
    if not cleaned:
        raise ValueError("tickers must contain at least one symbol")

    seen: set[str] = set()
    unique: list[str] = []
    for ticker in cleaned:
        if ticker in seen:
            continue
        seen.add(ticker)
        unique.append(ticker)

    if len(unique) < 2:
        raise ValueError("At least two unique tickers are required to estimate covariance")
    return unique


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


def _coerce_date(value: str | date | datetime | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.fromisoformat(value).date()
