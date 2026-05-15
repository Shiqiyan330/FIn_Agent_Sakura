"""LangChain tool wrappers for financial data access."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Literal

import pandas as pd
from langchain_core.tools import tool

from fin_agent_sakura.config import get_llm_config
from fin_agent_sakura.data import MarketDataClientFactory


MarketName = Literal["us", "cn"]
StatementName = Literal["balance_sheet", "cash_flow", "income_statement", "all"]
PeriodName = Literal["annual", "quarterly"]


def get_financial_tools() -> list[Any]:
    """Return all LangChain tools exposed by the financial data layer."""

    return [get_financial_statements, get_recent_news]


def create_financial_tool_calling_llm(
    model: str | None = None,
    *,
    temperature: float = 0.0,
    **kwargs: Any,
) -> Any:
    """Create a ChatOpenAI model bound to the financial LangChain tools.

    Args:
        model: OpenAI chat model name. If omitted, the value is read from the
            OPENAI_CHAT_MODEL environment variable, then falls back to
            "gpt-4o-mini".
        temperature: Sampling temperature for the chat model. Use 0.0 for
            deterministic tool-selection behavior in financial workflows.
        **kwargs: Additional keyword arguments passed to ChatOpenAI.

    Returns:
        A LangChain chat model with the financial statement and recent news
        tools bound for tool calling.
    """

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError("Install LLM dependencies with `pip install -e .[tools]`.") from exc

    config = get_llm_config()
    llm = ChatOpenAI(
        model=model or config.chat_model,
        temperature=temperature,
        api_key=config.api_key,
        base_url=config.base_url,
        **kwargs,
    )
    return llm.bind_tools(get_financial_tools())


@tool(parse_docstring=True)
def get_financial_statements(
    ticker: str,
    market: MarketName = "us",
    statement: StatementName = "all",
    period: PeriodName = "annual",
    limit: int = 5,
) -> str:
    """Fetch standardized financial statements for a listed company.

    Use this tool when an analyst agent needs balance sheet, cash flow, or
    income statement data for fundamental analysis, DCF modeling, profitability
    checks, leverage checks, or financial health scoring.

    Args:
        ticker: The stock ticker symbol to analyze. Examples: "AAPL" for a US
            stock, "MSFT" for a US stock, or "600519.SH" for a China A-share.
        market: The market region for the ticker. Use "us" for US equities and
            "cn" for China A-shares.
        statement: Which financial statement to fetch. Use "balance_sheet",
            "cash_flow", "income_statement", or "all" to fetch all three.
        period: Reporting period. Use "annual" for annual reports or
            "quarterly" for quarterly statements.
        limit: Maximum number of reporting periods to return. A value of 5 is
            usually enough for multi-year trend analysis.

    Returns:
        A JSON string containing the requested statement data. The top-level
        keys are statement names, and values are lists of row dictionaries.
    """

    _validate_limit(limit)
    result = asyncio.run(_fetch_financial_statements(ticker, market, statement, period, limit))
    return _to_json(result)


@tool(parse_docstring=True)
def get_recent_news(ticker: str, market: MarketName = "us", limit: int = 5) -> str:
    """Fetch the most recent news items for a listed company.

    Use this tool when an analyst agent needs current market sentiment,
    catalysts, recent company developments, earnings coverage, or risk events
    related to a ticker.

    Args:
        ticker: The stock ticker symbol to search news for. Examples: "AAPL",
            "NVDA", "MSFT", or "600519.SH".
        market: The market region for the ticker. Use "us" for US equities and
            "cn" for China A-shares. Currently the implementation supports
            US equity news through yfinance.
        limit: Maximum number of news items to return. Use 5 for the latest
            five news items unless a smaller set is needed.

    Returns:
        A JSON string containing up to `limit` news items. Each item includes
        title, publisher, link, publication time, summary, and ticker metadata
        when available.
    """

    _validate_limit(limit)
    if market == "cn":
        return _to_json({"ticker": ticker, "market": market, "news": _fetch_akshare_news(ticker, limit)})

    return _to_json({"ticker": ticker, "market": market, "news": _fetch_yfinance_news(ticker, limit)})


async def _fetch_financial_statements(
    ticker: str,
    market: MarketName,
    statement: StatementName,
    period: PeriodName,
    limit: int,
) -> dict[str, list[dict[str, Any]]]:
    client = MarketDataClientFactory.get_client(market)

    tasks: dict[str, Any] = {}
    if statement in {"balance_sheet", "all"}:
        tasks["balance_sheet"] = client.fetch_balance_sheet(ticker, period=period, limit=limit)
    if statement in {"cash_flow", "all"}:
        tasks["cash_flow"] = client.fetch_cash_flow(ticker, period=period, limit=limit)
    if statement in {"income_statement", "all"}:
        tasks["income_statement"] = client.fetch_income_statement(ticker, period=period, limit=limit)

    fetched = await asyncio.gather(*tasks.values())
    return {
        name: _frame_to_records(frame)
        for name, frame in zip(tasks.keys(), fetched, strict=True)
    }


def _fetch_yfinance_news(ticker: str, limit: int) -> list[dict[str, Any]]:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Install yfinance with `pip install -e .[market-data-us]`.") from exc

    raw_items = yf.Ticker(ticker).news or []
    return [_normalize_news_item(ticker, item) for item in raw_items[:limit]]


def _fetch_akshare_news(ticker: str, limit: int) -> list[dict[str, Any]]:
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError("Install akshare with `pip install -e .[market-data-cn]`.") from exc

    symbol = _to_akshare_symbol(ticker)
    for fn_name, kwargs in [
        ("stock_news_em", {"symbol": symbol}),
        ("stock_news_main_cx", {"symbol": symbol}),
    ]:
        fn = getattr(ak, fn_name, None)
        if fn is None:
            continue
        try:
            frame = fn(**kwargs)
        except Exception:
            continue
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            return [_normalize_akshare_news_item(ticker, row.to_dict()) for _, row in frame.head(limit).iterrows()]
    return []


def _normalize_news_item(ticker: str, item: dict[str, Any]) -> dict[str, Any]:
    content = item.get("content") if isinstance(item.get("content"), dict) else {}
    return {
        "ticker": ticker.upper(),
        "title": item.get("title") or content.get("title"),
        "publisher": item.get("publisher") or content.get("provider", {}).get("displayName"),
        "link": item.get("link") or content.get("canonicalUrl", {}).get("url"),
        "published_at": item.get("providerPublishTime") or content.get("pubDate"),
        "summary": item.get("summary") or content.get("summary"),
        "type": item.get("type") or content.get("contentType"),
    }


def _normalize_akshare_news_item(ticker: str, item: dict[str, Any]) -> dict[str, Any]:
    title = item.get("新闻标题") or item.get("标题") or item.get("title")
    summary = item.get("新闻内容") or item.get("摘要") or item.get("summary")
    link = item.get("新闻链接") or item.get("链接") or item.get("url")
    published_at = item.get("发布时间") or item.get("时间") or item.get("date")
    publisher = item.get("文章来源") or item.get("来源") or item.get("publisher")
    return {
        "ticker": ticker.upper(),
        "title": title,
        "publisher": publisher,
        "link": link,
        "published_at": published_at,
        "summary": summary,
        "type": "cn_stock_news",
    }


def _frame_to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    normalized = frame.where(pd.notna(frame), None)
    return normalized.to_dict(orient="records")


def _to_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def _validate_limit(limit: int) -> None:
    if limit <= 0:
        raise ValueError("limit must be positive")
    if limit > 20:
        raise ValueError("limit must be 20 or less to keep tool responses concise")


def _to_akshare_symbol(symbol: str) -> str:
    cleaned = symbol.strip().upper()
    if "." in cleaned:
        return cleaned.split(".", maxsplit=1)[0]
    if cleaned.startswith(("SH", "SZ", "BJ")):
        return cleaned[2:]
    return cleaned
