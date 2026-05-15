"""Single-ticker multi-agent analysis service."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from fin_agent_sakura.config import get_llm_config
from fin_agent_sakura.data import MarketDataClientFactory, TechnicalIndicators
from fin_agent_sakura.rag.financial_context import FinancialRAGError, retrieve_financial_context


MarketName = Literal["us", "cn"]
NodeStatus = Literal["success", "warning", "error"]
DecisionAction = Literal["buy", "hold", "sell", "avoid"]


@dataclass(frozen=True, slots=True)
class AgentNodeEvent:
    """Execution record for one analyst node."""

    node: str
    status: NodeStatus
    elapsed_seconds: float
    summary: str
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AgentAnalysisResult:
    """Persistable result for one multi-agent ticker analysis run."""

    ticker: str
    market: MarketName
    use_llm: bool
    generated_at: str
    final_decision: dict[str, Any]
    node_events: list[AgentNodeEvent]
    context: list[str]
    warnings: list[str]
    raw_state: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "market": self.market,
            "use_llm": self.use_llm,
            "generated_at": self.generated_at,
            "final_decision": self.final_decision,
            "node_events": [event.to_dict() for event in self.node_events],
            "context": self.context,
            "warnings": self.warnings,
            "raw_state": self.raw_state,
        }


def run_single_ticker_agent_analysis(
    ticker: str,
    market: MarketName = "cn",
    use_llm: bool = True,
    *,
    output_dir: str | Path = "data/processed",
) -> AgentAnalysisResult:
    """Run the four-agent workflow for a single ticker and persist the result."""

    runner = SingleTickerAgentAnalysisRunner(output_dir=output_dir)
    return runner.run(ticker=ticker, market=market, use_llm=use_llm)


class SingleTickerAgentAnalysisRunner:
    """Build and run tool-backed analyst nodes for one ticker."""

    def __init__(self, output_dir: str | Path = "data/processed") -> None:
        self.output_dir = Path(output_dir)

    def run(self, *, ticker: str, market: MarketName = "cn", use_llm: bool = True) -> AgentAnalysisResult:
        from fin_agent_sakura.agents import build_investment_workflow, create_initial_state

        graph = build_investment_workflow(
            fundamental_node=self.fundamental_node,
            sentiment_node=self.sentiment_node,
            technical_node=self.technical_node,
            portfolio_node=self.portfolio_node,
        )
        initial_state = create_initial_state(ticker, market=market, use_llm=use_llm)
        final_state = graph.invoke(initial_state)
        result = self._result_from_state(final_state)
        self.save_result(result)
        return result

    def save_result(self, result: AgentAnalysisResult) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "agent_analysis_latest.json"
        path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return path

    def load_latest_result(self) -> AgentAnalysisResult | None:
        path = self.output_dir / "agent_analysis_latest.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return AgentAnalysisResult(
            ticker=payload["ticker"],
            market=payload["market"],
            use_llm=payload["use_llm"],
            generated_at=payload["generated_at"],
            final_decision=payload["final_decision"],
            node_events=[AgentNodeEvent(**event) for event in payload.get("node_events", [])],
            context=payload.get("context", []),
            warnings=payload.get("warnings", []),
            raw_state=payload.get("raw_state", {}),
        )

    def fundamental_node(self, state: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        ticker = state["ticker"]
        market = state.get("market", "cn")
        warnings: list[str] = []

        try:
            financial_json = asyncio.run(_fetch_financial_json(ticker, market))
            rag_context: list[str] = []
            try:
                rag_context = retrieve_financial_context(
                    "毛利率趋势 长期债务 股东权益 现金流 经营风险 管理层讨论",
                    ticker,
                    top_k=3,
                )
            except FinancialRAGError as exc:
                warnings.append(f"未读取到本地财报RAG上下文，将仅使用三张表：{exc}")
            except Exception as exc:
                warnings.append(f"财报RAG检索失败，将仅使用三张表：{type(exc).__name__}: {exc}")

            analysis = _deterministic_fundamental_analysis(ticker, financial_json, rag_context)
            if state.get("use_llm"):
                try:
                    analysis = _llm_value_analysis(ticker, financial_json, [*state.get("context", []), *rag_context])
                except Exception as exc:
                    warnings.append(f"价值投资者LLM结构化输出失败，已使用确定性基本面摘要：{type(exc).__name__}: {exc}")

            event = _event(
                "基本面分析师",
                "success" if not warnings else "warning",
                started,
                analysis.get("summary", "基本面分析完成。"),
                analysis,
            )
            return {
                "financial_json": financial_json,
                "fundamental_analysis": analysis,
                "fundamental_view_vector": [float(analysis.get("expected_excess_return", 0.0))],
                "context": [event.summary, *[f"RAG: {item[:800]}" for item in rag_context]],
                "warnings": warnings,
                "node_events": [event.to_dict()],
            }
        except Exception as exc:
            analysis = _fallback_fundamental_analysis(ticker, exc)
            event = _event("基本面分析师", "error", started, analysis["summary"], analysis, exc)
            return {
                "fundamental_analysis": analysis,
                "fundamental_view_vector": [0.0],
                "context": [event.summary],
                "warnings": [event.error or event.summary],
                "node_events": [event.to_dict()],
            }

    def sentiment_node(self, state: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        ticker = state["ticker"]
        market = state.get("market", "cn")
        warnings: list[str] = []

        news: list[dict[str, Any]] = []
        try:
            news = _fetch_recent_news(ticker, market)
        except Exception as exc:
            warnings.append(f"新闻获取失败，使用中性情绪：{type(exc).__name__}: {exc}")

        analysis = _deterministic_sentiment_analysis(ticker, news)
        if state.get("use_llm") and news:
            try:
                analysis = _llm_sentiment_analysis(ticker, news)
            except Exception as exc:
                warnings.append(f"情绪LLM分析失败，已使用规则情绪：{type(exc).__name__}: {exc}")

        event = _event(
            "情绪分析师",
            "success" if not warnings else "warning",
            started,
            analysis.get("summary", "情绪分析完成。"),
            analysis,
        )
        return {
            "sentiment_analysis": analysis,
            "sentiment_score": float(analysis.get("sentiment_score", 0.0)),
            "context": [event.summary],
            "warnings": warnings,
            "node_events": [event.to_dict()],
        }

    def technical_node(self, state: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        ticker = state["ticker"]
        market = state.get("market", "cn")

        try:
            analysis = asyncio.run(_technical_analysis(ticker, market))
            event = _event("技术面分析师", "success", started, analysis["summary"], analysis)
            return {
                "technical_analysis": analysis,
                "technical_signals": analysis,
                "context": [event.summary],
                "node_events": [event.to_dict()],
            }
        except Exception as exc:
            analysis = _fallback_technical_analysis(ticker, exc)
            event = _event("技术面分析师", "error", started, analysis["summary"], analysis, exc)
            return {
                "technical_analysis": analysis,
                "technical_signals": analysis,
                "context": [event.summary],
                "warnings": [event.error or event.summary],
                "node_events": [event.to_dict()],
            }

    def portfolio_node(self, state: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        decision = _portfolio_decision(
            ticker=state["ticker"],
            fundamental=state.get("fundamental_analysis") or {},
            sentiment=state.get("sentiment_analysis") or {},
            technical=state.get("technical_analysis") or {},
        )
        event = _event("投资组合经理", "success", started, decision["rationale"], decision)
        return {
            "portfolio_decision": decision,
            "final_decision": decision,
            "risk_score": decision["risk_score"],
            "context": [event.summary],
            "node_events": [event.to_dict()],
        }

    def _result_from_state(self, state: dict[str, Any]) -> AgentAnalysisResult:
        node_events = [AgentNodeEvent(**event) for event in state.get("node_events", [])]
        return AgentAnalysisResult(
            ticker=state["ticker"],
            market=state.get("market", "cn"),
            use_llm=bool(state.get("use_llm", True)),
            generated_at=pd.Timestamp.now().isoformat(),
            final_decision=state.get("final_decision") or {},
            node_events=node_events,
            context=state.get("context", []),
            warnings=state.get("warnings", []),
            raw_state=_json_safe(state),
        )


async def _fetch_financial_json(ticker: str, market: MarketName) -> dict[str, Any]:
    client = MarketDataClientFactory.get_client(market)
    balance_sheet, cash_flow, income_statement = await asyncio.gather(
        client.fetch_balance_sheet(ticker, period="annual", limit=5),
        client.fetch_cash_flow(ticker, period="annual", limit=5),
        client.fetch_income_statement(ticker, period="annual", limit=5),
    )
    return {
        "balance_sheet": _records(balance_sheet),
        "cash_flow": _records(cash_flow),
        "income_statement": _records(income_statement),
    }


async def _technical_analysis(ticker: str, market: MarketName) -> dict[str, Any]:
    client = MarketDataClientFactory.get_client(market)
    start = date.today() - timedelta(days=365)
    frame = await client.fetch_ohlcv(ticker, start=start, interval="1d", adjusted=True)
    indicators = TechnicalIndicators(engine="pandas").calculate(frame)
    latest = indicators.iloc[-1]
    close = _number(latest.get("close"))
    ma_50 = _number(latest.get("ma_50"), close)
    ma_200 = _number(latest.get("ma_200"), close)
    rsi = _number(latest.get("rsi_14"), 50.0)
    macd = _number(latest.get("macd"), 0.0)
    macd_signal = _number(latest.get("macd_signal"), 0.0)

    trend = "uptrend" if close >= ma_50 >= ma_200 else "downtrend" if close < ma_50 < ma_200 else "mixed"
    overbought = rsi > 70
    oversold = rsi < 30
    bullish_momentum = macd > macd_signal and close > ma_50
    execution_signal = "buy" if bullish_momentum and not overbought else "sell" if trend == "downtrend" else "hold"

    return {
        "ticker": ticker,
        "close": close,
        "rsi_14": rsi,
        "macd": macd,
        "macd_signal": macd_signal,
        "ma_50": ma_50,
        "ma_200": ma_200,
        "trend": trend,
        "overbought": overbought,
        "oversold": oversold,
        "bullish_momentum": bullish_momentum,
        "execution_signal": execution_signal,
        "summary": f"技术面为{trend}，RSI {rsi:.1f}，执行建议 {execution_signal}。",
    }


def _deterministic_fundamental_analysis(
    ticker: str,
    financial_json: dict[str, Any],
    rag_context: list[str] | None = None,
) -> dict[str, Any]:
    income = financial_json.get("income_statement") or []
    balance = financial_json.get("balance_sheet") or []
    rag_context = rag_context or []
    gross_margin_trend = _gross_margin_trend(income)
    debt_to_equity = _debt_to_equity(balance)
    leverage_risk = "unavailable"
    if debt_to_equity is not None:
        leverage_risk = "low" if debt_to_equity < 0.5 else "moderate" if debt_to_equity < 1.5 else "high"

    confidence = 0.45
    expected_return = 0.02
    if gross_margin_trend == "improving":
        confidence += 0.1
        expected_return += 0.015
    if leverage_risk == "low":
        confidence += 0.1
        expected_return += 0.01
    if gross_margin_trend == "deteriorating" or leverage_risk == "high":
        confidence -= 0.15
        expected_return -= 0.02

    conclusion = "buy" if expected_return >= 0.04 else "hold" if expected_return >= 0 else "avoid"
    return {
        "ticker": ticker,
        "conclusion": conclusion,
        "confidence": max(0.0, min(1.0, confidence)),
        "expected_excess_return": expected_return,
        "quality_checks": {
            "gross_margin_trend": gross_margin_trend,
            "long_term_debt_to_equity": debt_to_equity,
            "leverage_risk": leverage_risk,
        },
        "rag_context_count": len(rag_context),
        "rag_context_preview": [item[:1000] for item in rag_context[:3]],
        "summary": (
            f"基本面规则分析：毛利率趋势 {gross_margin_trend}，杠杆风险 {leverage_risk}，"
            f"RAG段落 {len(rag_context)} 条，结论 {conclusion}。"
        ),
    }


def _llm_value_analysis(ticker: str, financial_json: dict[str, Any], context: list[str]) -> dict[str, Any]:
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError("Install langchain-openai to use LLM value analysis") from exc
    from fin_agent_sakura.agents import build_value_investor_chain

    cfg = get_llm_config()
    if not cfg.api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    llm = ChatOpenAI(model=cfg.chat_model, api_key=cfg.api_key, base_url=cfg.base_url, temperature=0.0)
    chain = build_value_investor_chain(llm)
    result = chain.invoke(
        {
            "ticker": ticker,
            "financial_json": json.dumps(financial_json, ensure_ascii=False, default=str)[:20000],
            "context": "\n".join(context[-5:]) or "No retrieved context supplied.",
        }
    )
    payload = result.model_dump()
    conclusion = payload.get("conclusion", "hold")
    confidence = float(payload.get("confidence", 0.0))
    expected_return = {
        "strong_buy": 0.08,
        "buy": 0.05,
        "hold": 0.01,
        "avoid": -0.03,
        "insufficient_data": 0.0,
    }.get(conclusion, 0.0)
    payload["expected_excess_return"] = expected_return
    payload["summary"] = payload.get("reasoning_summary") or f"价值投资者结构化输出：{conclusion}"
    payload["confidence"] = confidence
    return payload


def _fetch_recent_news(ticker: str, market: MarketName, limit: int = 5) -> list[dict[str, Any]]:
    if market == "us":
        from fin_agent_sakura.tools.financial_tools import get_recent_news

        payload = json.loads(get_recent_news.invoke({"ticker": ticker, "market": market, "limit": limit}))
        return payload.get("news", [])

    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError("Install akshare to fetch A-share news") from exc

    news_frames = []
    for fn_name, kwargs in [
        ("stock_news_em", {"symbol": _to_akshare_symbol(ticker)}),
        ("stock_news_main_cx", {"symbol": _to_akshare_symbol(ticker)}),
    ]:
        fn = getattr(ak, fn_name, None)
        if fn is None:
            continue
        try:
            frame = fn(**kwargs)
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                news_frames.append(frame.head(limit))
                break
        except Exception:
            continue

    if not news_frames:
        return []

    frame = news_frames[0]
    items = []
    for _, row in frame.head(limit).iterrows():
        data = {str(key): value for key, value in row.to_dict().items()}
        title = data.get("新闻标题") or data.get("标题") or data.get("title") or ""
        summary = data.get("新闻内容") or data.get("摘要") or data.get("summary") or ""
        items.append(
            {
                "ticker": ticker,
                "title": str(title),
                "summary": str(summary),
                "published_at": str(data.get("发布时间") or data.get("时间") or ""),
                "publisher": str(data.get("文章来源") or data.get("来源") or ""),
                "link": str(data.get("新闻链接") or data.get("链接") or ""),
            }
        )
    return items


def _deterministic_sentiment_analysis(ticker: str, news: list[dict[str, Any]]) -> dict[str, Any]:
    positive_words = ("增长", "盈利", "创新高", "回购", "增持", "中标", "突破", "利好")
    negative_words = ("下滑", "亏损", "减持", "处罚", "风险", "调查", "暴跌", "利空")
    score = 0.0
    events = []
    for item in news[:5]:
        text = f"{item.get('title', '')} {item.get('summary', '')}"
        pos = sum(word in text for word in positive_words)
        neg = sum(word in text for word in negative_words)
        score += 0.18 * (pos - neg)
        if item.get("title"):
            events.append(str(item["title"]))
    score = max(-1.0, min(1.0, score))
    confidence = min(0.8, 0.25 + len(news) * 0.1)
    return {
        "ticker": ticker,
        "sentiment_score": score,
        "confidence": confidence,
        "key_events": events,
        "news_count": len(news),
        "summary": f"最近新闻 {len(news)} 条，规则情绪分数 {score:.2f}。",
    }


def _llm_sentiment_analysis(ticker: str, news: list[dict[str, Any]]) -> dict[str, Any]:
    from openai import OpenAI

    cfg = get_llm_config()
    if not cfg.api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
    prompt = (
        "你是A股市场情绪分析师。请只返回JSON，不要markdown。"
        "字段：sentiment_score(-1到1), confidence(0到1), key_events(list), summary。\n"
        f"股票：{ticker}\n新闻：{json.dumps(news[:5], ensure_ascii=False, default=str)}"
    )
    response = client.chat.completions.create(
        model=cfg.chat_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
    )
    content = response.choices[0].message.content or "{}"
    payload = _parse_json_object(content)
    payload["ticker"] = ticker
    payload["sentiment_score"] = float(payload.get("sentiment_score", 0.0))
    payload["confidence"] = float(payload.get("confidence", 0.5))
    payload.setdefault("key_events", [])
    payload.setdefault("summary", "情绪LLM分析完成。")
    return payload


def _portfolio_decision(
    *,
    ticker: str,
    fundamental: dict[str, Any],
    sentiment: dict[str, Any],
    technical: dict[str, Any],
) -> dict[str, Any]:
    fundamental_return = float(fundamental.get("expected_excess_return", 0.0))
    fundamental_confidence = float(fundamental.get("confidence", 0.0))
    sentiment_score = float(sentiment.get("sentiment_score", 0.0))
    sentiment_confidence = float(sentiment.get("confidence", 0.0))
    technical_signal = str(technical.get("execution_signal", "hold"))
    technical_score = 0.2 if technical_signal == "buy" else -0.2 if technical_signal == "sell" else 0.0
    overbought_penalty = 0.12 if technical.get("overbought") else 0.0

    combined_score = (
        fundamental_return
        + 0.035 * sentiment_score * max(sentiment_confidence, 0.25)
        + 0.04 * technical_score
        - overbought_penalty
    )
    risk_score = max(0.0, min(1.0, 0.45 - combined_score + (0.15 if technical.get("trend") == "downtrend" else 0.0)))
    confidence = max(0.0, min(1.0, 0.45 * fundamental_confidence + 0.25 * sentiment_confidence + 0.3))

    if combined_score >= 0.045 and risk_score < 0.65:
        action: DecisionAction = "buy"
    elif combined_score <= -0.035 or risk_score >= 0.78:
        action = "avoid"
    elif technical_signal == "sell":
        action = "sell"
    else:
        action = "hold"

    return {
        "ticker": ticker,
        "action": action,
        "expected_excess_return": combined_score,
        "confidence": confidence,
        "risk_score": risk_score,
        "rationale": (
            f"综合基本面超额收益 {fundamental_return:.2%}、情绪 {sentiment_score:.2f}、"
            f"技术信号 {technical_signal}，最终建议 {action}。"
        ),
    }


def _fallback_fundamental_analysis(ticker: str, exc: Exception) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "conclusion": "insufficient_data",
        "confidence": 0.0,
        "expected_excess_return": 0.0,
        "summary": f"基本面分析失败：{type(exc).__name__}: {exc}",
    }


def _fallback_technical_analysis(ticker: str, exc: Exception) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "trend": "unknown",
        "execution_signal": "hold",
        "summary": f"技术面分析失败：{type(exc).__name__}: {exc}",
    }


def _gross_margin_trend(income_rows: list[dict[str, Any]]) -> str:
    margins: list[float] = []
    for row in income_rows[:5]:
        revenue = _first_number(row, ("revenue", "total_revenue", "营业总收入", "营业收入"))
        gross_profit = _first_number(row, ("grossProfit", "gross_profit", "营业毛利", "毛利"))
        if revenue and gross_profit is not None and revenue > 0:
            margins.append(gross_profit / revenue)
    if len(margins) < 2:
        return "unavailable"
    recent = margins[0]
    older = margins[-1]
    if recent > older + 0.015:
        return "improving"
    if recent < older - 0.015:
        return "deteriorating"
    return "stable"


def _debt_to_equity(balance_rows: list[dict[str, Any]]) -> float | None:
    if not balance_rows:
        return None
    row = balance_rows[0]
    debt = _first_number(row, ("longTermDebt", "lt_borr", "non_cur_liab", "长期借款", "非流动负债合计"))
    equity = _first_number(row, ("totalStockholdersEquity", "total_hldr_eqy_exc_min_int", "股东权益合计"))
    if debt is None or equity is None or equity <= 0:
        return None
    return debt / equity


def _first_number(row: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    lowered = {str(key).lower(): value for key, value in row.items()}
    for key in keys:
        value = row.get(key)
        if value is None:
            value = lowered.get(key.lower())
        if value is not None:
            number = _number(value, None)
            if number is not None:
                return number
    return None


def _number(value: Any, default: float | None = 0.0) -> float | None:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return frame.where(pd.notna(frame), None).to_dict("records")


def _event(
    node: str,
    status: NodeStatus,
    started: float,
    summary: str,
    result: dict[str, Any],
    error: Exception | None = None,
) -> AgentNodeEvent:
    return AgentNodeEvent(
        node=node,
        status=status,
        elapsed_seconds=round(time.perf_counter() - started, 3),
        summary=summary,
        result=_json_safe(result),
        error=f"{type(error).__name__}: {error}" if error else None,
    )


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _parse_json_object(content: str) -> dict[str, Any]:
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("LLM response did not contain a JSON object")
    parsed = json.loads(content[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("LLM response JSON was not an object")
    return parsed


def _to_akshare_symbol(symbol: str) -> str:
    cleaned = symbol.strip().upper()
    if "." in cleaned:
        return cleaned.split(".", maxsplit=1)[0]
    if cleaned.startswith(("SH", "SZ", "BJ")):
        return cleaned[2:]
    return cleaned
