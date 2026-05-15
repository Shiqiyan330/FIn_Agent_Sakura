"""LangGraph state machine for the multi-agent investment workflow."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, NotRequired, TypedDict


MarketName = Literal["us", "cn"]


class AgentState(TypedDict):
    """Shared state passed through the investment analysis graph."""

    ticker: str
    market: MarketName
    use_llm: bool
    context: Annotated[list[str], operator.add]
    warnings: Annotated[list[str], operator.add]
    node_events: Annotated[list[dict[str, Any]], operator.add]
    financial_json: dict[str, Any] | None
    fundamental_analysis: dict[str, Any] | None
    sentiment_analysis: dict[str, Any] | None
    technical_analysis: dict[str, Any] | None
    portfolio_decision: dict[str, Any] | None
    risk_score: float | None
    fundamental_view_vector: list[float]
    technical_signals: dict[str, float | str | bool]
    final_decision: dict[str, Any] | None
    sentiment_score: NotRequired[float | None]


NodeCallable = Any
NodeFactory = Any


def create_initial_state(
    ticker: str,
    *,
    market: MarketName = "cn",
    use_llm: bool = True,
) -> AgentState:
    """Create the minimal valid state for a new ticker analysis run."""

    return {
        "ticker": ticker.upper().strip(),
        "market": market,
        "use_llm": use_llm,
        "context": [],
        "warnings": [],
        "node_events": [],
        "financial_json": None,
        "fundamental_analysis": None,
        "sentiment_analysis": None,
        "technical_analysis": None,
        "portfolio_decision": None,
        "risk_score": None,
        "fundamental_view_vector": [],
        "technical_signals": {},
        "final_decision": None,
        "sentiment_score": None,
    }


def fundamental_analyst_node(state: AgentState) -> dict[str, Any]:
    """Fallback node for valuation and financial-statement analysis."""

    ticker = state["ticker"]
    analysis = {
        "ticker": ticker,
        "conclusion": "insufficient_data",
        "confidence": 0.0,
        "expected_excess_return": 0.0,
        "summary": "Fundamental analyst has not been configured with live tools.",
    }
    return {
        "context": [f"Fundamental analyst fallback completed for {ticker}."],
        "fundamental_analysis": analysis,
        "fundamental_view_vector": [0.0],
    }


def sentiment_analyst_node(state: AgentState) -> dict[str, Any]:
    """Fallback node for news, transcript, and market sentiment analysis."""

    ticker = state["ticker"]
    analysis = {
        "ticker": ticker,
        "sentiment_score": 0.0,
        "confidence": 0.0,
        "key_events": [],
        "summary": "Sentiment analyst has not been configured with live tools.",
    }
    return {
        "context": [f"Sentiment analyst fallback completed for {ticker}."],
        "sentiment_analysis": analysis,
        "sentiment_score": 0.0,
    }


def technical_analyst_node(state: AgentState) -> dict[str, Any]:
    """Fallback node for technical indicators and price-action signals."""

    ticker = state["ticker"]
    analysis = {
        "ticker": ticker,
        "trend": "unknown",
        "execution_signal": "hold",
        "summary": "Technical analyst has not been configured with live tools.",
    }
    return {
        "context": [f"Technical analyst fallback completed for {ticker}."],
        "technical_analysis": analysis,
        "technical_signals": {"status": "unknown"},
    }


def portfolio_manager_node(state: AgentState) -> dict[str, Any]:
    """Fallback node for final portfolio decision synthesis."""

    ticker = state["ticker"]
    decision = {
        "ticker": ticker,
        "action": "hold",
        "expected_excess_return": 0.0,
        "confidence": 0.0,
        "risk_score": state.get("risk_score"),
        "rationale": "Portfolio manager fallback awaiting real analyst signals.",
    }
    return {
        "context": [f"Portfolio manager fallback completed for {ticker}."],
        "portfolio_decision": decision,
        "final_decision": decision,
    }


def build_investment_workflow(
    *,
    compile_graph: bool = True,
    fundamental_node: NodeCallable | None = None,
    sentiment_node: NodeCallable | None = None,
    technical_node: NodeCallable | None = None,
    portfolio_node: NodeCallable | None = None,
) -> Any:
    """Build the directed LangGraph workflow for multi-agent analysis."""

    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise RuntimeError("Install LangGraph with `pip install -e .[agents]`.") from exc

    graph = StateGraph(AgentState)
    graph.add_node("fundamental_analyst", fundamental_node or fundamental_analyst_node)
    graph.add_node("sentiment_analyst", sentiment_node or sentiment_analyst_node)
    graph.add_node("technical_analyst", technical_node or technical_analyst_node)
    graph.add_node("portfolio_manager", portfolio_node or portfolio_manager_node)

    graph.add_edge(START, "fundamental_analyst")
    graph.add_edge("fundamental_analyst", "sentiment_analyst")
    graph.add_edge("sentiment_analyst", "technical_analyst")
    graph.add_edge("technical_analyst", "portfolio_manager")
    graph.add_edge("portfolio_manager", END)

    if compile_graph:
        return graph.compile()
    return graph
