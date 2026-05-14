"""LangGraph state machine for the multi-agent investment workflow."""

from __future__ import annotations

import operator
from typing import Annotated, Any, NotRequired, TypedDict


class AgentState(TypedDict):
    """Shared state passed through the investment analysis graph."""

    ticker: str
    context: Annotated[list[str], operator.add]
    risk_score: float | None
    fundamental_view_vector: list[float]
    technical_signals: dict[str, float | str | bool]
    final_decision: dict[str, Any] | None
    sentiment_score: NotRequired[float | None]


def create_initial_state(ticker: str) -> AgentState:
    """Create the minimal valid state for a new ticker analysis run."""

    return {
        "ticker": ticker.upper().strip(),
        "context": [],
        "risk_score": None,
        "fundamental_view_vector": [],
        "technical_signals": {},
        "final_decision": None,
        "sentiment_score": None,
    }


def fundamental_analyst_node(state: AgentState) -> dict[str, Any]:
    """Placeholder node for valuation and financial-statement analysis."""

    ticker = state["ticker"]
    return {
        "context": [f"Fundamental analyst placeholder completed for {ticker}."],
        "fundamental_view_vector": state.get("fundamental_view_vector", []),
    }


def sentiment_analyst_node(state: AgentState) -> dict[str, Any]:
    """Placeholder node for news, transcript, and market sentiment analysis."""

    ticker = state["ticker"]
    return {
        "context": [f"Sentiment analyst placeholder completed for {ticker}."],
        "sentiment_score": state.get("sentiment_score"),
    }


def technical_analyst_node(state: AgentState) -> dict[str, Any]:
    """Placeholder node for technical indicators and price-action signals."""

    ticker = state["ticker"]
    current_signals = dict(state.get("technical_signals", {}))
    current_signals.setdefault("status", "pending")
    return {
        "context": [f"Technical analyst placeholder completed for {ticker}."],
        "technical_signals": current_signals,
    }


def portfolio_manager_node(state: AgentState) -> dict[str, Any]:
    """Placeholder node for final portfolio decision synthesis."""

    ticker = state["ticker"]
    risk_score = state.get("risk_score")
    decision = {
        "ticker": ticker,
        "action": "hold",
        "confidence": 0.0,
        "rationale": "Portfolio manager placeholder awaiting real analyst signals.",
        "risk_score": risk_score,
    }
    return {
        "context": [f"Portfolio manager placeholder completed for {ticker}."],
        "final_decision": decision,
    }


def build_investment_workflow(*, compile_graph: bool = True) -> Any:
    """Build the directed LangGraph workflow for multi-agent analysis.

    The graph runs analyst nodes sequentially for now:
    fundamentals -> sentiment -> technicals -> portfolio manager.
    The node boundaries are intentionally explicit so each role can later be
    replaced by a richer tool-using agent without changing the graph contract.
    """

    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise RuntimeError("Install LangGraph with `pip install -e .[agents]`.") from exc

    graph = StateGraph(AgentState)
    graph.add_node("fundamental_analyst", fundamental_analyst_node)
    graph.add_node("sentiment_analyst", sentiment_analyst_node)
    graph.add_node("technical_analyst", technical_analyst_node)
    graph.add_node("portfolio_manager", portfolio_manager_node)

    graph.add_edge(START, "fundamental_analyst")
    graph.add_edge("fundamental_analyst", "sentiment_analyst")
    graph.add_edge("sentiment_analyst", "technical_analyst")
    graph.add_edge("technical_analyst", "portfolio_manager")
    graph.add_edge("portfolio_manager", END)

    if compile_graph:
        return graph.compile()
    return graph

