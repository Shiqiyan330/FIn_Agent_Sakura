"""Agent workflow definitions for the Sakura robo-advisor system.

Persona schemas are lightweight and imported eagerly. LangGraph workflow
objects are imported lazily so dashboard startup is not blocked by optional
workflow dependencies in constrained cloud deployments.
"""

from __future__ import annotations

from typing import Any

from fin_agent_sakura.agents.personas import (
    FinancialQualityChecks,
    IntrinsicValueRange,
    PERSONA_PROMPT_REGISTRY,
    RISK_MANAGER_SYSTEM_PROMPT,
    RevenueForecastCase,
    SENTIMENT_ANALYST_SYSTEM_PROMPT,
    TECHNICAL_ANALYST_SYSTEM_PROMPT,
    VALUE_INVESTOR_SYSTEM_PROMPT,
    ValueInvestorAnalysis,
    WACCAssumptions,
    build_value_investor_chain,
    build_value_investor_prompt,
)

_WORKFLOW_EXPORTS = {
    "AgentState",
    "build_investment_workflow",
    "create_initial_state",
    "fundamental_analyst_node",
    "portfolio_manager_node",
    "sentiment_analyst_node",
    "technical_analyst_node",
}


def __getattr__(name: str) -> Any:
    if name in _WORKFLOW_EXPORTS:
        from fin_agent_sakura.agents import workflow

        return getattr(workflow, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AgentState",
    "FinancialQualityChecks",
    "IntrinsicValueRange",
    "PERSONA_PROMPT_REGISTRY",
    "RISK_MANAGER_SYSTEM_PROMPT",
    "RevenueForecastCase",
    "SENTIMENT_ANALYST_SYSTEM_PROMPT",
    "TECHNICAL_ANALYST_SYSTEM_PROMPT",
    "VALUE_INVESTOR_SYSTEM_PROMPT",
    "ValueInvestorAnalysis",
    "WACCAssumptions",
    "build_investment_workflow",
    "build_value_investor_chain",
    "build_value_investor_prompt",
    "create_initial_state",
    "fundamental_analyst_node",
    "portfolio_manager_node",
    "sentiment_analyst_node",
    "technical_analyst_node",
]
