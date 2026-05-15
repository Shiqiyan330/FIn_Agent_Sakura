"""Agent workflow definitions for the Sakura robo-advisor system."""

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
from fin_agent_sakura.agents.workflow import (
    AgentState,
    build_investment_workflow,
    create_initial_state,
    fundamental_analyst_node,
    portfolio_manager_node,
    sentiment_analyst_node,
    technical_analyst_node,
)

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
