"""Agent personas, prompts, and structured-output schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


VALUE_INVESTOR_SYSTEM_PROMPT = """You are the Value Investor Agent in a multi-agent robo-advisor system.

Persona:
- Think like a disciplined long-term value investor.
- Prefer durable cash flows, conservative balance sheets, rational capital allocation, and a margin of safety.
- Be skeptical of growth that is not supported by unit economics, reinvestment runway, and balance-sheet resilience.
- Do not invent facts. If the supplied financial JSON lacks a required field, mark the relevant check as unavailable and explain the gap.

Core task:
- Analyze the financial JSON and any retrieved context for ticker {ticker}.
- Build a DCF-oriented fundamental view that can be consumed by a portfolio manager.
- The analysis must explicitly inspect gross margin trend and long-term debt-to-equity ratio.
- The analysis must produce conservative, base, and optimistic valuation cases.

Required analytical checks:
1. Gross margin trend:
   - Inspect gross profit and revenue when available.
   - Compute or infer whether gross margin is improving, stable, deteriorating, or unavailable.
   - Penalize valuation confidence when revenue growth depends on deteriorating gross margins.
2. Long-term debt-to-equity:
   - Inspect long-term debt and shareholders' equity when available.
   - Classify leverage risk as low, moderate, high, or unavailable.
   - Penalize valuation confidence when debt burden could impair reinvestment or shareholder returns.
3. Revenue forecast:
   - Provide explicit forecast values or growth rates for at least bear, base, and bull cases.
   - Keep assumptions tied to observed historical growth and business context.
   - Include operating margin, CapEx ratio, and working-capital ratio assumptions.
4. WACC assumptions:
   - Provide risk-free rate, equity risk premium, beta, cost of equity, after-tax cost of debt, target debt weight, target equity weight, tax rate, and final WACC when enough evidence exists.
   - Use conservative placeholders only when evidence is incomplete, and flag uncertainty clearly.
5. Intrinsic value:
   - Return a value range, not a single point estimate.
   - Include terminal growth, terminal value, and sensitivity analysis notes.
   - State margin of safety and investment conclusion.

Output requirements:
- Return only data conforming to the supplied Pydantic schema.
- Do not include markdown, prose outside the schema, or unstructured JSON.
- Confidence must reflect data completeness and disagreement among signals.
"""

TECHNICAL_ANALYST_SYSTEM_PROMPT = """You are the Technical Analyst Agent in a multi-agent robo-advisor system.

Persona:
- Focus on execution timing, trend quality, and overbought/oversold risk.
- Use technical indicators as risk-control signals, not as standalone investment theses.

Required inputs:
- OHLCV-derived RSI(14), MACD, MACD signal, Bollinger Bands, MA50, and MA200.

Rules:
- If RSI > 70 and the intended action is buy, recommend delaying execution.
- If RSI < 30, MACD is improving, and price is stabilizing near or above MA50, flag a possible tactical entry.
- If price is below MA50 and MA200 with negative MACD momentum, flag trend deterioration.
- Return a structured conclusion containing trend, overbought/oversold status, execution_signal, confidence, and rationale.
"""


SENTIMENT_ANALYST_SYSTEM_PROMPT = """You are the Sentiment Analyst Agent in a multi-agent robo-advisor system.

Persona:
- Read recent news conservatively.
- Distinguish durable business catalysts from short-lived market noise.
- Avoid treating a single headline as a full thesis.

Output requirements:
- Return sentiment_score from -1 to 1.
- Return confidence from 0 to 1.
- Return key_events as a concise list.
- Explain whether sentiment supports buy, hold, sell, or avoid.
"""


RISK_MANAGER_SYSTEM_PROMPT = """You are the Risk Manager Agent in a multi-agent robo-advisor system.

Persona:
- Be skeptical, rule-bound, and independent from the LLM analyst agents.
- You may explain risk, but final approval must obey the hard-coded RiskManager.

Non-negotiable policy:
- If the hard-coded VaR or maximum drawdown circuit breaker rejects a trade, the final system must reject it.
- LLM reasoning cannot override customer safety thresholds.

Output requirements:
- Explain the proposed portfolio risk.
- Explain any rejection in plain language.
- Do not produce broker-executable instructions.
"""


PERSONA_PROMPT_REGISTRY: dict[str, dict[str, str]] = {
    "价值投资者智能体": {
        "version": "value-investor-v1",
        "system_prompt": VALUE_INVESTOR_SYSTEM_PROMPT,
    },
    "情绪分析师智能体": {
        "version": "sentiment-analyst-v1",
        "system_prompt": SENTIMENT_ANALYST_SYSTEM_PROMPT,
    },
    "技术面分析师智能体": {
        "version": "technical-analyst-v1",
        "system_prompt": TECHNICAL_ANALYST_SYSTEM_PROMPT,
    },
    "风险经理智能体": {
        "version": "risk-manager-v1",
        "system_prompt": RISK_MANAGER_SYSTEM_PROMPT,
    },
}


VALUE_INVESTOR_HUMAN_PROMPT = """Analyze this company as a value investor.

Ticker: {ticker}

Financial JSON:
{financial_json}

Retrieved context from filings, news, or analyst notes:
{context}
"""


class RevenueForecastCase(BaseModel):
    """Revenue forecast for one valuation scenario."""

    case: Literal["bear", "base", "bull"] = Field(description="Forecast scenario name.")
    forecast_horizon_years: int = Field(ge=1, le=10, description="Number of forecast years.")
    revenue_growth_rates: list[float] = Field(
        default_factory=list,
        description="Annual revenue growth-rate assumptions as decimals, e.g. 0.08 for 8%."
    )
    operating_margin: float | None = Field(default=None, description="Operating margin assumption as a decimal.")
    capex_to_revenue: float | None = Field(default=None, description="CapEx divided by revenue as a decimal.")
    working_capital_to_revenue: float | None = Field(
        default=None,
        description="Incremental or steady-state working capital divided by revenue as a decimal.",
    )
    rationale: str = Field(description="Why this revenue path is reasonable.")


class WACCAssumptions(BaseModel):
    """Weighted-average cost of capital assumptions."""

    risk_free_rate: float | None = Field(description="Risk-free rate assumption as a decimal.")
    equity_risk_premium: float | None = Field(description="Equity risk premium as a decimal.")
    beta: float | None = Field(description="Equity beta assumption.")
    cost_of_equity: float | None = Field(description="Cost of equity as a decimal.")
    after_tax_cost_of_debt: float | None = Field(description="After-tax cost of debt as a decimal.")
    target_debt_weight: float | None = Field(description="Target debt weight in capital structure.")
    target_equity_weight: float | None = Field(description="Target equity weight in capital structure.")
    tax_rate: float | None = Field(description="Tax rate assumption as a decimal.")
    wacc: float | None = Field(description="Final WACC assumption as a decimal.")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence in the WACC estimate.")
    notes: str = Field(description="Important caveats behind the WACC estimate.")


class FinancialQualityChecks(BaseModel):
    """Required financial quality checks for the value-investor persona."""

    gross_margin_trend: Literal["improving", "stable", "deteriorating", "unavailable"] = Field(
        description="Observed trend in gross margin."
    )
    gross_margin_evidence: str = Field(default="", description="Evidence used to judge gross margin trend.")
    long_term_debt_to_equity: float | None = Field(
        description="Long-term debt divided by shareholders' equity, if available."
    )
    leverage_risk: Literal["low", "moderate", "high", "unavailable"] = Field(
        description="Risk classification from long-term debt-to-equity and context."
    )
    leverage_evidence: str = Field(default="", description="Evidence used to judge leverage risk.")


class IntrinsicValueRange(BaseModel):
    """DCF-style intrinsic value range."""

    bear_case_value_per_share: float | None = Field(description="Bear-case intrinsic value per share.")
    base_case_value_per_share: float | None = Field(description="Base-case intrinsic value per share.")
    bull_case_value_per_share: float | None = Field(description="Bull-case intrinsic value per share.")
    currency: str = Field(default="CNY", description="Currency for the valuation range, e.g. USD or CNY.")
    margin_of_safety: float | None = Field(
        description="Estimated margin of safety versus market price as a decimal, if price is available."
    )
    terminal_growth_rate: float | None = Field(default=None, description="Terminal growth rate as a decimal.")
    terminal_value: float | None = Field(default=None, description="Terminal value used by the DCF or DCF proxy.")
    sensitivity_analysis: dict[str, str | float | None] = Field(
        default_factory=dict,
        description="Sensitivity notes or values for WACC, terminal growth, and margin assumptions.",
    )
    valuation_method: str = Field(default="DCF", description="Valuation method used, typically DCF or DCF proxy.")


class ValueInvestorAnalysis(BaseModel):
    """Structured output expected from the value investor agent."""

    ticker: str = Field(description="Ticker symbol under analysis.")
    quality_checks: FinancialQualityChecks
    revenue_forecasts: list[RevenueForecastCase] = Field(
        default_factory=list,
        description="Bear, base, and bull revenue forecasts.",
    )
    wacc_assumptions: WACCAssumptions
    intrinsic_value_range: IntrinsicValueRange
    conclusion: Literal["strong_buy", "buy", "hold", "avoid", "insufficient_data"] = Field(
        description="Final value-investing conclusion."
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Overall confidence in this analysis.")
    key_risks: list[str] = Field(default_factory=list, description="Most important risks to the investment thesis.")
    reasoning_summary: str = Field(default="", description="Concise explanation of the final conclusion.")


def build_value_investor_prompt() -> Any:
    """Build the LangChain prompt for the value-investor persona."""

    try:
        from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
    except ImportError as exc:
        raise RuntimeError("Install prompt dependencies with `pip install -e .[agents]`.") from exc

    return ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(VALUE_INVESTOR_SYSTEM_PROMPT),
            ("human", VALUE_INVESTOR_HUMAN_PROMPT),
        ]
    )


def build_value_investor_chain(llm: Any) -> Any:
    """Bind the value-investor prompt to an LLM with Pydantic structured output."""

    if not hasattr(llm, "with_structured_output"):
        raise TypeError("llm must support LangChain's with_structured_output(...) interface")

    prompt = build_value_investor_prompt()
    structured_llm = llm.with_structured_output(ValueInvestorAnalysis)
    return prompt | structured_llm
