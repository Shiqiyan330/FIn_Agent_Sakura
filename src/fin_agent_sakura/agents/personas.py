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
4. WACC assumptions:
   - Provide risk-free rate, equity risk premium, beta, cost of equity, after-tax cost of debt, target debt weight, target equity weight, tax rate, and final WACC when enough evidence exists.
   - Use conservative placeholders only when evidence is incomplete, and flag uncertainty clearly.
5. Intrinsic value:
   - Return a value range, not a single point estimate.
   - State margin of safety and investment conclusion.

Output requirements:
- Return only data conforming to the supplied Pydantic schema.
- Do not include markdown, prose outside the schema, or unstructured JSON.
- Confidence must reflect data completeness and disagreement among signals.
"""


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
        description="Annual revenue growth-rate assumptions as decimals, e.g. 0.08 for 8%."
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
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the WACC estimate.")
    notes: str = Field(description="Important caveats behind the WACC estimate.")


class FinancialQualityChecks(BaseModel):
    """Required financial quality checks for the value-investor persona."""

    gross_margin_trend: Literal["improving", "stable", "deteriorating", "unavailable"] = Field(
        description="Observed trend in gross margin."
    )
    gross_margin_evidence: str = Field(description="Evidence used to judge gross margin trend.")
    long_term_debt_to_equity: float | None = Field(
        description="Long-term debt divided by shareholders' equity, if available."
    )
    leverage_risk: Literal["low", "moderate", "high", "unavailable"] = Field(
        description="Risk classification from long-term debt-to-equity and context."
    )
    leverage_evidence: str = Field(description="Evidence used to judge leverage risk.")


class IntrinsicValueRange(BaseModel):
    """DCF-style intrinsic value range."""

    bear_case_value_per_share: float | None = Field(description="Bear-case intrinsic value per share.")
    base_case_value_per_share: float | None = Field(description="Base-case intrinsic value per share.")
    bull_case_value_per_share: float | None = Field(description="Bull-case intrinsic value per share.")
    currency: str = Field(description="Currency for the valuation range, e.g. USD or CNY.")
    margin_of_safety: float | None = Field(
        description="Estimated margin of safety versus market price as a decimal, if price is available."
    )
    valuation_method: str = Field(description="Valuation method used, typically DCF or DCF proxy.")


class ValueInvestorAnalysis(BaseModel):
    """Structured output expected from the value investor agent."""

    ticker: str = Field(description="Ticker symbol under analysis.")
    quality_checks: FinancialQualityChecks
    revenue_forecasts: list[RevenueForecastCase] = Field(
        min_length=3,
        description="Bear, base, and bull revenue forecasts.",
    )
    wacc_assumptions: WACCAssumptions
    intrinsic_value_range: IntrinsicValueRange
    conclusion: Literal["strong_buy", "buy", "hold", "avoid", "insufficient_data"] = Field(
        description="Final value-investing conclusion."
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Overall confidence in this analysis.")
    key_risks: list[str] = Field(description="Most important risks to the investment thesis.")
    reasoning_summary: str = Field(description="Concise explanation of the final conclusion.")


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

