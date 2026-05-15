"""Portfolio weight optimization using Black-Litterman posterior estimates."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import pandas as pd


OptimizationObjective = Literal["max_sharpe", "min_volatility", "target_volatility", "target_return"]


@dataclass(frozen=True, slots=True)
class ClientPortfolioConstraints:
    """Client-specific constraints parsed from a questionnaire or chat profile."""

    max_volatility: float | None = None
    min_expected_return: float | None = None
    max_single_asset_weight: float = 1.0
    min_single_asset_weight: float = 0.0
    risk_free_rate: float = 0.02


@dataclass(frozen=True, slots=True)
class PortfolioOptimizationResult:
    """Optimized portfolio weights and expected performance."""

    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe_ratio: float
    objective: OptimizationObjective
    constraints: ClientPortfolioConstraints
    fallback_used: bool = False
    diagnostics: list[str] | None = None

    @property
    def weights_percent(self) -> pd.Series:
        """Asset weights expressed as percentages."""

        return self.weights * 100


def optimize_bl_portfolio_weights(
    posterior_returns: pd.Series,
    posterior_covariance: pd.DataFrame,
    *,
    constraints: ClientPortfolioConstraints | None = None,
    objective: OptimizationObjective | None = None,
    clean_cutoff: float = 1e-4,
) -> PortfolioOptimizationResult:
    """Solve final allocation weights from BL posterior return and covariance.

    Args:
        posterior_returns: Black-Litterman posterior expected returns indexed by ticker.
        posterior_covariance: Black-Litterman posterior covariance matrix whose
            rows and columns are ticker symbols.
        constraints: Client constraints, including maximum volatility and
            maximum single-asset weight.
        objective: Optional optimization objective. If omitted, the function
            chooses target volatility when max_volatility is supplied, target
            return when only min_expected_return is supplied, otherwise max
            Sharpe ratio.
        clean_cutoff: Weights with absolute value below this cutoff are rounded
            to zero by PyPortfolioOpt's clean_weights.

    Returns:
        PortfolioOptimizationResult with allocation weights as decimals and
        performance metrics.
    """

    client_constraints = constraints or ClientPortfolioConstraints()
    _validate_inputs(posterior_returns, posterior_covariance, client_constraints)
    aligned_returns, aligned_covariance = _align_posterior_inputs(
        posterior_returns,
        posterior_covariance,
    )
    client_constraints = _make_constraints_feasible(client_constraints, asset_count=len(aligned_returns))
    chosen_objective = objective or _choose_objective(client_constraints)

    ef = _build_efficient_frontier(aligned_returns, aligned_covariance, client_constraints)
    diagnostics = _constraint_diagnostics(aligned_returns, aligned_covariance, client_constraints)
    fallback_used = False
    try:
        _solve_frontier(ef, chosen_objective, client_constraints)
    except Exception as exc:
        fallback_used = True
        diagnostics.append(f"优化目标 {chosen_objective} 不可达，已回退到最小波动组合：{type(exc).__name__}: {exc}")
        ef = _build_efficient_frontier(aligned_returns, aligned_covariance, client_constraints)
        ef.min_volatility()

    weights = pd.Series(ef.clean_weights(cutoff=clean_cutoff), dtype="float64")
    weights = weights.reindex(aligned_returns.index).fillna(0.0)
    expected_return, volatility, sharpe_ratio = ef.portfolio_performance(
        risk_free_rate=client_constraints.risk_free_rate
    )

    return PortfolioOptimizationResult(
        weights=weights,
        expected_return=float(expected_return),
        volatility=float(volatility),
        sharpe_ratio=float(sharpe_ratio),
        objective=chosen_objective,
        constraints=client_constraints,
        fallback_used=fallback_used,
        diagnostics=diagnostics,
    )


def parse_client_constraints_from_text(profile_text: str) -> ClientPortfolioConstraints:
    """Map simple client profile text to optimization constraints.

    This lightweight parser covers common questionnaire/chat phrases and is
    designed as a deterministic fallback. A future LLM structured-output parser
    can return the same ClientPortfolioConstraints object.
    """

    text = profile_text.strip().lower()
    if not text:
        raise ValueError("profile_text must not be empty")

    if any(keyword in text for keyword in ("保守", "稳健", "低风险", "跑赢通胀", "conservative")):
        return ClientPortfolioConstraints(
            max_volatility=0.08,
            min_expected_return=0.03,
            max_single_asset_weight=0.15,
        )
    if any(keyword in text for keyword in ("激进", "高风险", "高收益", "aggressive")):
        return ClientPortfolioConstraints(
            max_volatility=0.25,
            min_expected_return=0.10,
            max_single_asset_weight=0.35,
        )
    return ClientPortfolioConstraints(
        max_volatility=0.15,
        min_expected_return=0.06,
        max_single_asset_weight=0.25,
    )


def _build_efficient_frontier(
    posterior_returns: pd.Series,
    posterior_covariance: pd.DataFrame,
    constraints: ClientPortfolioConstraints,
) -> object:
    try:
        from pypfopt.efficient_frontier import EfficientFrontier
    except ImportError as exc:
        raise RuntimeError("Install PyPortfolioOpt with `pip install -e .[portfolio]`.") from exc

    return EfficientFrontier(
        posterior_returns,
        posterior_covariance,
        weight_bounds=(constraints.min_single_asset_weight, constraints.max_single_asset_weight),
    )


def _solve_frontier(
    efficient_frontier: object,
    objective: OptimizationObjective,
    constraints: ClientPortfolioConstraints,
) -> None:
    if objective == "target_volatility":
        if constraints.max_volatility is None:
            raise ValueError("max_volatility is required for target_volatility objective")
        try:
            efficient_frontier.efficient_risk(
                target_volatility=constraints.max_volatility,
                market_neutral=False,
            )
        except ValueError as exc:
            if "minimum volatility" not in str(exc).lower():
                raise
            efficient_frontier.min_volatility()
        return

    if objective == "target_return":
        if constraints.min_expected_return is None:
            raise ValueError("min_expected_return is required for target_return objective")
        try:
            efficient_frontier.efficient_return(
                target_return=constraints.min_expected_return,
                market_neutral=False,
            )
        except ValueError as exc:
            if "minimum volatility" not in str(exc).lower() and "return" not in str(exc).lower():
                raise
            efficient_frontier.max_sharpe(risk_free_rate=constraints.risk_free_rate)
        return

    if objective == "min_volatility":
        efficient_frontier.min_volatility()
        return

    if objective == "max_sharpe":
        efficient_frontier.max_sharpe(risk_free_rate=constraints.risk_free_rate)
        return

    raise ValueError(f"Unsupported optimization objective: {objective}")


def _choose_objective(constraints: ClientPortfolioConstraints) -> OptimizationObjective:
    if constraints.max_volatility is not None:
        return "target_volatility"
    if constraints.min_expected_return is not None:
        return "target_return"
    return "max_sharpe"


def _align_posterior_inputs(
    posterior_returns: pd.Series,
    posterior_covariance: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame]:
    returns = posterior_returns.astype("float64").dropna()
    covariance = posterior_covariance.astype("float64")
    tickers = [ticker for ticker in returns.index if ticker in covariance.index and ticker in covariance.columns]
    if len(tickers) < 2:
        raise ValueError("At least two assets must be present in both returns and covariance")
    return returns.reindex(tickers), covariance.loc[tickers, tickers]


def _validate_inputs(
    posterior_returns: pd.Series,
    posterior_covariance: pd.DataFrame,
    constraints: ClientPortfolioConstraints,
) -> None:
    if posterior_returns.empty:
        raise ValueError("posterior_returns must not be empty")
    if posterior_covariance.empty:
        raise ValueError("posterior_covariance must not be empty")
    if constraints.max_volatility is not None and constraints.max_volatility <= 0:
        raise ValueError("max_volatility must be positive")
    if constraints.min_expected_return is not None and constraints.min_expected_return <= 0:
        raise ValueError("min_expected_return must be positive")
    if not 0 <= constraints.min_single_asset_weight <= constraints.max_single_asset_weight <= 1:
        raise ValueError("single-asset weight bounds must satisfy 0 <= min <= max <= 1")
    if constraints.risk_free_rate < 0:
        raise ValueError("risk_free_rate must be non-negative")


def _make_constraints_feasible(
    constraints: ClientPortfolioConstraints,
    *,
    asset_count: int,
) -> ClientPortfolioConstraints:
    if asset_count <= 0:
        raise ValueError("asset_count must be positive")
    minimum_required_max_weight = 1 / asset_count
    if constraints.max_single_asset_weight >= minimum_required_max_weight:
        return constraints
    return replace(
        constraints,
        max_single_asset_weight=minimum_required_max_weight,
    )


def _constraint_diagnostics(
    posterior_returns: pd.Series,
    posterior_covariance: pd.DataFrame,
    constraints: ClientPortfolioConstraints,
) -> list[str]:
    diagnostics: list[str] = []
    if constraints.min_expected_return is not None:
        max_asset_return = float(posterior_returns.max())
        if constraints.min_expected_return > max_asset_return:
            diagnostics.append(
                f"最低预期收益 {constraints.min_expected_return:.2%} 高于单资产最高后验收益 {max_asset_return:.2%}，可能不可达。"
            )
    if constraints.max_volatility is not None:
        min_variance_proxy = float(posterior_covariance.values.diagonal().min() ** 0.5)
        if constraints.max_volatility < min_variance_proxy * 0.5:
            diagnostics.append(
                f"目标波动率 {constraints.max_volatility:.2%} 可能过低，低于可观察单资产波动率下界的一半。"
            )
    if constraints.max_single_asset_weight < 1 / max(1, len(posterior_returns)):
        diagnostics.append("单只股票上限低于等权配置所需权重，系统会自动放宽到可行下限。")
    return diagnostics
