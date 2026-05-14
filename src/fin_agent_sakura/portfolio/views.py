"""Map LLM-generated investment views into Black-Litterman matrices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd


@dataclass(frozen=True, slots=True)
class BlackLittermanViews:
    """Picking matrix, views vector, and confidences for Black-Litterman."""

    tickers: list[str]
    view_tickers: list[str]
    picking_matrix: pd.DataFrame
    views_vector: pd.Series
    confidences: list[float]


def build_absolute_views_from_llm(
    llm_views: Mapping[str, Mapping[str, Any]],
    tickers: Sequence[str],
    *,
    return_key: str = "expected_excess_return",
    confidence_key: str = "confidence",
    min_confidence: float = 0.0,
    max_confidence: float = 1.0,
) -> BlackLittermanViews:
    """Convert LLM absolute return views into Black-Litterman P and Q.

    Expected input example:
        {
            "AAPL": {"expected_excess_return": 0.05, "confidence": 0.8},
            "MSFT": {"expected_excess_return": 0.03, "confidence": 0.6},
        }

    Each LLM ticker view is interpreted as an absolute view on that asset's
    excess return. For a universe ["AAPL", "MSFT", "NVDA"], the AAPL row in
    the picking matrix is [1, 0, 0], and the corresponding Q value is 0.05.
    """

    universe = _normalize_tickers(tickers)
    ticker_to_column = {ticker: index for index, ticker in enumerate(universe)}

    rows: list[list[float]] = []
    q_values: list[float] = []
    confidences: list[float] = []
    view_tickers: list[str] = []

    for raw_ticker, payload in llm_views.items():
        ticker = raw_ticker.upper().strip()
        if ticker not in ticker_to_column:
            raise ValueError(f"LLM view ticker {ticker} is not in the investable universe")
        if return_key not in payload:
            raise ValueError(f"LLM view for {ticker} missing required key: {return_key}")
        if confidence_key not in payload:
            raise ValueError(f"LLM view for {ticker} missing required key: {confidence_key}")

        expected_return = float(payload[return_key])
        confidence = float(payload[confidence_key])
        if not min_confidence <= confidence <= max_confidence:
            raise ValueError(
                f"Confidence for {ticker} must be between {min_confidence} and {max_confidence}"
            )

        row = [0.0] * len(universe)
        row[ticker_to_column[ticker]] = 1.0
        rows.append(row)
        q_values.append(expected_return)
        confidences.append(confidence)
        view_tickers.append(ticker)

    if not rows:
        raise ValueError("llm_views must contain at least one valid view")

    view_index = [f"view_{idx}_{ticker}" for idx, ticker in enumerate(view_tickers)]
    picking_matrix = pd.DataFrame(rows, index=view_index, columns=universe, dtype="float64")
    views_vector = pd.Series(q_values, index=view_index, name="views", dtype="float64")

    return BlackLittermanViews(
        tickers=universe,
        view_tickers=view_tickers,
        picking_matrix=picking_matrix,
        views_vector=views_vector,
        confidences=confidences,
    )


def build_idzorek_omega(
    views: BlackLittermanViews,
    covariance_matrix: pd.DataFrame,
    prior_returns: pd.Series,
    *,
    tau: float = 0.05,
    risk_aversion: float = 1.0,
) -> pd.DataFrame:
    """Use PyPortfolioOpt's Idzorek method to build the diagonal omega matrix."""

    try:
        from pypfopt.black_litterman import BlackLittermanModel
    except ImportError as exc:
        raise RuntimeError("Install PyPortfolioOpt with `pip install -e .[portfolio]`.") from exc

    aligned_covariance = covariance_matrix.loc[views.tickers, views.tickers]
    aligned_prior = prior_returns.reindex(views.tickers)
    if aligned_prior.isna().any():
        missing = ", ".join(aligned_prior[aligned_prior.isna()].index)
        raise ValueError(f"prior_returns missing tickers: {missing}")

    omega = BlackLittermanModel.idzorek_method(
        view_confidences=views.confidences,
        cov_matrix=aligned_covariance,
        pi=aligned_prior,
        Q=views.views_vector,
        P=views.picking_matrix,
        tau=tau,
        risk_aversion=risk_aversion,
    )
    return pd.DataFrame(omega, index=views.views_vector.index, columns=views.views_vector.index)


def build_black_litterman_model_with_idzorek(
    views: BlackLittermanViews,
    covariance_matrix: pd.DataFrame,
    prior_returns: pd.Series,
    *,
    tau: float = 0.05,
    risk_aversion: float = 1.0,
) -> Any:
    """Create a PyPortfolioOpt BlackLittermanModel using Idzorek confidences."""

    try:
        from pypfopt.black_litterman import BlackLittermanModel
    except ImportError as exc:
        raise RuntimeError("Install PyPortfolioOpt with `pip install -e .[portfolio]`.") from exc

    aligned_covariance = covariance_matrix.loc[views.tickers, views.tickers]
    aligned_prior = prior_returns.reindex(views.tickers)
    if aligned_prior.isna().any():
        missing = ", ".join(aligned_prior[aligned_prior.isna()].index)
        raise ValueError(f"prior_returns missing tickers: {missing}")

    return BlackLittermanModel(
        aligned_covariance,
        pi=aligned_prior,
        Q=views.views_vector,
        P=views.picking_matrix,
        omega="idzorek",
        view_confidences=views.confidences,
        tau=tau,
        risk_aversion=risk_aversion,
    )


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
    return unique

