"""A-share investment assistant pipeline for paper deployment."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from fin_agent_sakura.config import get_llm_config
from fin_agent_sakura.data import MarketDataClientFactory, TechnicalIndicators
from fin_agent_sakura.monitoring import RebalanceEvent, SentimentSignal, TechnicalSignal, TimingRuleEngine
from fin_agent_sakura.portfolio import (
    ClientPortfolioConstraints,
    build_absolute_views_from_llm,
    build_black_litterman_model_with_idzorek,
    build_market_equilibrium_prior_async,
    optimize_bl_portfolio_weights,
    parse_client_constraints_from_text,
)
from fin_agent_sakura.applications.rebalance_log import append_rebalance_analysis_events
from fin_agent_sakura.applications.risk_gate import evaluate_paper_orders_risk
from fin_agent_sakura.storage import PositionMemory


@dataclass(frozen=True, slots=True)
class StockCandidate:
    ticker: str
    name: str
    sector: str


@dataclass(frozen=True, slots=True)
class ChinaInvestmentResult:
    generated_at: str
    mode: str
    profile_text: str
    universe_size: int
    selected: list[dict[str, Any]]
    target_weights: dict[str, float]
    current_weights: dict[str, float]
    drift_alerts: list[dict[str, Any]]
    trade_orders: list[dict[str, Any]]
    risk_gate: dict[str, Any] | None
    research_report_html: str
    warnings: list[str]
    portfolio_engine: str = "black_litterman"
    portfolio_diagnostics: list[str] | None = None
    black_litterman_summary: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "mode": self.mode,
            "profile_text": self.profile_text,
            "universe_size": self.universe_size,
            "selected": self.selected,
            "target_weights": self.target_weights,
            "current_weights": self.current_weights,
            "drift_alerts": self.drift_alerts,
            "trade_orders": self.trade_orders,
            "risk_gate": self.risk_gate,
            "research_report_html": self.research_report_html,
            "warnings": self.warnings,
            "portfolio_engine": self.portfolio_engine,
            "portfolio_diagnostics": self.portfolio_diagnostics or [],
            "black_litterman_summary": self.black_litterman_summary or {},
        }


class ChinaInvestmentAssistant:
    """Run a safe A-share research and paper-deployment workflow."""

    def __init__(self, output_dir: str | Path = "data/processed") -> None:
        self.output_dir = Path(output_dir)

    async def run(
        self,
        *,
        profile_text: str,
        max_candidates: int = 80,
        selected_count: int = 10,
        use_llm_report: bool = True,
        current_weights: dict[str, float] | None = None,
        universe: list[StockCandidate] | None = None,
        drift_threshold: float = 0.05,
    ) -> ChinaInvestmentResult:
        warnings: list[str] = []
        candidate_universe = (universe or _default_a_share_universe())[:max_candidates]
        market_rows = await self._score_universe(candidate_universe, warnings)
        mode = "live_data" if not warnings else "offline_fallback"
        selected_count = min(max(selected_count, 5), len(market_rows))
        selected = market_rows.head(selected_count).reset_index(drop=True)

        constraints = parse_client_constraints_from_text(profile_text)
        portfolio_engine = "black_litterman"
        black_litterman_summary: dict[str, Any] = {}
        try:
            target_weights, black_litterman_summary = await _build_black_litterman_target_weights(
                selected,
                constraints,
                warnings,
            )
        except Exception as exc:
            portfolio_engine = "score_fallback_after_black_litterman_failure"
            warnings.append(
                "Black-Litterman完整链路未能完成，已显式降级为组合经理分数加权；"
                f"请优先修复行情/市值数据源后再用于真实调仓研究：{type(exc).__name__}: {exc}"
            )
            target_weights = _build_target_weights(selected, constraints.max_single_asset_weight)
            black_litterman_summary = {
                "attempted": True,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        actual_weights = current_weights or PositionMemory().load_weights() or _sample_current_weights(target_weights)
        drift_alerts, trade_orders = _build_rebalance_plan(
            actual_weights,
            target_weights,
            selected,
            drift_threshold=drift_threshold,
        )
        risk_gate = None
        if trade_orders:
            risk_report = await asyncio.to_thread(
                evaluate_paper_orders_risk,
                current_weights=actual_weights,
                orders=trade_orders,
                market="cn",
            )
            risk_gate = risk_report.to_dict()
            _apply_risk_gate_decision(trade_orders, drift_alerts, risk_gate)
            if not risk_report.approved:
                warnings.append("风险断路器拒绝了本次纸面订单，已保留订单为研究记录，不应作为可执行指令。")
        report_html = await self._build_report(profile_text, selected, target_weights, warnings, use_llm_report)

        generated_at = pd.Timestamp.now().isoformat()
        result = ChinaInvestmentResult(
            generated_at=generated_at,
            mode=mode,
            profile_text=profile_text,
            universe_size=len(candidate_universe),
            selected=selected.to_dict("records"),
            target_weights=target_weights,
            current_weights=actual_weights,
            drift_alerts=drift_alerts,
            trade_orders=trade_orders,
            risk_gate=risk_gate,
            research_report_html=report_html,
            warnings=warnings,
            portfolio_engine=portfolio_engine,
            portfolio_diagnostics=list(black_litterman_summary.get("diagnostics") or []),
            black_litterman_summary=black_litterman_summary,
        )
        self.save_result(result)
        append_rebalance_analysis_events(
            generated_at=generated_at,
            drift_alerts=drift_alerts,
            trade_orders=trade_orders,
            risk_gate=risk_gate,
        )
        return result

    def save_result(self, result: ChinaInvestmentResult) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "china_investment_result.json"
        path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def load_latest_result(self) -> ChinaInvestmentResult | None:
        path = self.output_dir / "china_investment_result.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.setdefault("risk_gate", None)
        payload.setdefault("portfolio_engine", "legacy_unknown")
        payload.setdefault("portfolio_diagnostics", [])
        payload.setdefault("black_litterman_summary", {})
        return ChinaInvestmentResult(**payload)

    async def _score_universe(self, universe: list[StockCandidate], warnings: list[str]) -> pd.DataFrame:
        try:
            live_rows = await self._score_universe_live(universe)
            if not live_rows.empty:
                return live_rows
        except Exception as exc:
            warnings.append(f"A股实时数据源暂不可用，已使用离线演示数据继续流程：{type(exc).__name__}: {exc}")
        return _offline_scored_universe(universe)

    async def _score_universe_live(self, universe: list[StockCandidate]) -> pd.DataFrame:
        client = MarketDataClientFactory.get_client("cn")
        start = date.today() - timedelta(days=365)
        rows: list[dict[str, Any]] = []
        for candidate in universe[:30]:
            frame = await client.fetch_ohlcv(candidate.ticker, start=start, adjusted=True)
            indicators = TechnicalIndicators(engine="pandas").calculate(frame)
            latest = indicators.iloc[-1]
            close = float(latest["close"])
            ma_50 = float(latest.get("ma_50")) if pd.notna(latest.get("ma_50")) else close
            ma_200 = float(latest.get("ma_200")) if pd.notna(latest.get("ma_200")) else close
            rsi = float(latest.get("rsi_14")) if pd.notna(latest.get("rsi_14")) else 50.0
            volume_ratio = float(latest.get("volume_ratio")) if pd.notna(latest.get("volume_ratio")) else 1.0
            volume_signal = str(latest.get("volume_signal") or "unavailable")
            returns = pd.to_numeric(frame["close"], errors="coerce").pct_change().dropna()
            momentum = close / float(frame["close"].iloc[max(0, len(frame) - 60)]) - 1
            volatility = float(returns.std() * (252**0.5)) if not returns.empty else 0.2
            score = momentum - 0.35 * volatility + (0.05 if close > ma_200 else -0.05)
            if rsi > 75:
                score -= 0.08
            trend_label = "趋势向上" if close >= ma_50 >= ma_200 else "趋势向下" if close < ma_50 < ma_200 else "震荡"
            rsi_label = "超买" if rsi > 70 else "超卖" if rsi < 30 else "中性"
            rows.append(
                {
                    "ticker": candidate.ticker,
                    "name": candidate.name,
                    "sector": candidate.sector,
                    "close": close,
                    "momentum_60d": momentum,
                    "volatility": volatility,
                    "rsi_14": rsi,
                    "ma_50": ma_50,
                    "ma_200": ma_200,
                    "trend_label": trend_label,
                    "rsi_label": rsi_label,
                    "volume_ratio": volume_ratio,
                    "volume_signal": volume_signal,
                    "score": score,
                }
            )
        return pd.DataFrame(rows).sort_values("score", ascending=False)

    async def _build_report(
        self,
        profile_text: str,
        selected: pd.DataFrame,
        target_weights: dict[str, float],
        warnings: list[str],
        use_llm_report: bool,
    ) -> str:
        fallback = _fallback_report_html(profile_text, selected, target_weights, warnings)
        if not use_llm_report:
            return fallback
        try:
            from openai import OpenAI

            cfg = get_llm_config()
            client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
            prompt = (
                "你是A股智能投顾助手。请基于以下候选组合生成一份适合普通投资者阅读的HTML研报，"
                "必须包含：组合定位、主要持仓、风险、再平衡建议。不要推荐绕过风控直接下单。\n"
                f"客户画像：{profile_text}\n"
                f"候选组合：{selected[['ticker','name','sector','score']].to_dict('records')}\n"
                f"目标权重：{target_weights}\n"
                f"警告：{warnings}"
            )
            stream = client.chat.completions.create(
                model=cfg.chat_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=900,
                stream=True,
            )
            chunks: list[str] = []
            for chunk in stream:
                choice = chunk.choices[0] if getattr(chunk, "choices", None) else None
                delta = getattr(getattr(choice, "delta", None), "content", None) if choice is not None else None
                if delta:
                    chunks.append(delta)
            return "".join(chunks).strip() or fallback
        except Exception as exc:
            warnings.append(f"LLM研报生成失败，已使用本地模板：{type(exc).__name__}: {exc}")
            return fallback


def run_china_investment_assistant(
    profile_text: str,
    *,
    max_candidates: int = 80,
    selected_count: int = 10,
    use_llm_report: bool = True,
    current_weights: dict[str, float] | None = None,
    universe: list[StockCandidate] | None = None,
    drift_threshold: float = 0.05,
) -> ChinaInvestmentResult:
    return asyncio.run(
        ChinaInvestmentAssistant().run(
            profile_text=profile_text,
            max_candidates=max_candidates,
            selected_count=selected_count,
            use_llm_report=use_llm_report,
            current_weights=current_weights,
            universe=universe,
            drift_threshold=drift_threshold,
        )
    )


async def _build_black_litterman_target_weights(
    selected: pd.DataFrame,
    constraints: ClientPortfolioConstraints,
    warnings: list[str],
) -> tuple[dict[str, float], dict[str, Any]]:
    tickers = [str(ticker).upper() for ticker in selected["ticker"].tolist()]
    if len(tickers) < 2:
        raise ValueError("Black-Litterman组合优化至少需要2只股票")

    prior = await build_market_equilibrium_prior_async(tickers, market="cn", years=5)
    llm_views = _build_manager_views_from_selected(selected)
    views = build_absolute_views_from_llm(llm_views, tickers, min_confidence=0.05)
    bl_model = build_black_litterman_model_with_idzorek(
        views,
        prior.covariance_matrix,
        prior.implied_prior_returns,
        risk_aversion=prior.risk_aversion,
    )
    posterior_returns = bl_model.bl_returns()
    posterior_covariance = bl_model.bl_cov()
    optimized = optimize_bl_portfolio_weights(
        posterior_returns,
        posterior_covariance,
        constraints=constraints,
    )

    weights = optimized.weights.astype("float64")
    if weights.sum() <= 0:
        raise ValueError("Black-Litterman优化返回空权重")
    weights = weights / weights.sum()
    target_weights = {str(ticker): float(weight) for ticker, weight in weights.items() if float(weight) > 1e-4}
    if len(target_weights) < 2:
        warnings.append("Black-Litterman结果过度集中，已保留优化结果但建议人工复核单票集中风险。")

    summary = {
        "attempted": True,
        "status": "success",
        "engine": "market_prior+views+optimizer",
        "tickers": tickers,
        "view_tickers": views.view_tickers,
        "view_confidences": views.confidences,
        "view_expected_excess_returns": {
            str(index): float(value) for index, value in views.views_vector.items()
        },
        "market_weights": {str(ticker): float(value) for ticker, value in prior.market_weights.items()},
        "implied_prior_returns": {
            str(ticker): float(value) for ticker, value in prior.implied_prior_returns.items()
        },
        "posterior_returns": {str(ticker): float(value) for ticker, value in posterior_returns.items()},
        "objective": optimized.objective,
        "expected_return": optimized.expected_return,
        "volatility": optimized.volatility,
        "sharpe_ratio": optimized.sharpe_ratio,
        "fallback_used": optimized.fallback_used,
        "diagnostics": optimized.diagnostics or [],
    }
    return target_weights, summary


def _build_manager_views_from_selected(selected: pd.DataFrame) -> dict[str, dict[str, float]]:
    score = pd.to_numeric(selected["score"], errors="coerce").fillna(0.0)
    momentum = pd.to_numeric(selected["momentum_60d"], errors="coerce").fillna(0.0)
    volatility = pd.to_numeric(selected["volatility"], errors="coerce").fillna(0.25).clip(lower=0.05)
    rsi = pd.to_numeric(selected["rsi_14"], errors="coerce").fillna(50.0)

    centered_score = score - float(score.median())
    score_scale = float(centered_score.abs().max()) or 1.0
    momentum_component = momentum.clip(-0.3, 0.3) * 0.12
    score_component = (centered_score / score_scale).clip(-1.0, 1.0) * 0.045
    volatility_penalty = (volatility - float(volatility.median())).clip(lower=0.0) * 0.08
    rsi_penalty = ((rsi - 70.0).clip(lower=0.0) / 100.0) * 0.08
    expected = (0.035 + momentum_component + score_component - volatility_penalty - rsi_penalty).clip(-0.08, 0.12)

    confidence = (0.45 + score.abs().rank(pct=True) * 0.25 + momentum.abs().rank(pct=True) * 0.15).clip(0.2, 0.85)
    views: dict[str, dict[str, float]] = {}
    for idx, row in selected.reset_index(drop=True).iterrows():
        ticker = str(row["ticker"]).upper()
        views[ticker] = {
            "expected_excess_return": float(expected.iloc[idx]),
            "confidence": float(confidence.iloc[idx]),
        }
    return views


def _build_target_weights(selected: pd.DataFrame, max_single_weight: float) -> dict[str, float]:
    scores = selected["score"].clip(lower=0)
    if scores.sum() <= 0:
        raw = pd.Series(1 / len(selected), index=selected["ticker"])
    else:
        raw = pd.Series(scores.values / scores.sum(), index=selected["ticker"])
    capped = raw.clip(upper=max_single_weight)
    residual = 1 - capped.sum()
    if residual > 0:
        uncapped = capped[capped < max_single_weight]
        if not uncapped.empty:
            capped.loc[uncapped.index] += residual * uncapped / uncapped.sum()
    normalized = capped / capped.sum()
    return {ticker: float(weight) for ticker, weight in normalized.items()}


def _sample_current_weights(target_weights: dict[str, float]) -> dict[str, float]:
    tickers = list(target_weights)
    current: dict[str, float] = {}
    for idx, ticker in enumerate(tickers):
        drift = 0.035 if idx % 3 == 0 else -0.025 if idx % 3 == 1 else 0.0
        current[ticker] = max(0.0, target_weights[ticker] + drift)
    total = sum(current.values())
    return {ticker: weight / total for ticker, weight in current.items()}


def _build_rebalance_plan(
    current_weights: dict[str, float],
    target_weights: dict[str, float],
    selected: pd.DataFrame,
    *,
    drift_threshold: float = 0.05,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    engine = TimingRuleEngine()
    selected_by_ticker = {str(row["ticker"]).upper(): row for _, row in selected.iterrows()}
    technicals = {
        str(row["ticker"]).upper(): TechnicalSignal(
            ticker=str(row["ticker"]).upper(),
            rsi_14=float(row["rsi_14"]),
            macd=1.0 if row["momentum_60d"] > 0 else -1.0,
            macd_signal=0.0,
            price_above_ma_50=bool(row["close"] >= row["ma_50"]),
            price_above_ma_200=bool(row["close"] >= row["ma_200"]),
        )
        for _, row in selected.iterrows()
    }
    alerts: list[dict[str, Any]] = []
    orders: list[dict[str, Any]] = []
    for ticker, target in target_weights.items():
        current = current_weights.get(ticker, 0.0)
        drift = current - target
        if abs(drift) <= drift_threshold:
            continue
        ticker_key = ticker.upper()
        technical = technicals.get(ticker_key)
        if technical is None:
            continue
        row = selected_by_ticker.get(ticker_key)
        action = "sell" if drift > 0 else "buy"
        event = RebalanceEvent("paper_client", ticker, current, target, drift, drift_threshold)
        sentiment = SentimentSignal(ticker=ticker, score=0.35 if target > current else -0.1, confidence=0.7)
        result = engine.evaluate(event, technical, sentiment)
        technical_signal = _technical_signal_label(technical)
        rule_label = _rule_execution_label(result.decision, None)
        suggested_batches = _suggested_batches(action, abs(drift), technical, sentiment)
        common = {
            "ticker": ticker,
            "action": action,
            "current_weight": current,
            "target_weight": target,
            "drift": drift,
            "threshold": drift_threshold,
            "technical_signal": technical_signal,
            "technical_rsi_14": technical.rsi_14,
            "technical_momentum": "bullish" if technical.has_bullish_momentum else "bearish" if technical.has_bearish_momentum else "mixed",
            "price_above_ma_50": technical.price_above_ma_50,
            "price_above_ma_200": technical.price_above_ma_200,
            "sentiment_score": sentiment.score,
            "sentiment_confidence": sentiment.confidence,
            "decision": result.decision,
            "execution_label": rule_label,
            "reason": " ".join(result.reasons),
            "reasons": result.reasons,
        }
        if row is not None:
            common["name"] = row.get("name")
            common["trend_label"] = row.get("trend_label")
            common["rsi_label"] = row.get("rsi_label")
        alerts.append(
            {
                **common,
                "suggested_batches": suggested_batches,
                "batch_weight_delta": abs(drift) / suggested_batches,
            }
        )
        if result.order:
            orders.append(
                {
                    **common,
                    "ticker": result.order.ticker,
                    "action": result.order.action,
                    "target_weight_delta": result.order.target_weight_delta,
                    "suggested_batches": suggested_batches,
                    "batch_weight_delta": result.order.target_weight_delta / suggested_batches,
                    "reason": result.order.reason,
                }
            )
    return alerts, orders


def _technical_signal_label(signal: TechnicalSignal) -> str:
    if signal.is_overbought:
        return "overbought"
    if signal.is_oversold:
        return "oversold"
    if signal.has_bullish_momentum:
        return "bullish"
    if signal.has_bearish_momentum:
        return "bearish"
    return "mixed"


def _suggested_batches(
    action: str,
    target_weight_delta: float,
    technical_signal: TechnicalSignal,
    sentiment_signal: SentimentSignal,
) -> int:
    if action != "buy":
        return 1
    if target_weight_delta >= 0.08 or sentiment_signal.weighted_score < 0.35 or technical_signal.is_overbought:
        return 3
    if target_weight_delta >= 0.04:
        return 2
    return 1


def _rule_execution_label(decision: str, risk_gate_decision: str | None) -> str:
    if risk_gate_decision == "rejected":
        return "风控拒绝"
    if decision == "defer":
        return "推迟执行"
    if decision == "execute" and risk_gate_decision == "approved":
        return "需要人工确认"
    if decision == "execute":
        return "立即执行"
    return "忽略"


def _apply_risk_gate_decision(
    trade_orders: list[dict[str, Any]],
    drift_alerts: list[dict[str, Any]],
    risk_gate: dict[str, Any],
) -> None:
    decision = str(risk_gate.get("decision") or "not_run")
    for order in trade_orders:
        order["risk_gate_decision"] = decision
        order["execution_label"] = _rule_execution_label(str(order.get("decision") or "execute"), decision)
    order_tickers = {str(order.get("ticker", "")).upper(): order for order in trade_orders}
    for alert in drift_alerts:
        order = order_tickers.get(str(alert.get("ticker", "")).upper())
        if order is not None:
            alert["risk_gate_decision"] = decision
            alert["execution_label"] = order["execution_label"]
        else:
            alert["risk_gate_decision"] = "not_run"
            alert["execution_label"] = _rule_execution_label(str(alert.get("decision") or "defer"), None)


def _offline_scored_universe(universe: list[StockCandidate]) -> pd.DataFrame:
    rows = []
    for idx, candidate in enumerate(universe):
        sector_bonus = {
            "消费": 0.08,
            "医药": 0.06,
            "新能源": 0.05,
            "金融": 0.03,
            "科技": 0.04,
            "工业": 0.02,
        }.get(candidate.sector, 0.01)
        momentum = 0.14 - (idx % 9) * 0.018
        volatility = 0.18 + (idx % 7) * 0.025
        rsi = 42 + (idx % 8) * 5
        score = sector_bonus + momentum - 0.25 * volatility
        close = 20 + idx * 3.7
        rows.append(
            {
                "ticker": candidate.ticker,
                "name": candidate.name,
                "sector": candidate.sector,
                "close": close,
                "momentum_60d": momentum,
                "volatility": volatility,
                "rsi_14": rsi,
                "ma_50": close * 0.98,
                "ma_200": close * 0.95,
                "trend_label": "趋势向上",
                "rsi_label": "超买" if rsi > 70 else "超卖" if rsi < 30 else "中性",
                "volume_ratio": 1.0 + (idx % 5) * 0.18,
                "volume_signal": "expanding" if idx % 5 >= 3 else "normal",
                "score": score,
            }
        )
    return pd.DataFrame(rows).sort_values("score", ascending=False)


def _fallback_report_html(
    profile_text: str,
    selected: pd.DataFrame,
    target_weights: dict[str, float],
    warnings: list[str],
) -> str:
    rows = "".join(
        f"<li>{row['ticker']} {row['name']}（{row['sector']}）：目标权重 {target_weights[row['ticker']]:.1%}</li>"
        for _, row in selected.head(10).iterrows()
    )
    warning_html = "".join(f"<li>{warning}</li>" for warning in warnings) or "<li>暂无系统警告。</li>"
    return f"""
    <section style="font-family: Arial, sans-serif; line-height:1.55;">
      <h2>A股纸面投资部署报告</h2>
      <p><strong>客户画像：</strong>{profile_text}</p>
      <p><strong>组合定位：</strong>以宽股票池筛选后的优质 A 股为候选，采用分散持仓和漂移再平衡，不自动下单。</p>
      <h3>建议持仓</h3>
      <ul>{rows}</ul>
      <h3>风险提示</h3>
      <ul>{warning_html}</ul>
      <p><strong>执行原则：</strong>仅生成纸面订单；真实交易前必须人工确认、检查流动性和风险限额。</p>
    </section>
    """


def _default_a_share_universe() -> list[StockCandidate]:
    return [
        StockCandidate("600519.SH", "贵州茅台", "消费"),
        StockCandidate("000858.SZ", "五粮液", "消费"),
        StockCandidate("000333.SZ", "美的集团", "消费"),
        StockCandidate("600887.SH", "伊利股份", "消费"),
        StockCandidate("300750.SZ", "宁德时代", "新能源"),
        StockCandidate("002594.SZ", "比亚迪", "新能源"),
        StockCandidate("601012.SH", "隆基绿能", "新能源"),
        StockCandidate("600900.SH", "长江电力", "公用事业"),
        StockCandidate("601318.SH", "中国平安", "金融"),
        StockCandidate("600036.SH", "招商银行", "金融"),
        StockCandidate("601166.SH", "兴业银行", "金融"),
        StockCandidate("600030.SH", "中信证券", "金融"),
        StockCandidate("300760.SZ", "迈瑞医疗", "医药"),
        StockCandidate("600276.SH", "恒瑞医药", "医药"),
        StockCandidate("300015.SZ", "爱尔眼科", "医药"),
        StockCandidate("000661.SZ", "长春高新", "医药"),
        StockCandidate("002415.SZ", "海康威视", "科技"),
        StockCandidate("000725.SZ", "京东方A", "科技"),
        StockCandidate("603501.SH", "韦尔股份", "科技"),
        StockCandidate("688981.SH", "中芯国际", "科技"),
        StockCandidate("601899.SH", "紫金矿业", "资源"),
        StockCandidate("600309.SH", "万华化学", "工业"),
        StockCandidate("601088.SH", "中国神华", "资源"),
        StockCandidate("600406.SH", "国电南瑞", "工业"),
        StockCandidate("601668.SH", "中国建筑", "工业"),
        StockCandidate("000002.SZ", "万科A", "地产"),
        StockCandidate("600048.SH", "保利发展", "地产"),
        StockCandidate("601888.SH", "中国中免", "消费"),
        StockCandidate("600690.SH", "海尔智家", "消费"),
        StockCandidate("002352.SZ", "顺丰控股", "工业"),
    ]
