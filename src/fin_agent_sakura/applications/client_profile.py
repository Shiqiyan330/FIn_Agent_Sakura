"""Client risk-profile questionnaire parsing and persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from fin_agent_sakura.portfolio import ClientPortfolioConstraints, parse_client_constraints_from_text


DEFAULT_CLIENT_PROFILE_PATH = Path("data/processed/client_profile_latest.json")
RiskLevel = Literal["保守型", "稳健型", "平衡型", "成长型", "激进型"]
InvestmentHorizon = Literal["1年以内", "1-3年", "3-5年", "5年以上"]
LiquidityNeed = Literal["高", "中", "低"]


@dataclass(frozen=True, slots=True)
class ClientProfileResult:
    """Parsed client profile and portfolio constraints."""

    generated_at: str
    risk_level: RiskLevel
    horizon: InvestmentHorizon
    liquidity_need: LiquidityNeed
    max_drawdown_tolerance: float
    natural_language_profile: str
    constraints: ClientPortfolioConstraints
    explanation: str
    used_llm: bool = False
    warnings: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "risk_level": self.risk_level,
            "horizon": self.horizon,
            "liquidity_need": self.liquidity_need,
            "max_drawdown_tolerance": self.max_drawdown_tolerance,
            "natural_language_profile": self.natural_language_profile,
            "constraints": {
                "max_volatility": self.constraints.max_volatility,
                "min_expected_return": self.constraints.min_expected_return,
                "max_single_asset_weight": self.constraints.max_single_asset_weight,
                "min_single_asset_weight": self.constraints.min_single_asset_weight,
                "risk_free_rate": self.constraints.risk_free_rate,
            },
            "explanation": self.explanation,
            "used_llm": self.used_llm,
            "warnings": self.warnings or [],
        }


def parse_client_profile_questionnaire(
    *,
    risk_level: RiskLevel,
    horizon: InvestmentHorizon,
    liquidity_need: LiquidityNeed,
    max_drawdown_tolerance: float,
    natural_language_profile: str,
    use_llm: bool = False,
    output_path: str | Path = DEFAULT_CLIENT_PROFILE_PATH,
) -> ClientProfileResult:
    """Convert a non-technical questionnaire into optimization constraints."""

    warnings: list[str] = []
    used_llm = False
    constraints: ClientPortfolioConstraints | None = None

    if use_llm:
        try:
            constraints = _parse_profile_with_llm(
                risk_level=risk_level,
                horizon=horizon,
                liquidity_need=liquidity_need,
                max_drawdown_tolerance=max_drawdown_tolerance,
                natural_language_profile=natural_language_profile,
            )
            used_llm = True
        except Exception as exc:
            warnings.append(f"LLM 客户画像解析失败，已使用规则解析：{type(exc).__name__}: {exc}")

    if constraints is None:
        constraints = _deterministic_constraints(
            risk_level=risk_level,
            horizon=horizon,
            liquidity_need=liquidity_need,
            max_drawdown_tolerance=max_drawdown_tolerance,
            natural_language_profile=natural_language_profile,
        )

    result = ClientProfileResult(
        generated_at=pd.Timestamp.utcnow().isoformat(),
        risk_level=risk_level,
        horizon=horizon,
        liquidity_need=liquidity_need,
        max_drawdown_tolerance=max_drawdown_tolerance,
        natural_language_profile=natural_language_profile,
        constraints=constraints,
        explanation=_explain_constraints(constraints, risk_level, horizon, liquidity_need, max_drawdown_tolerance),
        used_llm=used_llm,
        warnings=warnings,
    )
    _save_profile(result, output_path)
    return result


def load_latest_client_profile(path: str | Path = DEFAULT_CLIENT_PROFILE_PATH) -> ClientProfileResult | None:
    """Load the latest parsed client profile."""

    profile_path = Path(path)
    if not profile_path.exists():
        return None
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    raw_constraints = payload.get("constraints") or {}
    return ClientProfileResult(
        generated_at=str(payload.get("generated_at", "")),
        risk_level=payload.get("risk_level", "稳健型"),
        horizon=payload.get("horizon", "3-5年"),
        liquidity_need=payload.get("liquidity_need", "中"),
        max_drawdown_tolerance=float(payload.get("max_drawdown_tolerance", 0.15)),
        natural_language_profile=str(payload.get("natural_language_profile", "")),
        constraints=ClientPortfolioConstraints(
            max_volatility=raw_constraints.get("max_volatility"),
            min_expected_return=raw_constraints.get("min_expected_return"),
            max_single_asset_weight=float(raw_constraints.get("max_single_asset_weight", 0.25)),
            min_single_asset_weight=float(raw_constraints.get("min_single_asset_weight", 0.0)),
            risk_free_rate=float(raw_constraints.get("risk_free_rate", 0.02)),
        ),
        explanation=str(payload.get("explanation", "")),
        used_llm=bool(payload.get("used_llm", False)),
        warnings=list(payload.get("warnings") or []),
    )


def _deterministic_constraints(
    *,
    risk_level: RiskLevel,
    horizon: InvestmentHorizon,
    liquidity_need: LiquidityNeed,
    max_drawdown_tolerance: float,
    natural_language_profile: str,
) -> ClientPortfolioConstraints:
    level_map: dict[RiskLevel, tuple[float, float, float]] = {
        "保守型": (0.08, 0.03, 0.15),
        "稳健型": (0.11, 0.045, 0.20),
        "平衡型": (0.15, 0.06, 0.25),
        "成长型": (0.20, 0.08, 0.30),
        "激进型": (0.25, 0.10, 0.35),
    }
    max_volatility, min_return, max_weight = level_map[risk_level]

    if horizon in {"1年以内", "1-3年"}:
        max_volatility *= 0.85
        max_weight = min(max_weight, 0.22)
    elif horizon == "5年以上":
        min_return += 0.01

    if liquidity_need == "高":
        max_volatility *= 0.85
        max_weight = min(max_weight, 0.20)
    elif liquidity_need == "低":
        min_return += 0.005

    if max_drawdown_tolerance > 0:
        max_volatility = min(max_volatility, max_drawdown_tolerance * 0.8)

    text_constraints = parse_client_constraints_from_text(natural_language_profile or risk_level)
    return ClientPortfolioConstraints(
        max_volatility=round(min(max_volatility, text_constraints.max_volatility or max_volatility), 4),
        min_expected_return=round(max(min_return, text_constraints.min_expected_return or min_return), 4),
        max_single_asset_weight=round(min(max_weight, text_constraints.max_single_asset_weight), 4),
        risk_free_rate=0.02,
    )


def _parse_profile_with_llm(
    *,
    risk_level: RiskLevel,
    horizon: InvestmentHorizon,
    liquidity_need: LiquidityNeed,
    max_drawdown_tolerance: float,
    natural_language_profile: str,
) -> ClientPortfolioConstraints:
    from openai import OpenAI

    from fin_agent_sakura.config import get_llm_config

    cfg = get_llm_config()
    if not cfg.api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
    prompt = (
        "你是投资顾问的客户画像解析器。只返回JSON，不要markdown。"
        "字段必须是 max_volatility, min_expected_return, max_single_asset_weight。"
        "数值使用小数，例如 0.12 表示 12%。"
        "约束范围：max_volatility 0.05-0.30, min_expected_return 0.02-0.12, "
        "max_single_asset_weight 0.10-0.40。\n"
        f"风险等级：{risk_level}\n投资期限：{horizon}\n流动性需求：{liquidity_need}\n"
        f"最大可承受回撤：{max_drawdown_tolerance:.2%}\n用户描述：{natural_language_profile}"
    )
    response = client.chat.completions.create(
        model=cfg.chat_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
    )
    content = response.choices[0].message.content or "{}"
    payload = _parse_json_object(content)
    return ClientPortfolioConstraints(
        max_volatility=_clip(float(payload.get("max_volatility", 0.15)), 0.05, 0.30),
        min_expected_return=_clip(float(payload.get("min_expected_return", 0.06)), 0.02, 0.12),
        max_single_asset_weight=_clip(float(payload.get("max_single_asset_weight", 0.25)), 0.10, 0.40),
        risk_free_rate=0.02,
    )


def _explain_constraints(
    constraints: ClientPortfolioConstraints,
    risk_level: RiskLevel,
    horizon: InvestmentHorizon,
    liquidity_need: LiquidityNeed,
    max_drawdown_tolerance: float,
) -> str:
    return (
        f"根据{risk_level}、投资期限{horizon}、流动性需求{liquidity_need}和"
        f"{max_drawdown_tolerance:.0%}最大可承受回撤，系统建议目标年化波动率不超过"
        f"{(constraints.max_volatility or 0):.1%}，最低期望收益不低于"
        f"{(constraints.min_expected_return or 0):.1%}，单只股票权重不超过"
        f"{constraints.max_single_asset_weight:.1%}。"
    )


def _parse_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    if "```" in text:
        text = text.replace("```json", "```")
        parts = text.split("```")
        text = max(parts, key=len).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end >= start:
        text = text[start : end + 1]
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("LLM did not return a JSON object")
    return parsed


def _clip(value: float, lower: float, upper: float) -> float:
    return round(max(lower, min(upper, value)), 4)


def _save_profile(result: ClientProfileResult, path: str | Path) -> None:
    profile_path = Path(path)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
