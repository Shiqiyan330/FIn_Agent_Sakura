"""Conversational robo-advisor orchestration for non-technical users."""

from __future__ import annotations

import csv
import copy
import json
import re
import textwrap
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import pandas as pd

from fin_agent_sakura.applications.a_share_universe import build_a_share_universe, load_latest_a_share_universe
from fin_agent_sakura.applications.china_investment_assistant import ChinaInvestmentAssistant, StockCandidate
from fin_agent_sakura.applications.client_profile import parse_client_profile_questionnaire
from fin_agent_sakura.applications.search_agent import SearchAgentResult, run_search_agent
from fin_agent_sakura.applications.user_accounts import (
    LocalUserAccount,
    get_or_create_active_user,
    load_user_account,
    save_user_account,
    user_data_dir,
)
from fin_agent_sakura.config import get_llm_config
from fin_agent_sakura.storage import SQLiteStore, record_llm_usage


DEFAULT_CONVERSATION_DIR = Path("data/processed/conversations")
DEFAULT_CONVERSATION_LATEST = Path("data/processed/conversation_latest.json")
DEFAULT_STOCK_SELECTION_LIMIT = 8
MAX_MANUAL_TICKERS = 10
AdvisorStage = Literal["profile", "universe", "portfolio", "completed"]
AgentProgressCallback = Callable[["AgentCallEvent"], None]


@dataclass(frozen=True, slots=True)
class AdvisorMessage:
    """One chat message in the conversational advisor."""

    role: Literal["user", "assistant", "system"]
    content: str
    created_at: str = field(default_factory=lambda: pd.Timestamp.now().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AgentCallEvent:
    """Visible trace for one advisor agent/tool call."""

    agent: str
    status: Literal["pending", "running", "success", "failed", "skipped"]
    summary: str
    input_summary: str = ""
    output_summary: str = ""
    elapsed_seconds: float = 0.0
    error: str = ""
    created_at: str = field(default_factory=lambda: pd.Timestamp.now().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AgentCallEvent":
        return cls(
            agent=str(payload.get("agent") or "unknown"),
            status=payload.get("status", "success"),
            summary=str(payload.get("summary") or ""),
            input_summary=str(payload.get("input_summary") or ""),
            output_summary=str(payload.get("output_summary") or ""),
            elapsed_seconds=float(payload.get("elapsed_seconds") or 0.0),
            error=str(payload.get("error") or ""),
            created_at=str(payload.get("created_at") or pd.Timestamp.now().isoformat()),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True, slots=True)
class TurnContext:
    """Resolved context for one user turn."""

    raw_message: str
    expanded_intent: str
    recent_messages: list[dict[str, Any]]
    is_short_reply: bool
    is_option_reply: bool
    option_value: str = ""
    referenced_prompt: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ConversationalAdvisorSession:
    """Persistable local conversation state."""

    session_id: str
    stage: AdvisorStage
    messages: list[AdvisorMessage]
    user_id: str = "default_user"
    investment_role: str = "个人投资者"
    goals: str = ""
    profile_text: str = ""
    tickers: list[str] = field(default_factory=list)
    market: str = "cn"
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    latest_result: dict[str, Any] = field(default_factory=dict)
    stock_selection: dict[str, Any] = field(default_factory=dict)
    agent_events: list[AgentCallEvent] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    updated_at: str = field(default_factory=lambda: pd.Timestamp.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "stage": self.stage,
            "messages": [message.to_dict() for message in self.messages],
            "user_id": self.user_id,
            "investment_role": self.investment_role,
            "goals": self.goals,
            "profile_text": self.profile_text,
            "tickers": self.tickers,
            "market": self.market,
            "artifacts": self.artifacts,
            "latest_result": self.latest_result,
            "stock_selection": self.stock_selection,
            "agent_events": [event.to_dict() for event in self.agent_events],
            "warnings": self.warnings,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ConversationalAdvisorSession":
        return cls(
            session_id=str(payload.get("session_id") or uuid.uuid4().hex[:12]),
            stage=payload.get("stage", "profile"),
            messages=[AdvisorMessage(**item) for item in payload.get("messages", [])],
            user_id=str(payload.get("user_id") or "default_user"),
            investment_role=str(payload.get("investment_role") or "个人投资者"),
            goals=str(payload.get("goals") or ""),
            profile_text=str(payload.get("profile_text") or ""),
            tickers=list(payload.get("tickers") or []),
            market=str(payload.get("market") or "cn"),
            artifacts=list(payload.get("artifacts") or []),
            latest_result=dict(payload.get("latest_result") or {}),
            stock_selection=dict(payload.get("stock_selection") or {}),
            agent_events=[AgentCallEvent.from_dict(item) for item in payload.get("agent_events", [])],
            warnings=list(payload.get("warnings") or []),
            updated_at=str(payload.get("updated_at") or pd.Timestamp.now().isoformat()),
        )


def start_conversational_advisor_session(
    *,
    market: str = "cn",
    user: LocalUserAccount | None = None,
    output_dir: str | Path = DEFAULT_CONVERSATION_DIR,
) -> ConversationalAdvisorSession:
    """Create and persist a new advisor conversation."""

    account = user or get_or_create_active_user()
    goal_note = f"我已经记得你的当前目标：{account.goals}。" if account.goals else "你可以直接告诉我这次投资想达成什么目标。"
    session = ConversationalAdvisorSession(
        session_id=uuid.uuid4().hex[:12],
        stage="profile",
        market=market,
        user_id=account.user_id,
        investment_role=account.investment_role,
        goals=account.goals,
        messages=[
            AdvisorMessage(
                role="assistant",
                content=(
                    f"你好，{account.display_name}。我是 Sakura 对话式投顾，会以“{account.investment_role}”的偏好来辅助你。"
                    f"{goal_note} 接下来你可以自然地说出风险承受能力、投资期限、偏好的股票池，"
                    "也可以直接说“沪深300”或给我几个 A 股代码。"
                ),
            )
        ],
    )
    _save_session(session, output_dir=output_dir)
    return session


def load_latest_conversational_advisor_session(
    *,
    user_id: str | None = None,
    latest_path: str | Path = DEFAULT_CONVERSATION_LATEST,
) -> ConversationalAdvisorSession | None:
    if user_id:
        user_latest = user_data_dir(user_id) / "conversation_latest.json"
        if user_latest.exists():
            return ConversationalAdvisorSession.from_dict(json.loads(user_latest.read_text(encoding="utf-8")))
    path = Path(latest_path)
    if not path.exists():
        return None
    return ConversationalAdvisorSession.from_dict(json.loads(path.read_text(encoding="utf-8")))


def continue_conversational_advisor_session(
    session: ConversationalAdvisorSession,
    user_message: str,
    *,
    use_llm: bool = True,
    output_dir: str | Path = DEFAULT_CONVERSATION_DIR,
    progress_callback: AgentProgressCallback | None = None,
) -> ConversationalAdvisorSession:
    """Advance the local advisor session by one user turn."""

    user = AdvisorMessage(role="user", content=user_message)
    messages = [*session.messages, user]
    warnings = list(session.warnings)
    profile_text = session.profile_text
    goals = session.goals
    tickers = list(session.tickers)
    stage: AdvisorStage = session.stage
    artifacts = list(session.artifacts)
    latest_result = dict(session.latest_result)
    stock_selection = dict(session.stock_selection)
    agent_events = list(session.agent_events)
    is_portfolio_update = False
    update_since = ""
    turn_context = _build_turn_context(user_message, messages)

    if stage == "profile":
        profile_start = time.perf_counter()
        _emit_event(
            agent_events,
            _agent_event("客户画像Agent", "running", "正在解析你的目标、期限和风险约束", input_summary=user_message),
            progress_callback,
        )
        profile_text = _extract_profile_text(
            user_message,
            use_llm=use_llm,
            warnings=warnings,
            chat_history=messages,
            agent_events=agent_events,
            turn_context=turn_context,
            progress_callback=progress_callback,
        )
        _emit_event(
            agent_events,
            _agent_event(
                "客户画像Agent",
                "success",
                "解析并保存用户投资目标与风险约束",
                input_summary=user_message,
                output_summary=profile_text,
                elapsed_seconds=time.perf_counter() - profile_start,
                metadata={"stage": "profile"},
            ),
            progress_callback,
        )
        goals = profile_text
        account = load_user_account(session.user_id)
        if account is not None:
            save_user_account(
                account.display_name,
                investment_role=account.investment_role,
                goals=goals,
                user_id=account.user_id,
            )
        parse_client_profile_questionnaire(
            risk_level=_infer_risk_level(profile_text),
            horizon=_infer_horizon(profile_text),
            liquidity_need="中",
            max_drawdown_tolerance=_infer_max_drawdown(profile_text),
            natural_language_profile=profile_text,
            use_llm=False,
        )
        assistant_reply = _llm_chat_reply(
            "客户画像已保存。请自然地引导用户给出股票池或股票代码，不要像表单一样逐项盘问。",
            profile_text=profile_text,
            latest_result={},
            use_llm=use_llm,
            warnings=warnings,
            fallback="我记下了。接下来告诉我你想从哪些股票开始，比如“沪深300”“中证500”，或者直接输入几个股票代码。",
            agent_events=agent_events,
            chat_history=messages,
            turn_context=turn_context,
            progress_callback=progress_callback,
        )
        stage = "universe"
    elif stage == "universe":
        selection_start = time.perf_counter()
        _emit_event(
            agent_events,
            _agent_event("选股Agent", "running", "正在识别股票池或代码，并控制分析数量", input_summary=user_message),
            progress_callback,
        )
        selection = _select_theme_stocks_if_requested(turn_context, warnings=warnings) or _select_stocks_with_agent(
            user_message,
            warnings=warnings,
        )
        tickers = list(selection["tickers"])
        stock_selection = selection
        _emit_event(
            agent_events,
            _agent_event(
                "选股Agent",
                "success",
                "识别用户股票池并主动控制到可分析数量",
                input_summary=user_message,
                output_summary=f"{selection.get('source')} -> {', '.join(tickers)}",
                elapsed_seconds=time.perf_counter() - selection_start,
                metadata={
                    "requested_count": selection.get("requested_count"),
                    "selected_count": selection.get("selected_count"),
                    "reason": selection.get("reason"),
                },
            ),
            progress_callback,
        )
        progress_note = (
            f"我会把候选范围控制在 {len(tickers)} 只左右，先查新闻和财务线索，再给出纸面组合建议。"
        )
        portfolio_payload, new_artifacts, new_warnings, new_events = _run_conversational_portfolio(
            profile_text=session.profile_text or profile_text,
            tickers=tickers,
            market=session.market,
            use_llm=use_llm,
            output_dir=user_data_dir(session.user_id) / "conversations" / session.session_id,
            progress_callback=progress_callback,
        )
        agent_events.extend(_new_events_only(agent_events, new_events))
        artifacts.extend(new_artifacts)
        warnings.extend(new_warnings)
        latest_result = portfolio_payload
        latest_result["stock_selection"] = stock_selection
        _save_user_portfolio_snapshot(session.user_id, latest_result, artifacts)
        assistant_reply = _build_completed_reply(
            portfolio_payload,
            artifacts,
            use_llm=use_llm,
            warnings=warnings,
            context_note=progress_note,
            agent_events=agent_events,
            chat_history=messages,
            turn_context=turn_context,
            progress_callback=progress_callback,
        )
        stage = "completed"
    else:
        intent = _classify_followup(turn_context)
        if intent == "restart":
            account = LocalUserAccount(
                user_id=session.user_id,
                display_name=session.user_id,
                investment_role=session.investment_role,
                goals=session.goals,
            )
            return start_conversational_advisor_session(market=session.market, user=account, output_dir=output_dir)
        if intent in {"rerun", "agent_flow", "portfolio_update"}:
            is_portfolio_update = intent == "portfolio_update"
            themed_selection = _select_theme_stocks_if_requested(turn_context, warnings=warnings)
            if is_portfolio_update:
                saved_portfolio = _load_user_portfolio_snapshot(session.user_id)
                if saved_portfolio:
                    latest_result = _merge_saved_portfolio(latest_result, saved_portfolio)
            update_since = _last_portfolio_timestamp(latest_result, artifacts, session)
            if intent == "agent_flow" or themed_selection:
                profile_start = time.perf_counter()
                _emit_event(
                    agent_events,
                    _agent_event("客户画像Agent", "running", "正在固化最新约束，并作为后续Agent输入", input_summary=user_message),
                    progress_callback,
                )
                profile_text = _extract_profile_text(
                    user_message,
                    use_llm=use_llm,
                    warnings=warnings,
                    chat_history=messages,
                    agent_events=agent_events,
                    turn_context=turn_context,
                    progress_callback=progress_callback,
                )
                goals = profile_text
                _emit_event(
                    agent_events,
                    _agent_event(
                        "客户画像Agent",
                        "success",
                        "已固化最新客户画像约束，继续调用多智能体链路",
                        input_summary=user_message,
                        output_summary=profile_text,
                        elapsed_seconds=time.perf_counter() - profile_start,
                        metadata={"stage": "completed", "trigger": "agent_flow"},
                    ),
                    progress_callback,
                )
                account = load_user_account(session.user_id)
                if account is not None:
                    save_user_account(
                        account.display_name,
                        investment_role=account.investment_role,
                        goals=goals,
                        user_id=account.user_id,
                    )
            if themed_selection:
                selection_start = time.perf_counter()
                tickers = list(themed_selection["tickers"])
                stock_selection = themed_selection
                _emit_event(
                    agent_events,
                    _agent_event(
                        "选股Agent",
                        "success",
                        "识别到新的主题口径，已切换主题股票池并控制到可分析数量",
                        input_summary=user_message,
                        output_summary=f"{themed_selection.get('source')} -> {', '.join(tickers)}",
                        elapsed_seconds=time.perf_counter() - selection_start,
                        metadata={"trigger": "theme_switch", "theme": themed_selection.get("theme")},
                    ),
                    progress_callback,
                )
            else:
                tickers = _recover_tickers_for_agent_flow(tickers, latest_result, stock_selection)
            if not tickers:
                selection_start = time.perf_counter()
                _emit_event(
                    agent_events,
                    _agent_event("选股Agent", "running", "没有可复用股票池，正在从当前消息重新识别", input_summary=user_message),
                    progress_callback,
                )
                selection = _select_stocks_with_agent(user_message, warnings=warnings)
                tickers = list(selection["tickers"])
                stock_selection = selection
                _emit_event(
                    agent_events,
                    _agent_event(
                        "选股Agent",
                        "success",
                        "已补齐股票池并控制到可分析数量",
                        input_summary=user_message,
                        output_summary=f"{selection.get('source')} -> {', '.join(tickers)}",
                        elapsed_seconds=time.perf_counter() - selection_start,
                        metadata={"trigger": "agent_flow"},
                    ),
                    progress_callback,
                )
            run_output_dir = user_data_dir(session.user_id) / "conversations" / session.session_id
            if is_portfolio_update:
                run_output_dir = run_output_dir / "updates" / pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            portfolio_payload, new_artifacts, new_warnings, new_events = _run_conversational_portfolio(
                profile_text=profile_text,
                tickers=tickers,
                market=session.market,
                use_llm=use_llm,
                output_dir=run_output_dir,
                update_since=update_since if is_portfolio_update else None,
                progress_callback=progress_callback,
            )
            agent_events.extend(_new_events_only(agent_events, new_events))
            artifacts.extend(new_artifacts)
            warnings.extend(new_warnings)
            latest_result = portfolio_payload
            if is_portfolio_update:
                latest_result["portfolio_update"] = {
                    "updated_from": update_since,
                    "updated_at": pd.Timestamp.now().isoformat(),
                    "source": "account_saved_portfolio",
                    "tickers": tickers,
                }
            _save_user_portfolio_snapshot(session.user_id, latest_result, artifacts)
            assistant_reply = _build_completed_reply(
                portfolio_payload,
                artifacts,
                use_llm=use_llm,
                warnings=warnings,
                context_note=_portfolio_run_context_note(intent, update_since),
                agent_events=agent_events,
                chat_history=messages,
                turn_context=turn_context,
                progress_callback=progress_callback,
            )
        else:
            assistant_reply = _answer_followup(
                user_message,
                latest_result,
                use_llm=use_llm,
                warnings=warnings,
                agent_events=agent_events,
                chat_history=messages,
                turn_context=turn_context,
                progress_callback=progress_callback,
            )

    updated = ConversationalAdvisorSession(
        session_id=session.session_id,
        stage=stage,
        messages=[*messages, AdvisorMessage(role="assistant", content=assistant_reply)],
        user_id=session.user_id,
        investment_role=session.investment_role,
        goals=goals,
        profile_text=profile_text,
        tickers=tickers,
        market=session.market,
        artifacts=artifacts,
        latest_result=latest_result,
        stock_selection=stock_selection,
        agent_events=agent_events,
        warnings=warnings,
    )
    _save_session(updated, output_dir=output_dir)
    SQLiteStore().save_state_record("conversational_advisor", updated.session_id, updated.to_dict())
    return updated


def _run_conversational_portfolio(
    *,
    profile_text: str,
    tickers: list[str],
    market: str,
    use_llm: bool,
    output_dir: Path,
    update_since: str | None = None,
    progress_callback: AgentProgressCallback | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str], list[AgentCallEvent]]:
    warnings: list[str] = []
    events: list[AgentCallEvent] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    universe = [
        StockCandidate(ticker=ticker, name=ticker, sector="用户选择")
        for ticker in tickers
    ]
    result = ChinaInvestmentAssistant(output_dir=output_dir).run
    portfolio_start = time.perf_counter()
    _emit_event(
        events,
        _agent_event(
            "组合经理Agent",
            "running",
            "正在生成目标权重、偏离度和纸面调仓建议",
            input_summary=f"{len(universe)}只候选股：{', '.join(tickers)}",
        ),
        progress_callback,
    )
    try:
        investment = _run_async_investment(
            result,
            profile_text=profile_text,
            universe=universe,
            max_candidates=len(universe),
            selected_count=min(10, max(1, len(universe))),
            use_llm_report=use_llm,
        )
        _emit_event(
            events,
            _agent_event(
                "组合经理Agent",
                "success",
                "融合画像与候选股票，生成纸面目标权重和调仓建议",
                input_summary=f"{len(universe)}只候选股：{', '.join(tickers)}",
                output_summary=f"{len(getattr(investment, 'target_weights', {}) or {})}个目标持仓",
                elapsed_seconds=time.perf_counter() - portfolio_start,
                metadata={"ticker_count": len(tickers)},
            ),
            progress_callback,
        )
    except Exception as exc:
        _emit_event(
            events,
            _agent_event(
                "组合经理Agent",
                "failed",
                "组合生成失败",
                input_summary=f"{len(universe)}只候选股：{', '.join(tickers)}",
                elapsed_seconds=time.perf_counter() - portfolio_start,
                error=f"{type(exc).__name__}: {exc}",
            ),
            progress_callback,
        )
        raise

    search_results: list[SearchAgentResult] = []
    for ticker in tickers[: min(5, len(tickers))]:
        search_start = time.perf_counter()
        search_path = output_dir / f"search_{ticker.replace('.', '_')}.json"
        search_query = _portfolio_update_search_query(ticker, update_since)
        _emit_event(
            events,
            _agent_event(
                "搜索Agent",
                "running",
                f"正在检索 {ticker} 的新闻和公开资料",
                input_summary=search_query,
                metadata={"ticker": ticker, "update_since": update_since},
            ),
            progress_callback,
        )
        try:
            if search_path.exists():
                result_item = SearchAgentResult.from_dict(json.loads(search_path.read_text(encoding="utf-8")))
                cache_note = "（使用本地缓存）"
            else:
                result_item = run_search_agent(
                    search_query,
                    ticker=ticker,
                    market=market,
                    max_results=2,
                    use_llm=use_llm,
                    output_path=search_path,
                )
                cache_note = ""
            search_results.append(result_item)
            _emit_event(
                events,
                _agent_event(
                    "搜索Agent",
                    "success",
                    f"检索 {ticker} 的近期新闻和公开资料{cache_note}",
                    input_summary=search_query,
                    output_summary=result_item.answer[:260],
                    elapsed_seconds=time.perf_counter() - search_start,
                    metadata={"ticker": ticker, "source_count": len(result_item.sources), "update_since": update_since},
                ),
                progress_callback,
            )
        except Exception as exc:
            warning = f"{ticker} 搜索Agent失败，已跳过该股票新闻证据：{type(exc).__name__}: {exc}"
            warnings.append(warning)
            _emit_event(
                events,
                _agent_event(
                    "搜索Agent",
                    "failed",
                    f"检索 {ticker} 的近期新闻和公开资料失败",
                    input_summary=ticker,
                    elapsed_seconds=time.perf_counter() - search_start,
                    error=warning,
                    metadata={"ticker": ticker},
                ),
                progress_callback,
            )

    for item in search_results:
        warnings.extend(item.warnings)

    investment_payload = _run_rebalance_agent_governance(
        investment.to_dict(),
        search_results,
        events=events,
        progress_callback=progress_callback,
    )

    artifact_start = time.perf_counter()
    _emit_event(
        events,
        _agent_event("报告Agent", "running", "正在保存本地文件并整理纸面报告", input_summary=str(output_dir)),
        progress_callback,
    )
    news_csv = _write_news_csv(search_results, output_dir / "conversation_news.csv")
    report_pdf = _write_pdf_report(profile_text, investment_payload, search_results, output_dir / "conversation_report.pdf")
    investment_json = output_dir / "conversation_portfolio.json"
    target_weights_csv = _write_weights_csv(investment_payload, output_dir / "target_weights.csv")
    trade_orders_csv = _write_trade_orders_csv(investment_payload, output_dir / "paper_trade_orders.csv")
    investment_json.write_text(json.dumps(investment_payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _emit_event(
        events,
        _agent_event(
            "报告Agent",
            "success",
            "整理新闻 CSV、目标权重 CSV、纸面订单 CSV、组合 JSON 和投顾 PDF 报告",
            input_summary=f"{len(search_results)}个搜索结果",
            output_summary=f"{news_csv.name} / {target_weights_csv.name} / {trade_orders_csv.name} / {investment_json.name} / {report_pdf.name}",
            elapsed_seconds=time.perf_counter() - artifact_start,
            metadata={"artifact_count": 5, "output_dir": str(output_dir)},
        ),
        progress_callback,
    )
    artifacts = [
        _artifact("投资组合 JSON", "json", investment_json, "对话式投顾"),
        _artifact("目标权重 CSV", "csv", target_weights_csv, "组合经理Agent"),
        _artifact("纸面调仓 CSV", "csv", trade_orders_csv, "组合经理Agent"),
        _artifact("新闻证据 CSV", "csv", news_csv, "搜索Agent"),
        _artifact("投顾报告 PDF", "pdf", report_pdf, "对话式投顾"),
    ]
    return investment_payload, artifacts, warnings, events


def _save_user_portfolio_snapshot(user_id: str, portfolio_payload: dict[str, Any], artifacts: list[dict[str, Any]]) -> None:
    root = user_data_dir(user_id)
    root.mkdir(parents=True, exist_ok=True)
    (root / "portfolio_latest.json").write_text(
        json.dumps(portfolio_payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    target_weights = portfolio_payload.get("target_weights") or {}
    if target_weights:
        pd.DataFrame(
            [{"ticker": ticker, "target_weight": weight} for ticker, weight in target_weights.items()]
        ).to_csv(root / "target_weights_latest.csv", index=False, encoding="utf-8-sig")
    trade_orders = portfolio_payload.get("trade_orders") or []
    pd.DataFrame(trade_orders).to_csv(root / "paper_trade_orders_latest.csv", index=False, encoding="utf-8-sig")
    (root / "portfolio_artifacts_latest.json").write_text(
        json.dumps(artifacts, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _load_user_portfolio_snapshot(user_id: str) -> dict[str, Any]:
    path = user_data_dir(user_id) / "portfolio_latest.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _merge_saved_portfolio(current: dict[str, Any], saved: dict[str, Any]) -> dict[str, Any]:
    merged = dict(saved)
    merged.update({key: value for key, value in current.items() if value not in (None, "", [], {})})
    if not merged.get("target_weights") and saved.get("target_weights"):
        merged["target_weights"] = saved["target_weights"]
    if not merged.get("selected") and saved.get("selected"):
        merged["selected"] = saved["selected"]
    return merged


def _run_async_investment(run_callable: Any, **kwargs: Any) -> Any:
    import asyncio

    return asyncio.run(run_callable(**kwargs))


def _portfolio_run_context_note(intent: str, update_since: str) -> str:
    if intent == "portfolio_update":
        since_text = update_since or "上次生成"
        return f"我已经读取账户中保存的上次投资组合，并自动检索 {since_text} 以来的新闻后更新组合。"
    if intent == "agent_flow":
        return "我已经按你的继续指令实际调用了客户画像、基本面、情绪、技术面、风控和组合经理等Agent链路。"
    return "我根据当前记忆重新生成了组合。"


def _portfolio_update_search_query(ticker: str, update_since: str | None) -> str:
    base = f"{ticker} 最近业绩 分红 监管 风险 新闻"
    if not update_since:
        return base
    return f"{ticker} 自 {update_since} 以来 最新公告 新闻 业绩 分红 监管 风险 调仓影响"


def _last_portfolio_timestamp(
    latest_result: dict[str, Any],
    artifacts: list[dict[str, Any]],
    session: ConversationalAdvisorSession,
) -> str:
    candidates = [
        latest_result.get("generated_at"),
        latest_result.get("portfolio_update", {}).get("updated_at") if isinstance(latest_result.get("portfolio_update"), dict) else None,
        session.updated_at,
    ]
    for artifact in reversed(artifacts):
        generated_at = artifact.get("generated_at") if isinstance(artifact, dict) else None
        if generated_at:
            candidates.insert(0, generated_at)
    for candidate in candidates:
        if candidate:
            return str(candidate)
    return ""


def _emit_event(
    events: list[AgentCallEvent],
    event: AgentCallEvent,
    callback: AgentProgressCallback | None,
) -> None:
    events.append(event)
    if callback is not None:
        callback(event)


def _new_events_only(existing: list[AgentCallEvent], incoming: list[AgentCallEvent]) -> list[AgentCallEvent]:
    existing_ids = {(event.agent, event.status, event.created_at, event.summary) for event in existing}
    filtered: list[AgentCallEvent] = []
    running_streams_with_success = {
        str(event.metadata.get("stream_id"))
        for event in incoming
        if event.status == "success" and event.metadata.get("stream_id")
    }
    seen_streams: set[str] = set()
    for event in incoming:
        stream_id = str(event.metadata.get("stream_id") or "")
        if stream_id:
            if event.status == "running" and stream_id in running_streams_with_success:
                continue
            if stream_id in seen_streams and event.status == "running":
                continue
            seen_streams.add(stream_id)
        if (event.agent, event.status, event.created_at, event.summary) not in existing_ids:
            filtered.append(event)
    return filtered


def _run_rebalance_agent_governance(
    investment: dict[str, Any],
    search_results: list[SearchAgentResult],
    *,
    events: list[AgentCallEvent],
    progress_callback: AgentProgressCallback | None,
) -> dict[str, Any]:
    """Enforce the planned multi-agent chain before any rebalance output leaves the system."""

    payload = copy.deepcopy(investment)
    selected = payload.get("selected") or []
    target_weights = payload.get("target_weights") or {}
    trade_orders = payload.get("trade_orders") or []
    risk_gate = payload.get("risk_gate") or {}

    governance: list[dict[str, Any]] = []
    checks = [
        (
            "客户画像Agent",
            "客户风险、期限和目标已进入组合约束",
            bool(payload.get("profile_text")),
            payload.get("profile_text", "")[:260],
        ),
        (
            "基本面分析师",
            "已用候选股票评分/财务线索形成基本面输入；深度 DCF 不足时标记为简化版",
            bool(selected),
            _selected_summary(selected),
        ),
        (
            "市场情绪分析师",
            "已读取搜索Agent新闻摘要和降级警告，作为情绪/事件风险输入",
            bool(search_results),
            _sentiment_summary(search_results),
        ),
        (
            "技术面分析师",
            "已用 RSI、趋势、均线、动量信号过滤买卖时机",
            bool(selected and all("rsi_14" in item for item in selected)),
            _technical_summary(selected),
        ),
        (
            "风险管理总监",
            "已通过硬编码风险断路器；若风控拒绝则订单只能作为研究记录",
            _risk_gate_allows_orders(risk_gate, trade_orders),
            _risk_summary(risk_gate, trade_orders),
        ),
        (
            "投资组合经理",
            "仅在整合画像、基本面、情绪、技术面和风控后输出目标权重",
            bool(target_weights),
            _target_weight_summary(target_weights),
        ),
    ]

    for agent, summary, passed, detail in checks:
        status = "success" if passed else "failed"
        governance.append(
            {
                "agent": agent,
                "passed": passed,
                "summary": summary,
                "detail": detail,
            }
        )
        _emit_event(
            events,
            _agent_event(
                agent,
                status,
                summary,
                output_summary=detail,
                metadata={"governance_required": True},
            ),
            progress_callback,
        )

    all_passed = all(item["passed"] for item in governance)
    payload["agent_governance"] = {
        "required_agents": [item["agent"] for item in governance],
        "all_passed": all_passed,
        "checks": governance,
        "policy": "真正调整投资组合前必须经过客户画像、基本面、情绪、技术面、风险管理总监和投资组合经理；LLM 不得绕过硬编码风控。",
    }

    if not all_passed:
        payload["trade_orders"] = []
        payload.setdefault("warnings", []).append(
            "多智能体治理链路未全部通过，本轮不输出可执行调仓订单，仅保留研究记录和目标仓位参考。"
        )
    else:
        for order in payload.get("trade_orders") or []:
            order["agent_governance_passed"] = True
            order["required_agent_chain"] = "客户画像Agent -> 基本面分析师 -> 市场情绪分析师 -> 技术面分析师 -> 风险管理总监 -> 投资组合经理"
    return payload


def _selected_summary(selected: list[dict[str, Any]]) -> str:
    if not selected:
        return "没有候选股票评分结果。"
    top = selected[:5]
    return "；".join(
        f"{item.get('ticker')} score={float(item.get('score') or 0.0):.3f}"
        for item in top
    )


def _sentiment_summary(search_results: list[SearchAgentResult]) -> str:
    if not search_results:
        return "没有可用新闻/搜索结果，情绪分析未通过。"
    parts = []
    for result in search_results[:5]:
        warning_count = len(result.warnings)
        parts.append(f"{result.ticker or 'general'} provider={result.provider}, sources={len(result.sources)}, warnings={warning_count}")
    return "；".join(parts)


def _technical_summary(selected: list[dict[str, Any]]) -> str:
    if not selected:
        return "没有技术面输入。"
    parts = []
    for item in selected[:5]:
        parts.append(
            f"{item.get('ticker')} RSI={float(item.get('rsi_14') or 0.0):.1f}, trend={item.get('trend_label')}"
        )
    return "；".join(parts)


def _risk_gate_allows_orders(risk_gate: dict[str, Any], trade_orders: list[dict[str, Any]]) -> bool:
    if not trade_orders:
        return True
    decision = str(risk_gate.get("decision") or "").lower()
    return decision in {"approved", "not_run"}


def _risk_summary(risk_gate: dict[str, Any], trade_orders: list[dict[str, Any]]) -> str:
    if not trade_orders:
        return "本轮没有需要执行的纸面订单，风控门无需放行交易。"
    if not risk_gate:
        return "存在纸面订单但没有风险断路器报告，风控未通过。"
    return (
        f"decision={risk_gate.get('decision')}, "
        f"VaR={risk_gate.get('portfolio_var')}, "
        f"max_drawdown={risk_gate.get('max_drawdown')}"
    )


def _target_weight_summary(target_weights: dict[str, Any]) -> str:
    if not target_weights:
        return "没有目标权重。"
    top = sorted(target_weights.items(), key=lambda item: float(item[1] or 0.0), reverse=True)[:6]
    return "；".join(f"{ticker}={float(weight or 0.0):.1%}" for ticker, weight in top)


def _agent_event(
    agent: str,
    status: Literal["pending", "running", "success", "failed", "skipped"],
    summary: str,
    *,
    input_summary: str = "",
    output_summary: str = "",
    elapsed_seconds: float = 0.0,
    error: str = "",
    metadata: dict[str, Any] | None = None,
) -> AgentCallEvent:
    return AgentCallEvent(
        agent=agent,
        status=status,
        summary=summary,
        input_summary=_clip(input_summary),
        output_summary=_clip(output_summary),
        elapsed_seconds=max(0.0, float(elapsed_seconds)),
        error=_clip(error, limit=500),
        metadata=metadata or {},
    )


def _clip(value: Any, *, limit: int = 360) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def _stream_openai_chat(
    *,
    prompt: str,
    fallback: str,
    feature: str,
    max_tokens: int,
    warnings: list[str],
    progress_callback: AgentProgressCallback | None = None,
    agent_name: str = "对话LLM",
    metadata: dict[str, Any] | None = None,
) -> str:
    cfg = get_llm_config()
    if not cfg.api_key:
        warnings.append("OPENAI_API_KEY 未配置，已使用本地规则回复。")
        return fallback
    stream_id = f"{agent_name}:{feature}:{uuid.uuid4().hex[:8]}"
    try:
        from openai import OpenAI

        client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
        stream = client.chat.completions.create(
            model=cfg.chat_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            stream=True,
        )
        chunks: list[str] = []
        last_emit = 0.0
        for chunk in stream:
            choice = chunk.choices[0] if getattr(chunk, "choices", None) else None
            delta = getattr(getattr(choice, "delta", None), "content", None) if choice is not None else None
            if not delta:
                continue
            chunks.append(delta)
            now = time.perf_counter()
            if progress_callback is not None and now - last_emit >= 0.35:
                last_emit = now
                progress_callback(
                    _agent_event(
                        agent_name,
                        "running",
                        "正在流式生成回复",
                        output_summary="".join(chunks)[-360:],
                        metadata={"streaming": True, "stream_id": stream_id, **(metadata or {})},
                    )
                )
        content = "".join(chunks).strip() or fallback
        record = record_llm_usage(
            feature=feature,
            model=cfg.chat_model,
            prompt_text=prompt,
            completion_text=content,
            metadata=metadata or {},
        )
        SQLiteStore().save_llm_usage_record(record)
        if progress_callback is not None:
            progress_callback(
                _agent_event(
                    agent_name,
                    "success",
                    "流式回复生成完成",
                    output_summary=content[-360:],
                    metadata={"streaming": True, "stream_id": stream_id, **(metadata or {})},
                )
            )
        return content
    except Exception as exc:
        warnings.append(f"{feature}失败，已使用本地回复：{type(exc).__name__}: {exc}")
        return fallback


def _extract_profile_text(
    user_message: str,
    *,
    use_llm: bool,
    warnings: list[str],
    chat_history: list[AdvisorMessage] | None = None,
    agent_events: list[AgentCallEvent] | None = None,
    turn_context: TurnContext | None = None,
    progress_callback: AgentProgressCallback | None = None,
) -> str:
    fallback = user_message.strip()
    if not use_llm:
        return fallback

    history_payload = _conversation_history_payload(chat_history or [])
    agent_memory = _agent_memory_payload(agent_events or [])
    prompt = (
        "你是客户画像Agent。请把用户的投资目标整理成一段中文客户画像，包含风险承受能力、投资期限、收益目标、"
        "行业偏好/排除项、最大回撤容忍度。不要编造用户未提到的金额。"
        "如果信息不足，要说明缺口，但仍给出可继续对话的最小画像。\n"
        f"最近聊天记录：{json.dumps(history_payload, ensure_ascii=False, default=str)[:6000]}\n"
        f"过往Agent结论：{json.dumps(agent_memory, ensure_ascii=False, default=str)[:4000]}\n"
        f"本轮解析上下文：{json.dumps(turn_context.to_dict() if turn_context else {}, ensure_ascii=False, default=str)[:4000]}\n"
        f"当前用户输入：{user_message}"
    )
    return _stream_openai_chat(
        prompt=prompt,
        fallback=fallback,
        feature="对话式投顾画像整理",
        max_tokens=500,
        warnings=warnings,
        progress_callback=progress_callback,
        agent_name="客户画像Agent",
        metadata={"stage": "profile"},
    )


def _infer_risk_level(profile_text: str) -> Any:
    if any(word in profile_text for word in ["保守", "低风险", "本金"]):
        return "保守型"
    if any(word in profile_text for word in ["激进", "高风险", "高收益"]):
        return "激进型"
    if any(word in profile_text for word in ["成长", "进取"]):
        return "成长型"
    return "稳健型"


def _infer_horizon(profile_text: str) -> Any:
    if any(word in profile_text for word in ["五年", "5年", "长期"]):
        return "5年以上"
    if any(word in profile_text for word in ["三年", "3年", "中期"]):
        return "3-5年"
    if any(word in profile_text for word in ["一年", "1年", "短期"]):
        return "1-3年"
    return "3-5年"


def _infer_max_drawdown(profile_text: str) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", profile_text)
    if match and any(word in profile_text for word in ["回撤", "亏损", "跌"]):
        return max(0.01, min(0.8, float(match.group(1)) / 100))
    if "保守" in profile_text:
        return 0.08
    if "激进" in profile_text:
        return 0.35
    return 0.15


def _select_stocks_with_agent(
    user_message: str,
    *,
    warnings: list[str],
    max_selected: int = DEFAULT_STOCK_SELECTION_LIMIT,
) -> dict[str, Any]:
    tickers = _extract_tickers(user_message)
    if tickers:
        limit = min(max_selected, MAX_MANUAL_TICKERS)
        selected = tickers[:limit]
        if len(tickers) > limit:
            warnings.append(f"用户输入了 {len(tickers)} 只股票，系统已先控制为 {limit} 只以降低搜索和模型成本。")
        return {
            "source": "用户手动输入",
            "requested_count": len(tickers),
            "selected_count": len(selected),
            "tickers": selected,
            "items": [{"ticker": ticker, "name": ticker, "industry": "用户输入", "board": ""} for ticker in selected],
            "reason": "用户明确输入股票代码，按输入顺序保留并执行数量上限。",
        }

    source = "沪深300"
    if "中证500" in user_message:
        source = "中证500"
    elif "中证1000" in user_message:
        source = "中证1000"
    elif "全A" in user_message or "全a" in user_message:
        source = "全A"

    try:
        pool_size = _source_pool_size(source)
        universe = build_a_share_universe([source], max_count=pool_size, force_refresh=False)
    except Exception as exc:
        warnings.append(f"股票池构建失败，尝试读取最近一次股票池：{type(exc).__name__}: {exc}")
        universe = load_latest_a_share_universe()
    if universe is None or not universe.tickers:
        warnings.append("没有可用股票池，使用内置示例股票。")
        fallback = ["600519.SH", "000858.SZ", "000333.SZ", "300750.SZ", "600036.SH"]
        return {
            "source": "fallback",
            "requested_count": len(fallback),
            "selected_count": len(fallback),
            "tickers": fallback,
            "items": [{"ticker": ticker, "name": ticker, "industry": "fallback", "board": ""} for ticker in fallback],
            "reason": "数据源不可用时使用项目内置核心股票池。",
        }
    return _diversified_stock_selection(universe, source=source, max_selected=max_selected)


def _select_theme_stocks_if_requested(
    turn_context: TurnContext | str,
    *,
    warnings: list[str],
    max_selected: int = DEFAULT_STOCK_SELECTION_LIMIT,
) -> dict[str, Any] | None:
    combined_text = turn_context.expanded_intent if isinstance(turn_context, TurnContext) else str(turn_context)
    text = re.sub(r"\s+", "", combined_text.lower())
    if any(word in text for word in ["军工", "國防", "国防", "航空航天", "航天", "军工电子", "信息化"]):
        return _select_from_static_theme_universe(
            theme="军工电子/信息化",
            items=[
                ("600760.SH", "中航沈飞", "航空装备"),
                ("600893.SH", "航发动力", "航空发动机"),
                ("000768.SZ", "中航西飞", "航空装备"),
                ("002179.SZ", "中航光电", "军工电子"),
                ("002465.SZ", "海格通信", "军工通信"),
                ("000733.SZ", "振华科技", "军工电子"),
                ("600372.SH", "中航机载", "航空电子"),
                ("688122.SH", "西部超导", "军工材料"),
                ("002414.SZ", "高德红外", "红外装备"),
                ("300474.SZ", "景嘉微", "军工芯片"),
                ("600562.SH", "国睿科技", "雷达信息化"),
                ("688002.SH", "睿创微纳", "红外芯片"),
            ],
            max_selected=max_selected,
            warnings=warnings,
        )
    return None


def _select_from_static_theme_universe(
    *,
    theme: str,
    items: list[tuple[str, str, str]],
    max_selected: int,
    warnings: list[str],
) -> dict[str, Any]:
    selected = items[:max_selected]
    if len(items) > max_selected:
        warnings.append(f"已识别主题“{theme}”，候选池 {len(items)} 只，先控制到 {max_selected} 只以节省搜索和模型调用。")
    return {
        "source": f"主题股票池：{theme}",
        "theme": theme,
        "requested_count": len(items),
        "selected_count": len(selected),
        "tickers": [ticker for ticker, _, _ in selected],
        "items": [
            {"ticker": ticker, "name": name, "industry": industry, "board": "", "theme": theme}
            for ticker, name, industry in selected
        ],
        "reason": f"用户表达了“{theme}”主题切换信号，系统不复用旧组合，改用主题候选池重新跑多智能体闭环。",
    }


def _source_pool_size(source: str) -> int:
    if source == "全A":
        return 300
    if source in {"沪深300", "中证500", "中证1000"}:
        return 120
    return 200


def _diversified_stock_selection(universe: Any, *, source: str, max_selected: int) -> dict[str, Any]:
    items = [item.to_dict() for item in universe.items]
    if len(items) <= max_selected:
        return {
            "source": source,
            "requested_count": len(items),
            "selected_count": len(items),
            "tickers": [item["ticker"] for item in items],
            "items": items,
            "reason": "股票池规模未超过上限，全部保留。",
        }

    selected: list[dict[str, Any]] = []
    industry_counts: dict[str, int] = {}
    board_counts: dict[str, int] = {}
    max_per_industry = max(1, max_selected // 4)
    max_per_board = max(3, max_selected // 2)

    for item in items:
        industry = str(item.get("industry") or "未知")
        board = str(item.get("board") or "未知")
        if industry_counts.get(industry, 0) >= max_per_industry:
            continue
        if board_counts.get(board, 0) >= max_per_board:
            continue
        selected.append(item)
        industry_counts[industry] = industry_counts.get(industry, 0) + 1
        board_counts[board] = board_counts.get(board, 0) + 1
        if len(selected) >= max_selected:
            break

    if len(selected) < max_selected:
        selected_tickers = {item["ticker"] for item in selected}
        for item in items:
            if item["ticker"] in selected_tickers:
                continue
            selected.append(item)
            selected_tickers.add(item["ticker"])
            if len(selected) >= max_selected:
                break

    return {
        "source": source,
        "requested_count": len(items),
        "selected_count": len(selected),
        "tickers": [item["ticker"] for item in selected],
        "items": selected,
        "reason": (
            f"原始股票池 {len(items)} 只，系统按行业和板块分散规则控制到 {len(selected)} 只。"
        ),
        "industry_counts": industry_counts,
        "board_counts": board_counts,
    }


def _recover_tickers_for_agent_flow(
    session_tickers: list[str],
    latest_result: dict[str, Any],
    stock_selection: dict[str, Any],
) -> list[str]:
    """Reuse the most recent candidate pool when the user asks to continue."""

    recovered: list[str] = []
    sources: list[Any] = [
        session_tickers,
        stock_selection.get("tickers"),
        (latest_result.get("stock_selection") or {}).get("tickers"),
        list((latest_result.get("target_weights") or {}).keys()),
    ]
    selected = latest_result.get("selected") or []
    if selected:
        sources.append([item.get("ticker") for item in selected if isinstance(item, dict)])
    for source in sources:
        for ticker in source or []:
            normalized = _normalize_ticker(str(ticker))
            if normalized and normalized not in recovered:
                recovered.append(normalized)
            if len(recovered) >= DEFAULT_STOCK_SELECTION_LIMIT:
                return recovered
    return recovered


def _normalize_ticker(value: str) -> str:
    text = value.strip().upper()
    if not text:
        return ""
    match = re.search(r"(?<!\d)(?:SH|SZ|BJ)?(\d{6})(?:\.(?:SH|SZ|BJ))?(?!\d)", text)
    if not match:
        return text if re.fullmatch(r"\d{6}\.(?:SH|SZ|BJ)", text) else ""
    raw = match.group(0)
    code = match.group(1)
    if "." in raw and re.fullmatch(r"\d{6}\.(?:SH|SZ|BJ)", raw):
        return raw
    if raw.startswith(("SH", "SZ", "BJ")):
        return f"{code}.{raw[:2]}"
    if code.startswith(("6", "9")):
        return f"{code}.SH"
    if code.startswith(("4", "8")):
        return f"{code}.BJ"
    return f"{code}.SZ"


def _extract_tickers(text: str) -> list[str]:
    found: list[str] = []
    for match in re.finditer(r"(?<!\d)(?:SH|SZ|BJ)?(\d{6})(?:\.(?:SH|SZ|BJ))?(?!\d)", text.upper()):
        raw = match.group(0)
        code = match.group(1)
        if "." in raw:
            ticker = raw
        elif raw.startswith(("SH", "SZ", "BJ")):
            ticker = f"{code}.{raw[:2]}"
        elif code.startswith(("6", "9")):
            ticker = f"{code}.SH"
        elif code.startswith(("4", "8")):
            ticker = f"{code}.BJ"
        else:
            ticker = f"{code}.SZ"
        if ticker not in found:
            found.append(ticker)
    return found


def _write_news_csv(results: list[SearchAgentResult], path: Path) -> Path:
    rows: list[dict[str, Any]] = []
    for result in results:
        for source in result.sources:
            rows.append(
                {
                    "query": result.query,
                    "ticker": result.ticker,
                    "title": source.title,
                    "url": source.url,
                    "snippet": source.snippet,
                    "provider": source.provider,
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query", "ticker", "title", "url", "snippet", "provider"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_weights_csv(investment: dict[str, Any], path: Path) -> Path:
    rows = []
    current_weights = investment.get("current_weights") or {}
    target_weights = investment.get("target_weights") or {}
    for ticker, target in target_weights.items():
        current = float(current_weights.get(ticker, 0.0) or 0.0)
        target_float = float(target or 0.0)
        rows.append(
            {
                "ticker": ticker,
                "current_weight": current,
                "target_weight": target_float,
                "weight_delta": target_float - current,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ticker", "current_weight", "target_weight", "weight_delta"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_trade_orders_csv(investment: dict[str, Any], path: Path) -> Path:
    rows = []
    for order in investment.get("trade_orders") or []:
        rows.append(
            {
                "ticker": order.get("ticker"),
                "action": order.get("action"),
                "target_weight_delta": order.get("target_weight_delta"),
                "suggested_batches": order.get("suggested_batches"),
                "batch_weight_delta": order.get("batch_weight_delta"),
                "execution_label": order.get("execution_label"),
                "agent_governance_passed": order.get("agent_governance_passed"),
                "required_agent_chain": order.get("required_agent_chain"),
                "reason": order.get("reason"),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "ticker",
                "action",
                "target_weight_delta",
                "suggested_batches",
                "batch_weight_delta",
                "execution_label",
                "agent_governance_passed",
                "required_agent_chain",
                "reason",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_pdf_report(
    profile_text: str,
    investment: dict[str, Any],
    search_results: list[SearchAgentResult],
    path: Path,
) -> Path:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.pdfgen import canvas
    except ImportError:
        return _write_text_pdf_fallback(profile_text, investment, search_results, path)

    path.parent.mkdir(parents=True, exist_ok=True)
    font_name = _register_chinese_font(pdfmetrics, TTFont)
    c = canvas.Canvas(str(path), pagesize=A4)
    width, height = A4
    y = height - 48
    c.setFont(font_name, 14)
    c.drawString(48, y, "Sakura 对话式投顾报告")
    y -= 30
    for line in _report_lines(profile_text, investment, search_results):
        if y < 48:
            c.showPage()
            c.setFont(font_name, 10)
            y = height - 48
        c.setFont(font_name, 10)
        c.drawString(48, y, line[:110])
        y -= 16
    c.save()
    return path


def _write_text_pdf_fallback(
    profile_text: str,
    investment: dict[str, Any],
    search_results: list[SearchAgentResult],
    path: Path,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(_report_lines(profile_text, investment, search_results))
    path.write_text(content, encoding="utf-8")
    return path


def _register_chinese_font(pdfmetrics: Any, TTFont: Any) -> str:
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                pdfmetrics.registerFont(TTFont("SakuraCN", str(candidate)))
                return "SakuraCN"
            except Exception:
                continue
    return "Helvetica"


def _report_lines(
    profile_text: str,
    investment: dict[str, Any],
    search_results: list[SearchAgentResult],
) -> list[str]:
    lines = ["客户画像：", *textwrap.wrap(profile_text, width=70), "", "组合建议："]
    for ticker, weight in (investment.get("target_weights") or {}).items():
        lines.append(f"- {ticker}: {float(weight):.2%}")
    lines.extend(["", "搜索证据摘要："])
    for result in search_results:
        lines.append(f"- {result.ticker}: {result.answer[:240]}")
    lines.extend(["", "风险提示：本报告为研究和纸面组合建议，不构成自动下单指令。"])
    return lines


def _build_completed_reply(
    portfolio_payload: dict[str, Any],
    artifacts: list[dict[str, Any]],
    *,
    use_llm: bool,
    warnings: list[str],
    context_note: str = "",
    agent_events: list[AgentCallEvent] | None = None,
    chat_history: list[AdvisorMessage] | None = None,
    turn_context: TurnContext | None = None,
    progress_callback: AgentProgressCallback | None = None,
) -> str:
    orders = len(portfolio_payload.get("trade_orders") or [])
    holdings = len(portfolio_payload.get("target_weights") or {})
    artifact_names = "、".join(item["name"] for item in artifacts[-3:])
    combined_warnings = [*warnings, *(portfolio_payload.get("warnings") or [])]
    warning_note = _warning_digest(combined_warnings)
    allocation_note = _allocation_digest(portfolio_payload, artifacts)
    governance_note = _governance_digest(portfolio_payload)
    agent_conclusions_note = _agent_conclusions_digest(agent_events or [], portfolio_payload)
    fallback = (
        f"{context_note}\n\n我已经完成这一轮分析：组合里有 {holdings} 个目标持仓，"
        f"生成 {orders} 条纸面调整建议，并准备好了 {artifact_names}。"
        f"\n\n{agent_conclusions_note}"
        f"\n\n{governance_note}"
        f"\n\n{allocation_note}"
        "\n\n这些结果只作为研究记录，真实交易前还需要你确认风险和仓位。"
        f"{warning_note}"
    ).strip()
    return _llm_chat_reply(
        (
            "请以自然、克制的投顾口吻向用户说明本轮组合已经生成。必须分角色展示各Agent的分析结论，"
            "至少包含客户画像、选股/基本面、搜索/市场情绪、技术面、风险管理、投资组合经理、报告保存。"
            "可以展示Agent结论摘要和是否通过，不要输出原始调试日志，不要鼓励直接下单。"
        ),
        profile_text=str(portfolio_payload.get("profile_text") or ""),
        latest_result={
            "target_weights": portfolio_payload.get("target_weights"),
            "trade_order_count": orders,
            "artifact_names": artifact_names,
            "artifact_paths": [item.get("path") for item in artifacts[-5:]],
            "current_weights": portfolio_payload.get("current_weights"),
            "trade_orders": portfolio_payload.get("trade_orders"),
            "agent_governance": portfolio_payload.get("agent_governance"),
            "agent_conclusions_note": agent_conclusions_note,
            "governance_note": governance_note,
            "allocation_note": allocation_note,
            "warnings": combined_warnings,
            "context_note": context_note,
            "agent_events": [event.to_dict() for event in (agent_events or [])[-20:]],
        },
        use_llm=use_llm,
        warnings=warnings,
        fallback=fallback,
        agent_events=agent_events,
        chat_history=chat_history,
        turn_context=turn_context,
        progress_callback=progress_callback,
    )


def _warning_digest(warnings: list[str]) -> str:
    """Return a concise user-facing anomaly note for the chat answer."""

    unique: list[str] = []
    for warning in warnings:
        clean = str(warning).strip()
        if clean and clean not in unique:
            unique.append(clean)
    if not unique:
        return ""
    summary = "；".join(unique[:3])
    return f"\n\n另外我注意到后台有一些异常或降级：{summary}。我已经采用可用数据继续分析，但结论需要更谨慎。"


def _agent_conclusions_digest(events: list[AgentCallEvent], portfolio_payload: dict[str, Any]) -> str:
    """Build a user-facing digest of the latest agent conclusions."""

    conclusions = _agent_memory_payload(events, limit=24)
    if not conclusions and not portfolio_payload.get("agent_governance"):
        return "各Agent分析结论：本轮尚未记录可展示的Agent结论。"

    alias_groups = [
        ("客户画像Agent", ("客户画像Agent",)),
        ("选股/基本面", ("选股Agent", "基本面分析师")),
        ("搜索/市场情绪", ("搜索Agent", "市场情绪分析师")),
        ("技术面分析师", ("技术面分析师",)),
        ("风险管理总监", ("风险管理总监",)),
        ("投资组合经理", ("组合经理Agent", "投资组合经理")),
        ("报告Agent", ("报告Agent",)),
    ]
    by_agent = {str(item.get("agent")): item for item in conclusions}
    governance_checks = {
        str(item.get("agent")): item
        for item in (portfolio_payload.get("agent_governance") or {}).get("checks", [])
        if isinstance(item, dict)
    }
    lines = ["各Agent分析结论："]
    for label, names in alias_groups:
        item = next((by_agent[name] for name in names if name in by_agent), None)
        check = next((governance_checks[name] for name in names if name in governance_checks), None)
        if item is None and check is None:
            continue
        status = str((item or {}).get("status") or ("success" if (check or {}).get("passed") else "failed"))
        status_text = "通过" if status == "success" else "未通过" if status == "failed" else status
        summary = str((item or {}).get("summary") or (check or {}).get("detail") or (check or {}).get("summary") or "")
        if len(summary) > 180:
            summary = summary[:180].rstrip() + "..."
        lines.append(f"- {label}：{status_text}。{summary}")
    if len(lines) == 1:
        lines.append("- 本轮Agent结论已写入本地结果文件，但没有可压缩展示的摘要。")
    return "\n".join(lines)


def _allocation_digest(portfolio_payload: dict[str, Any], artifacts: list[dict[str, Any]]) -> str:
    target_weights = portfolio_payload.get("target_weights") or {}
    current_weights = portfolio_payload.get("current_weights") or {}
    trade_orders = portfolio_payload.get("trade_orders") or []
    if not target_weights:
        return "本轮没有生成可用目标权重，请先补充股票池或检查数据源。"

    top_weights = sorted(target_weights.items(), key=lambda item: float(item[1] or 0.0), reverse=True)[:8]
    weight_lines = [
        f"- {ticker}: 当前 {float(current_weights.get(ticker, 0.0) or 0.0):.1%} -> 目标 {float(weight):.1%}"
        for ticker, weight in top_weights
    ]
    engine = str(portfolio_payload.get("portfolio_engine") or "unknown")
    bl_summary = portfolio_payload.get("black_litterman_summary") or {}
    if engine == "black_litterman":
        engine_text = "组合经理Agent：已调用完整 Black-Litterman 链路 market_prior.py -> views.py -> optimizer.py。"
    else:
        engine_text = f"组合经理Agent：Black-Litterman链路未完成，本轮使用 {engine}，该结果只能作为降级研究记录。"
    diagnostics = portfolio_payload.get("portfolio_diagnostics") or bl_summary.get("diagnostics") or []
    diagnostics_text = "\n".join(f"- {item}" for item in diagnostics[:4])

    if trade_orders:
        order_lines = []
        for order in trade_orders[:8]:
            action = "买入" if order.get("action") == "buy" else "卖出" if order.get("action") == "sell" else str(order.get("action"))
            delta = abs(float(order.get("target_weight_delta") or order.get("batch_weight_delta") or 0.0))
            batches = int(order.get("suggested_batches") or 1)
            per_batch = abs(float(order.get("batch_weight_delta") or (delta / max(1, batches))))
            label = order.get("execution_label") or "需要人工确认"
            order_lines.append(f"- {ticker_name(order)}: {action}总仓位约 {delta:.1%}，建议分 {batches} 批，每批约 {per_batch:.1%}；状态：{label}")
        rebalance_text = "具体调仓建议：\n" + "\n".join(order_lines)
    else:
        rebalance_text = "当前模拟仓位与目标仓位的偏离没有超过阈值，暂不需要生成纸面调仓订单。"

    paths = [str(item.get("path")) for item in artifacts if item.get("path")]
    path_text = "\n".join(f"- {path}" for path in paths[-5:])
    engine_prefix = engine_text + (f"\n优化诊断：\n{diagnostics_text}" if diagnostics_text else "") + "\n\n"
    return (
        engine_prefix
        + "目标仓位如下：\n"
        + "\n".join(weight_lines)
        + "\n\n"
        + rebalance_text
        + "\n\n结果已保存到本地：\n"
        + path_text
    )


def _governance_digest(portfolio_payload: dict[str, Any]) -> str:
    governance = portfolio_payload.get("agent_governance") or {}
    checks = governance.get("checks") or []
    if not checks:
        return "本轮尚未记录完整多智能体治理链路，因此不会把结果描述为可执行调仓。"
    lines = []
    for item in checks:
        status = "通过" if item.get("passed") else "未通过"
        lines.append(f"- {item.get('agent')}: {status}，{item.get('summary')}")
    conclusion = (
        "本轮调仓链路已通过所有必需 agents。"
        if governance.get("all_passed")
        else "本轮多智能体治理链路未全部通过，只能作为研究记录，不应执行调仓。"
    )
    return "调仓前置 Agent 审核：\n" + "\n".join(lines) + f"\n{conclusion}"


def ticker_name(order: dict[str, Any]) -> str:
    ticker = str(order.get("ticker") or "")
    name = str(order.get("name") or "").strip()
    return f"{ticker} {name}".strip()


def _llm_chat_reply(
    instruction: str,
    *,
    profile_text: str,
    latest_result: dict[str, Any],
    use_llm: bool,
    warnings: list[str],
    fallback: str,
    agent_events: list[AgentCallEvent] | None = None,
    chat_history: list[AdvisorMessage] | None = None,
    turn_context: TurnContext | None = None,
    progress_callback: AgentProgressCallback | None = None,
) -> str:
    if not use_llm:
        return fallback
    prompt = _advisor_orchestrator_prompt(
        instruction=instruction,
        profile_text=profile_text,
        latest_result=latest_result,
        agent_events=agent_events or [],
        chat_history=chat_history or [],
        turn_context=turn_context,
    )
    return _stream_openai_chat(
        prompt=prompt,
        fallback=fallback,
        feature="对话式投顾自然回复",
        max_tokens=700,
        warnings=warnings,
        progress_callback=progress_callback,
        agent_name="主对话LLM",
        metadata={"agent_event_count": len(agent_events or [])},
    )


def _classify_followup(context: TurnContext | str, *, chat_history: list[AdvisorMessage] | None = None) -> str:
    message = context.raw_message if isinstance(context, TurnContext) else str(context)
    if any(word in message for word in ["重新开始", "重来", "新建"]):
        return "restart"
    if _looks_like_portfolio_update_request(message):
        return "portfolio_update"
    if any(word in message for word in ["重新生成", "再跑", "更新组合"]):
        return "rerun"
    if _looks_like_agent_flow_request(context, chat_history=chat_history):
        return "agent_flow"
    return "qa"


def _looks_like_portfolio_update_request(message: str) -> bool:
    text = re.sub(r"\s+", "", message.strip().lower())
    if not text:
        return False
    update_words = [
        "更新投资组合",
        "更新组合",
        "刷新投资组合",
        "刷新组合",
        "根据最新新闻更新",
        "按最新新闻更新",
        "重新评估持仓",
        "更新账户组合",
        "更新我的组合",
        "更新仓位",
        "调整仓位",
    ]
    return any(word in text for word in update_words)


def _looks_like_agent_flow_request(context: TurnContext | str, *, chat_history: list[AdvisorMessage] | None = None) -> bool:
    expanded = context.expanded_intent if isinstance(context, TurnContext) else _combined_user_intent_text(str(context), chat_history)
    text = re.sub(r"\s+", "", expanded.lower())
    if not text:
        return False
    if isinstance(context, TurnContext) and _context_reply_accepts_agent_action(context):
        return True
    flow_words = [
        "继续",
        "确认",
        "按流程",
        "流程调用",
        "调用agent",
        "调用agents",
        "调用智能体",
        "选股agent",
        "搜索agent",
        "发起选股",
        "正式发起",
        "进入闭环",
        "调度哪些agent",
        "调度agent",
        "自动调agents",
        "自动调agent",
        "固化",
        "确认固化",
        "开始分析",
        "开始执行",
        "开始跑",
        "开始闭环",
        "闭环",
        "重跑闭环",
        "重新跑闭环",
        "新股票池",
        "候选池",
        "重新走",
        "必须重跑",
        "继续跑",
        "继续推进",
        "生成组合",
        "给出组合",
        "调仓建议",
        "仓位建议",
        "主仓",
        "主题更新",
    ]
    if any(word in text for word in flow_words):
        return True
    return False


def _build_turn_context(message: str, chat_history: list[AdvisorMessage] | None = None) -> TurnContext:
    raw = message.strip()
    recent_messages = _conversation_history_payload(chat_history or [], limit=8)
    normalized = re.sub(r"\s+", "", raw)
    option_match = re.fullmatch(r"(?i)(?:选|选择)?([a-d]|[1-4])(?:口径|方案|档)?", normalized)
    is_short_reply = _is_context_dependent_short_reply(raw)
    referenced_prompt = _recent_context_prompt(chat_history or []) if is_short_reply or option_match else ""
    expanded = raw
    option_value = ""
    if option_match:
        option_value = option_match.group(1).upper()
        option_hint = _resolve_option_hint(option_value, referenced_prompt)
        expanded = f"用户选择了选项 {option_value}。{option_hint}\n上一轮上下文：{referenced_prompt}\n当前回复：{raw}"
    elif is_short_reply and referenced_prompt:
        expanded = f"用户短回复：{raw}\n它是在回应上一轮上下文：{referenced_prompt}"
    return TurnContext(
        raw_message=raw,
        expanded_intent=expanded,
        recent_messages=recent_messages,
        is_short_reply=bool(is_short_reply),
        is_option_reply=bool(option_match),
        option_value=option_value,
        referenced_prompt=referenced_prompt[:3000],
    )


def _context_reply_accepts_agent_action(context: TurnContext) -> bool:
    text = re.sub(r"\s+", "", context.expanded_intent.lower())
    prompt = re.sub(r"\s+", "", context.referenced_prompt.lower())
    if not (context.is_short_reply or context.is_option_reply):
        return False
    action_words = [
        "调用agent",
        "调用agents",
        "选股agent",
        "搜索agent",
        "发起选股",
        "正式发起",
        "开始跑",
        "继续调度",
        "进入闭环",
        "重新跑闭环",
        "继续往下跑",
        "产出可执行组合",
        "agent",
        "调用",
        "发起",
        "启动",
        "生成",
        "产出",
    ]
    subject_words = [
        "股票池",
        "候选池",
        "组合",
        "调仓",
        "风控",
        "闭环",
        "选股",
        "投资",
        "主仓",
        "行业",
        "主题",
        "口径",
        "风险",
        "仓位",
        "军工",
        "信息化",
    ]
    has_action = any(word in text or word in prompt for word in action_words)
    has_subject = any(word in text or word in prompt for word in subject_words)
    if context.is_option_reply and has_subject:
        return True
    return has_action and has_subject


def _is_context_dependent_short_reply(text: str) -> bool:
    normalized = re.sub(r"\s+", "", text.strip())
    if not normalized:
        return False
    if re.fullmatch(r"(?i)(?:选|选择)?[a-d1-4](?:口径|方案|档)?", normalized):
        return True
    return normalized.lower() in {"确认", "继续", "可以", "好的", "好", "ok", "yes", "嗯", "是", "同意", "执行", "开始"}


def _recent_context_prompt(chat_history: list[AdvisorMessage]) -> str:
    snippets: list[str] = []
    for item in reversed(chat_history[:-1]):
        if item.role in {"assistant", "user"} and item.content:
            snippets.append(item.content.strip())
        if len(snippets) >= 4:
            break
    return "\n".join(reversed(snippets))


def _resolve_option_hint(option_value: str, referenced_prompt: str) -> str:
    prompt = referenced_prompt or ""
    upper = option_value.upper()
    patterns = [
        rf"{upper}\s*口径[（(]?([^：:\n）)]*)[）)]?[：:：]?\s*([^\n]+)",
        rf"{upper}\s*[）).、]\s*([^\n]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, prompt, flags=re.IGNORECASE)
        if match:
            return " ".join(part.strip() for part in match.groups() if part and part.strip())
    if upper == "B" and any(word in prompt for word in ["硬件+软件", "软件", "C4ISR", "信息化"]):
        return "B口径：硬件+软件，在A基础上额外纳入指挥控制/C4ISR、军工信息化软件、仿真训练、数据链/信息安全等。"
    if upper == "A" and any(word in prompt for word in ["偏硬件", "雷达", "射频"]):
        return "A口径：偏硬件，雷达/射频微波、军用通信、电子对抗、航电、军用元器件、北斗终端等。"
    return ""


def _combined_user_intent_text(message: str, chat_history: list[AdvisorMessage] | None = None) -> str:
    return _build_turn_context(message, chat_history).expanded_intent


def _advisor_orchestrator_prompt(
    *,
    instruction: str,
    profile_text: str,
    latest_result: dict[str, Any],
    agent_events: list[AgentCallEvent],
    chat_history: list[AdvisorMessage] | None = None,
    turn_context: TurnContext | None = None,
) -> str:
    events_payload = [event.to_dict() for event in agent_events[-24:]]
    history_payload = _conversation_history_payload(chat_history or [])
    agent_memory = _agent_memory_payload(agent_events)
    return (
        "你是 Sakura 主对话LLM，也是多智能体投顾的调度员。你的回答必须尽可能参考已调用 agents 的输出，"
        "而不是只凭自己的常识重新判断。\n"
        "可用/已用角色包括：客户画像Agent、选股Agent、基本面分析师、市场情绪分析师、技术面分析师、"
        "风险管理总监、投资组合经理、搜索Agent、报告Agent。\n"
        "行为规则：\n"
        "1. 先读取 agent_events 和 latest_result，再回答用户。\n"
        "2. 如果已有相关 agent 结果，明确引用其结论，例如“技术面分析师显示...”“风险管理总监已/未通过...”。\n"
        "3. 如果需要新资料或缺少关键证据，主动说明下一步应该调用哪个 agent，而不是假装已有结论。\n"
        "4. 真正调整投资组合前必须经过：客户画像Agent、基本面分析师、市场情绪分析师、技术面分析师、风险管理总监、投资组合经理。\n"
        "5. 如果 agent_governance 不是 all_passed，必须明确说只能作为研究记录，不能执行调仓。\n"
        "6. 当已经生成组合结果时，必须说明目标仓位、买卖仓位差、是否分批、保存到哪些本地文件。\n"
        "7. 当 latest_result 中出现 agent_conclusions_note 或本轮组合已生成时，必须用清晰小节展示各Agent分析结论，"
        "包括客户画像、选股/基本面、搜索/市场情绪、技术面、风险管理、投资组合经理和报告保存，不要只说“已完成”。\n"
        "8. 如果 warnings 里有数据源、RAG、搜索、财报、行情或模型异常，必须解释降级和影响。\n"
        "9. 语气自然，不要表格式盘问；但可以温和引导用户补充目标、股票范围、仓位或风险偏好。\n\n"
        f"任务：{instruction}\n"
        f"客户画像：{profile_text}\n"
        f"本轮上下文解析：{json.dumps(turn_context.to_dict() if turn_context else {}, ensure_ascii=False, default=str)[:5000]}\n"
        f"最近聊天记录：{json.dumps(history_payload, ensure_ascii=False, default=str)[:8000]}\n"
        f"Agent过往结论摘要：{json.dumps(agent_memory, ensure_ascii=False, default=str)[:8000]}\n"
        f"当前结果：{json.dumps(latest_result, ensure_ascii=False, default=str)[:12000]}\n"
        f"Agent调用记录：{json.dumps(events_payload, ensure_ascii=False, default=str)[:10000]}"
    )


def _conversation_history_payload(messages: list[AdvisorMessage], *, limit: int = 14) -> list[dict[str, Any]]:
    """Return recent chat turns in a compact prompt-friendly shape."""

    payload: list[dict[str, Any]] = []
    for message in messages[-limit:]:
        content = str(message.content or "").strip()
        if not content:
            continue
        payload.append(
            {
                "role": message.role,
                "content": content[:1600],
                "created_at": message.created_at,
            }
        )
    return payload


def _agent_memory_payload(events: list[AgentCallEvent], *, limit: int = 18) -> list[dict[str, Any]]:
    """Compress prior agent conclusions so the dialogue model can reuse them."""

    latest_by_agent: dict[str, dict[str, Any]] = {}
    for event in events:
        summary = str(event.output_summary or event.summary or "").strip()
        if not summary and not event.error:
            continue
        latest_by_agent[event.agent] = {
            "agent": event.agent,
            "status": event.status,
            "summary": summary[:1800],
            "error": str(event.error or "")[:800],
            "elapsed_seconds": round(float(event.elapsed_seconds or 0.0), 3),
            "created_at": event.created_at,
            "metadata": _compact_prompt_metadata(event.metadata),
        }
    return list(latest_by_agent.values())[-limit:]


def _compact_prompt_metadata(metadata: dict[str, Any], *, max_items: int = 8) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in list((metadata or {}).items())[:max_items]:
        if isinstance(value, (str, int, float, bool)) or value is None:
            compact[key] = value
        elif isinstance(value, list):
            compact[key] = value[:10]
        elif isinstance(value, dict):
            compact[key] = {str(k): v for k, v in list(value.items())[:8]}
        else:
            compact[key] = str(value)[:500]
    return compact


def _answer_followup(
    message: str,
    latest_result: dict[str, Any],
    *,
    use_llm: bool,
    warnings: list[str],
    agent_events: list[AgentCallEvent] | None = None,
    chat_history: list[AdvisorMessage] | None = None,
    turn_context: TurnContext | None = None,
    progress_callback: AgentProgressCallback | None = None,
) -> str:
    fallback = (
        "我可以继续解释组合、风险和订单。若要重新开始，请输入“重新开始”；若要更新组合，请输入“重新生成”。"
        f"{_warning_digest(warnings + list(latest_result.get('warnings') or []))}"
    )
    if not use_llm or not latest_result:
        return fallback
    prompt = _advisor_orchestrator_prompt(
        instruction=f"回答用户追问：{message}",
        profile_text=str(latest_result.get("profile_text") or ""),
        latest_result=latest_result,
        agent_events=agent_events or [],
        chat_history=chat_history or [],
        turn_context=turn_context,
    )
    return _stream_openai_chat(
        prompt=prompt,
        fallback=fallback,
        feature="对话式投顾追问",
        max_tokens=700,
        warnings=warnings,
        progress_callback=progress_callback,
        agent_name="主对话LLM",
        metadata={"followup": True, "agent_event_count": len(agent_events or [])},
    )


def _artifact(name: str, artifact_type: str, path: Path, source_step: str) -> dict[str, Any]:
    return {
        "name": name,
        "artifact_type": artifact_type,
        "path": str(path),
        "source_step": source_step,
        "generated_at": pd.Timestamp.now().isoformat(),
    }


def _save_session(session: ConversationalAdvisorSession, *, output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(session.to_dict(), ensure_ascii=False, indent=2, default=str)
    (out / f"{session.session_id}.json").write_text(payload, encoding="utf-8")
    user_root = user_data_dir(session.user_id)
    user_conversation_dir = user_root / "conversations"
    user_conversation_dir.mkdir(parents=True, exist_ok=True)
    (user_conversation_dir / f"{session.session_id}.json").write_text(payload, encoding="utf-8")
    (user_root / "conversation_latest.json").write_text(payload, encoding="utf-8")
    DEFAULT_CONVERSATION_LATEST.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_CONVERSATION_LATEST.write_text(payload, encoding="utf-8")
