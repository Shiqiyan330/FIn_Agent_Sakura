"""Streamlit dashboard control center for the A-share robo-advisor."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from fin_agent_sakura.applications.agent_analysis import (
    AgentAnalysisResult,
    AgentNodeEvent,
    SingleTickerAgentAnalysisRunner,
    run_single_ticker_agent_analysis,
)
from fin_agent_sakura.applications.a_share_universe import (
    AShareUniverseResult,
    build_a_share_universe,
    load_latest_a_share_universe,
)
from fin_agent_sakura.applications.backtest_service import (
    BacktestRunReport,
    load_latest_backtest_report,
    run_a_share_backtest,
    save_backtest_news_csv,
)
from fin_agent_sakura.applications.china_investment_assistant import (
    ChinaInvestmentAssistant,
    ChinaInvestmentResult,
    StockCandidate,
)
from fin_agent_sakura.applications.client_profile import (
    ClientProfileResult,
    load_latest_client_profile,
    parse_client_profile_questionnaire,
)
from fin_agent_sakura.applications.conversational_advisor import (
    ConversationalAdvisorSession,
    continue_conversational_advisor_session,
    load_latest_conversational_advisor_session,
    start_conversational_advisor_session,
)
from fin_agent_sakura.applications.data_source_health import (
    DataSourceHealthReport,
    load_latest_data_source_health_report,
    run_a_share_data_source_health_check,
)
from fin_agent_sakura.applications.full_workflow import (
    InvestmentArtifact,
    InvestmentRun,
    list_investment_run_artifacts,
    load_latest_investment_run,
    run_full_advisory_workflow,
)
from fin_agent_sakura.applications.monitor_schedule import (
    DailyMonitorCheckResult,
    load_daily_monitor_schedule,
    load_latest_daily_monitor_result,
    run_daily_monitor_check_once,
    save_daily_monitor_schedule,
)
from fin_agent_sakura.applications.rag_service import (
    FinancialContextAnswer,
    FinancialReportIndexInfo,
    ask_financial_report,
    delete_indexed_financial_report,
    ingest_uploaded_financial_report,
    list_indexed_financial_reports,
    save_uploaded_report,
)
from fin_agent_sakura.applications.rebalance_log import load_rebalance_event_log
from fin_agent_sakura.applications.risk_gate import (
    evaluate_paper_orders_risk,
    load_latest_risk_gate_report,
)
from fin_agent_sakura.applications.search_agent import (
    SearchAgentResult,
    load_latest_search_agent_result,
    run_search_agent,
)
from fin_agent_sakura.applications.smoke_tests import (
    SmokeTestReport,
    load_latest_smoke_test_report,
    run_gui_smoke_tests,
)
from fin_agent_sakura.applications.technical_analysis_service import (
    TechnicalAnalysisReport,
    load_latest_technical_analysis_report,
    run_single_ticker_technical_analysis,
)
from fin_agent_sakura.applications.user_accounts import (
    LocalUserAccount,
    get_or_create_active_user,
    list_user_accounts,
    save_user_account,
    set_active_user,
)
from fin_agent_sakura.config import get_llm_config, get_tushare_config
from fin_agent_sakura.portfolio import (
    BlackLittermanViews,
    build_absolute_views_from_llm,
    build_black_litterman_model_with_idzorek,
    build_idzorek_omega,
    build_market_equilibrium_prior,
    optimize_bl_portfolio_weights,
    parse_client_constraints_from_text,
)
from fin_agent_sakura.storage import PositionMemory
from fin_agent_sakura.storage import SQLiteStore, load_llm_usage, record_llm_usage, summarize_llm_usage


HealthStatus = Literal["normal", "warning", "error", "unknown"]


@dataclass(frozen=True, slots=True)
class HealthCheck:
    name: str
    status: HealthStatus
    detail: str


def main() -> None:
    st.set_page_config(page_title="Sakura 对话式投顾", layout="wide")
    st.title("Sakura 对话式投顾")
    st.caption("你只需要对话。画像、选股、搜索、组合、报告和异常解释会由投顾助手在后台处理。")
    _render_conversational_advisor_page()


def _render_home(assistant: ChinaInvestmentAssistant, position_memory: PositionMemory) -> None:
    st.subheader("系统健康度")
    checks = _build_passive_health_checks(position_memory)
    cols = st.columns(min(4, len(checks)))
    for index, check in enumerate(checks):
        col = cols[index % len(cols)]
        col.metric(check.name, _status_label(check.status))
        col.caption(check.detail)

    st.subheader("计划书任务状态灯")
    _render_plan_status_lights()
    _render_cost_control_summary()

    st.divider()
    st.subheader("完全按计划书运行一次")
    latest_profile = load_latest_client_profile()
    default_profile = (
        latest_profile.natural_language_profile
        if latest_profile is not None and latest_profile.natural_language_profile
        else "我是保守型投资者，期望跑赢通胀即可"
    )
    run_col_a, run_col_b, run_col_c, run_col_d = st.columns([1.4, 0.7, 0.7, 0.7])
    one_click_profile = run_col_a.text_input("本次客户画像", value=default_profile)
    one_click_max = run_col_b.number_input("候选池数量", min_value=5, max_value=100, value=30, step=5)
    one_click_selected = run_col_c.number_input("持仓数量", min_value=3, max_value=15, value=10, step=1)
    include_backtest = run_col_d.checkbox("包含回测", value=False)
    mode_col_a, mode_col_b = st.columns([0.35, 0.65])
    expert_mode = mode_col_a.toggle("专家模式", value=False)
    mode_col_b.caption("新手模式显示中文结论；专家模式额外展示步骤 JSON 与产物路径。")

    if st.button("一键运行全流程", type="primary", use_container_width=True):
        with st.status("正在按计划书运行全流程...", expanded=True) as status:
            st.write("1. 检查 LLM 与 A 股数据源")
            st.write("2. 准备股票池与本地仓位")
            st.write("3. 生成投资方案、纸面订单、风险报告和漂移日志")
            st.write("4. 可选生成回测快照，并汇总结果中心产物")
            run = run_full_advisory_workflow(
                profile_text=one_click_profile,
                max_candidates=int(one_click_max),
                selected_count=int(one_click_selected),
                include_backtest=include_backtest,
                expert_mode=expert_mode,
            )
            status.update(label="全流程运行完成", state="complete" if run.status == "success" else "error")
        _render_investment_run(run, expert_mode=expert_mode)
    else:
        latest_run = load_latest_investment_run()
        if latest_run is not None:
            with st.expander("最近一次全流程运行", expanded=False):
                _render_investment_run(latest_run, expert_mode=latest_run.expert_mode, allow_expanders=False)

    latest = assistant.load_latest_result()
    left, right = st.columns([1, 1])
    with left:
        st.subheader("最近一次投资方案")
        if latest is None:
            st.info("还没有生成过投资方案。")
        else:
            st.write(f"生成时间：{latest.generated_at}")
            st.write(f"运行模式：{'真实数据' if latest.mode == 'live_data' else '离线兜底'}")
            st.write(f"股票池数量：{latest.universe_size}")
            st.write(f"最终持仓：{len(latest.selected)}")
            st.write(f"漂移警报：{len(latest.drift_alerts)}")
            st.write(f"纸面订单：{len(latest.trade_orders)}")

    with right:
        st.subheader("下一步建议")
        st.write("1. 在“LLM测试”确认模型能响应。")
        st.write("2. 在“A股数据源测试”确认 TuShare 行情和财报能返回数据。")
        st.write("3. 在“本地仓位记忆”录入你的当前持仓。")
        st.write("4. 在“投资方案生成”运行一次纸面投顾流程。")

    st.divider()
    st.subheader("计划书完成度概览")
    progress = pd.DataFrame(
        [
            {"stage": "第一阶段 数据管道", "done": 0.65},
            {"stage": "第二阶段 多智能体", "done": 0.35},
            {"stage": "第三阶段 BL配置", "done": 0.35},
            {"stage": "第四阶段 风控调仓", "done": 0.45},
            {"stage": "第五阶段 回测与GUI", "done": 0.40},
        ]
    )
    st.plotly_chart(_progress_chart(progress), use_container_width=True)


def _render_llm_test() -> None:
    st.subheader("LLM 与工具测试")
    cfg = get_llm_config()
    checks = [
        HealthCheck("API Key", "normal" if cfg.api_key else "error", "已配置" if cfg.api_key else "未配置"),
        HealthCheck("Base URL", "normal" if cfg.base_url else "warning", cfg.base_url or "使用默认 OpenAI 地址"),
        HealthCheck("Chat Model", "normal", cfg.chat_model),
        HealthCheck("Embedding Model", "normal", cfg.embedding_model),
    ]
    _render_health_cards(checks)

    tab_llm, tab_tools = st.tabs(["LLM 连接", "Tool Calling 调试台"])
    with tab_llm:
        prompt = st.text_input("测试提示词", value="请用一句中文说明你已准备好协助A股投研。")
        if st.button("测试 LLM", type="primary"):
            with st.status("正在请求大模型...", expanded=True) as status:
                try:
                    from openai import OpenAI

                    start = time.perf_counter()
                    client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
                    response = client.chat.completions.create(
                        model=cfg.chat_model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=300,
                    )
                    elapsed = time.perf_counter() - start
                    content = response.choices[0].message.content or ""
                    usage = getattr(response, "usage", None)
                    record = record_llm_usage(
                        feature="GUI LLM测试",
                        model=cfg.chat_model,
                        prompt_text=prompt,
                        completion_text=content,
                        prompt_tokens=getattr(usage, "prompt_tokens", None) if usage is not None else None,
                        completion_tokens=getattr(usage, "completion_tokens", None) if usage is not None else None,
                        metadata={"elapsed_seconds": elapsed},
                    )
                    SQLiteStore().save_llm_usage_record(record)
                    status.update(label="LLM 测试成功", state="complete")
                    st.success(f"响应耗时：{elapsed:.2f} 秒")
                    st.caption(f"估算 token：{record.total_tokens}，估算成本：${record.estimated_cost_usd:.4f}")
                    st.markdown(content)
                except Exception as exc:
                    status.update(label="LLM 测试失败", state="error")
                    st.error(_friendly_error(exc))

    with tab_tools:
        tool_col_a, tool_col_b, tool_col_c = st.columns(3)
        tool_ticker = tool_col_a.text_input("工具测试股票", value="600519.SH")
        tool_market = tool_col_b.selectbox("市场", options=["cn", "us"], index=0)
        tool_limit = tool_col_c.slider("返回条数/期数", min_value=1, max_value=5, value=2)
        tool_choice = st.multiselect(
            "要测试的工具",
            options=["财务报表工具", "最近新闻工具"],
            default=["财务报表工具", "最近新闻工具"],
        )
        use_bound_llm = st.checkbox("额外测试 LLM 是否能绑定这些工具", value=False)

        if st.button("运行工具测试", use_container_width=True):
            trace = _run_financial_tool_debug_trace(
                ticker=tool_ticker,
                market=tool_market,
                limit=tool_limit,
                include_statements="财务报表工具" in tool_choice,
                include_news="最近新闻工具" in tool_choice,
                use_bound_llm=use_bound_llm,
            )
            _render_tool_debug_trace(trace)


def _render_a_share_data_test() -> None:
    st.subheader("A股数据源测试")
    cfg = get_tushare_config()
    checks = [
        HealthCheck("TuShare Token", "normal" if cfg.token else "error", "已配置" if cfg.token else "未配置"),
        HealthCheck("TuShare 网关", "normal" if cfg.http_url else "warning", cfg.http_url or "使用 TuShare 默认地址"),
    ]
    _render_health_cards(checks)

    col_a, col_b, col_c = st.columns(3)
    ticker = col_a.text_input("行情测试股票", value="000001.SZ")
    statement_ticker = col_b.text_input("财报测试股票", value="600519.SH")
    source_mode_label = col_c.selectbox(
        "数据源模式",
        options=["自动兜底", "TuShare 优先", "AkShare 优先"],
        index=0,
    )
    start_date = st.text_input("行情开始日期", value="2018-07-01")
    end_date = st.text_input("行情结束日期", value="2018-07-18")
    mode_map = {"自动兜底": "auto", "TuShare 优先": "tushare", "AkShare 优先": "akshare"}

    if st.button("一键测试 A股行情与三张财报", type="primary", use_container_width=True):
        with st.status("正在访问 TuShare/AkShare...", expanded=True) as status:
            try:
                st.write("1. 检查 OHLCV 日线行情")
                st.write("2. 检查资产负债表、现金流量表、利润表")
                st.write("3. 保存最近一次数据源健康报告")
                report = asyncio.run(
                    run_a_share_data_source_health_check(
                        price_ticker=ticker,
                        statement_ticker=statement_ticker,
                        start_date=start_date,
                        end_date=end_date,
                        mode=mode_map[source_mode_label],
                    )
                )
            except Exception as exc:
                status.update(label="A股数据源测试失败", state="error")
                st.error(_friendly_error(exc))
                return
            status.update(
                label="A股数据源测试完成",
                state="complete" if report.is_healthy else "error",
            )
        _render_data_source_health_report(report)
    else:
        latest = load_latest_data_source_health_report()
        if latest is not None:
            st.info("显示最近一次数据源健康检查结果。")
            _render_data_source_health_report(latest)


def _render_test_tools_page() -> None:
    st.subheader("测试工具")
    st.write("这里集中放置 LLM、Tool Calling、A 股数据源和 GUI 验收测试。普通投资流程不需要频繁进入这些调试页。")
    tab_llm, tab_data, tab_acceptance = st.tabs(["LLM / Tool Calling", "A股数据源", "GUI验收"])
    with tab_llm:
        _render_llm_test()
    with tab_data:
        _render_a_share_data_test()
    with tab_acceptance:
        _render_todo_acceptance()


def _render_conversational_advisor_page() -> None:
    account = _render_local_account_center()
    if st.button("开始新的投顾对话", type="secondary"):
        session = start_conversational_advisor_session(market="cn", user=account)
        st.session_state["advisor_session_id"] = session.session_id
        st.rerun()

    session = load_latest_conversational_advisor_session(user_id=account.user_id)
    if session is None:
        session = start_conversational_advisor_session(market="cn", user=account)

    _render_advisor_chat(session)
    if session.agent_events:
        _render_agent_call_trace(session.agent_events, expanded=False)
    user_message = st.chat_input("输入你的投资目标、股票选择或追问...")
    if user_message:
        with st.chat_message("user"):
            st.markdown(user_message)
        with st.chat_message("assistant"):
            placeholder = st.empty()
            trace_box = st.empty()
            progress_events: list[Any] = []

            def on_agent_event(event: Any) -> None:
                _append_live_agent_event(progress_events, event)
                with trace_box.container():
                    _render_live_agent_trace(progress_events)

            try:
                _stream_advisor_progress(placeholder, session.stage)
                session = continue_conversational_advisor_session(
                    session,
                    user_message,
                    use_llm=True,
                    progress_callback=on_agent_event,
                )
            except Exception as exc:
                placeholder.error(_friendly_error(exc))
                return
            placeholder.markdown(session.messages[-1].content)
            with trace_box.container():
                _render_agent_call_trace(session.agent_events, expanded=True)
        st.rerun()

    if session.artifacts:
        _render_conversation_artifacts(session.artifacts)


def _render_local_account_center() -> LocalUserAccount:
    accounts = list_user_accounts()
    active = get_or_create_active_user()
    if not accounts:
        accounts = [active]
    account_labels = {f"{item.display_name}（{item.user_id}）": item for item in accounts}
    active_label = next(
        (label for label, item in account_labels.items() if item.user_id == active.user_id),
        next(iter(account_labels)),
    )

    with st.expander("账号与本地记忆", expanded=False):
        st.caption("不同账号的对话、画像、投资角色、目标和产物会分开保存在 data/processed/users/。")
        col_a, col_b = st.columns([0.55, 0.45])
        selected_label = col_a.selectbox("当前账号", list(account_labels.keys()), index=list(account_labels).index(active_label))
        selected = account_labels[selected_label]
        if selected.user_id != active.user_id:
            set_active_user(selected.user_id)
            active = selected
            st.rerun()

        col_b.metric("投资角色", active.investment_role)
        with st.form("local_account_form"):
            name = st.text_input("账号名称", value=active.display_name)
            role = st.text_input("投资角色", value=active.investment_role)
            goals = st.text_area("长期目标 / 偏好", value=active.goals, height=90)
            submitted = st.form_submit_button("保存账号记忆", use_container_width=True)
        if submitted:
            active = save_user_account(name, investment_role=role, goals=goals, user_id=active.user_id)
            st.success("账号记忆已保存。")
            st.rerun()
        new_col_a, new_col_b = st.columns([0.7, 0.3])
        new_name = new_col_a.text_input("新建账号名称", value="", placeholder="例如：家庭稳健账户")
        if new_col_b.button("新建账号", use_container_width=True, disabled=not new_name.strip()):
            active = save_user_account(new_name.strip(), investment_role="个人投资者")
            st.rerun()
    return active


def _render_advisor_chat(session: ConversationalAdvisorSession) -> None:
    for message in session.messages[-20:]:
        with st.chat_message(message.role):
            st.markdown(message.content)


def _append_live_agent_event(events: list[Any], event: Any) -> None:
    payload = event.to_dict() if hasattr(event, "to_dict") else dict(event)
    metadata = payload.get("metadata") or {}
    stream_id = metadata.get("stream_id")
    if stream_id and payload.get("status") in {"running", "success"}:
        for idx in range(len(events) - 1, -1, -1):
            existing = events[idx].to_dict() if hasattr(events[idx], "to_dict") else dict(events[idx])
            existing_metadata = existing.get("metadata") or {}
            if existing_metadata.get("stream_id") == stream_id and existing.get("status") == "running":
                events[idx] = event
                return
    events.append(event)


def _stream_advisor_progress(placeholder: Any, stage: str) -> None:
    if stage == "profile":
        lines = [
            "客户画像Agent：正在理解你的目标、期限、回撤容忍和约束...",
            "记忆模块：准备把投资角色和长期目标保存到本地账号...",
            "对话LLM：准备自然引导你给出股票池或代码...",
        ]
    elif stage == "universe":
        lines = [
            "选股Agent：正在识别你提到的股票池或股票代码...",
            "选股Agent：如果范围过大，会主动控制到 5-10 只可分析股票...",
            "组合经理Agent：准备生成纸面组合候选...",
            "搜索Agent：准备逐只检索新闻和公开资料...",
            "报告Agent：准备整理 CSV、JSON 和 PDF 研究记录...",
        ]
    else:
        lines = [
            "对话LLM：正在读取已生成的组合、风险和工具轨迹...",
            "相关Agent：必要时会重新生成组合或补充解释...",
            "异常解释器：如果后台有降级，会主动说明影响...",
        ]
    content = ""
    for line in lines:
        content += line + "\n\n"
        placeholder.markdown(content)
        time.sleep(0.25)


def _render_agent_call_trace(events: list[Any], *, expanded: bool = True) -> None:
    if not events:
        return
    with st.expander("Agent 调用过程", expanded=expanded):
        for event in events[-12:]:
            payload = event.to_dict() if hasattr(event, "to_dict") else dict(event)
            status = str(payload.get("status") or "unknown")
            icon = {
                "success": "完成",
                "failed": "失败",
                "running": "运行中",
                "pending": "等待",
                "skipped": "跳过",
            }.get(status, status)
            elapsed = float(payload.get("elapsed_seconds") or 0.0)
            st.markdown(f"**{payload.get('agent', 'Agent')}** · {icon} · {elapsed:.2f}s")
            if payload.get("summary"):
                st.caption(str(payload["summary"]))
            if payload.get("input_summary"):
                st.write(f"输入：{payload['input_summary']}")
            if payload.get("output_summary"):
                st.write(f"输出：{payload['output_summary']}")
            if payload.get("error"):
                st.warning(str(payload["error"]))
            st.divider()


def _render_live_agent_trace(events: list[Any]) -> None:
    st.markdown("**Agent 调用过程**")
    visible_events = _compact_live_agent_events(events)
    for event in visible_events[-8:]:
        payload = event.to_dict() if hasattr(event, "to_dict") else dict(event)
        status = str(payload.get("status") or "unknown")
        streaming = bool((payload.get("metadata") or {}).get("streaming"))
        icon = {
            "success": "完成",
            "failed": "失败",
            "running": "运行中",
            "pending": "等待",
            "skipped": "跳过",
        }.get(status, status)
        elapsed = float(payload.get("elapsed_seconds") or 0.0)
        stream_label = " · 流式输出" if streaming else ""
        st.write(f"{payload.get('agent', 'Agent')} · {icon}{stream_label} · {elapsed:.2f}s：{payload.get('summary', '')}")
        if payload.get("input_summary"):
            st.caption(f"输入：{payload['input_summary']}")
        if payload.get("output_summary"):
            st.code(str(payload["output_summary"]), language="markdown")
        if payload.get("error"):
            st.warning(str(payload["error"]))


def _compact_live_agent_events(events: list[Any]) -> list[Any]:
    compacted: list[Any] = []
    stream_indexes: dict[str, int] = {}
    for event in events:
        payload = event.to_dict() if hasattr(event, "to_dict") else dict(event)
        metadata = payload.get("metadata") or {}
        stream_id = metadata.get("stream_id")
        if stream_id:
            idx = stream_indexes.get(stream_id)
            if idx is not None:
                compacted[idx] = event
                continue
            stream_indexes[stream_id] = len(compacted)
        compacted.append(event)
    return compacted


def _render_conversation_artifacts(artifacts: list[dict[str, Any]]) -> None:
    st.caption("本轮结果已保存到本地，可下载留档。")
    for idx, artifact in enumerate(artifacts[-6:]):
        path = Path(str(artifact.get("path", "")))
        if not path.exists():
            continue
        mime = {
            "json": "application/json",
            "csv": "text/csv",
            "pdf": "application/pdf",
        }.get(str(artifact.get("artifact_type")), "application/octet-stream")
        st.download_button(
            f"下载 {artifact.get('name')}",
            path.read_bytes(),
            file_name=path.name,
            mime=mime,
            use_container_width=True,
            key=f"conversation_artifact_download_{idx}_{abs(hash(str(path.resolve())))}",
        )


def _render_a_share_universe_page() -> None:
    st.subheader("股票池选择器")
    st.write("选择板块或指数成分，生成可复用的 A 股股票池。BL、投资方案和历史回测会优先使用最近一次生成的股票池。")

    col_a, col_b, col_c = st.columns([1.4, 0.6, 0.6])
    sources = col_a.multiselect(
        "股票池来源",
        options=["沪深300", "中证500", "中证1000", "全A", "主板", "创业板", "科创板", "北交所"],
        default=["沪深300"],
    )
    max_count = col_b.number_input("最大股票数量", min_value=5, max_value=1000, value=100, step=5)
    force_refresh = col_c.checkbox("强制刷新", value=False)

    if st.button("生成股票池", type="primary", use_container_width=True):
        with st.status("正在构建 A 股股票池...", expanded=True) as status:
            try:
                st.write("1. 检查本地缓存")
                st.write("2. 调用 TuShare stock_basic / 指数成分接口")
                st.write("3. 写入最近一次股票池结果")
                result = build_a_share_universe(sources=sources, max_count=int(max_count), force_refresh=force_refresh)
            except Exception as exc:
                status.update(label="股票池生成失败", state="error")
                st.error(_friendly_error(exc))
                return
            status.update(label="股票池生成完成", state="complete")
        _render_a_share_universe_result(result)
    else:
        latest = load_latest_a_share_universe()
        if latest is not None:
            st.info("显示最近一次股票池结果。")
            _render_a_share_universe_result(latest)


def _render_financial_rag_page() -> None:
    st.subheader("财报知识库")
    st.write("上传本地 PDF 年报，系统会分块、生成页摘要，并建立向量搜索 + BM25 混合检索索引。")

    col_a, col_b = st.columns([0.6, 0.4])
    with col_a:
        ticker = st.text_input("绑定股票代码", value="600519.SH")
        uploaded_pdf = st.file_uploader("上传 PDF 年报", type=["pdf"])
    with col_b:
        overwrite = st.checkbox("覆盖该股票已有索引", value=True)
        top_k = st.slider("检索段落数量", min_value=1, max_value=10, value=5, step=1)

    if st.button("导入财报索引", type="primary", use_container_width=True):
        if uploaded_pdf is None:
            st.error("请先上传 PDF 文件。")
        else:
            with st.status("正在导入财报知识库...", expanded=True) as status:
                try:
                    st.write("1. 保存上传的 PDF")
                    pdf_path = save_uploaded_report(uploaded_pdf, ticker=ticker)
                    st.write("2. 读取 PDF 页面并生成页摘要")
                    st.write("3. 分块并写入 Chroma 向量库与 BM25 文档库")
                    info = ingest_uploaded_financial_report(pdf_path, ticker=ticker, overwrite=overwrite)
                except Exception as exc:
                    status.update(label="财报导入失败", state="error")
                    st.error(_friendly_error(exc))
                    return
                status.update(label="财报导入完成", state="complete")
            _render_financial_index_info(info)

    st.divider()
    st.subheader("已导入财报")
    infos = list_indexed_financial_reports()
    if not infos:
        st.info("还没有导入任何财报 PDF。")
    else:
        st.dataframe(
            pd.DataFrame([info.to_dict() for info in infos]).drop(columns=["page_summaries"], errors="ignore"),
            use_container_width=True,
            hide_index=True,
        )
        selected_ticker = st.selectbox("选择已导入股票", options=[info.ticker for info in infos])
        selected_info = next(info for info in infos if info.ticker == selected_ticker)
        _render_financial_index_info(selected_info)
        delete_col, rebuild_col = st.columns([0.35, 0.65])
        if delete_col.button("删除该股票RAG索引", type="secondary", use_container_width=True):
            try:
                deleted_info = delete_indexed_financial_report(selected_ticker)
            except Exception as exc:
                st.error(_friendly_error(exc))
            else:
                st.success(f"已删除 {selected_ticker} 的本地RAG索引。")
                _render_financial_index_info(deleted_info)
        rebuild_col.caption("重建索引：重新上传该股票 PDF，并勾选“覆盖该股票已有索引”。")

    st.divider()
    st.subheader("问财报")
    query = st.text_input("问题", value="公司的毛利率趋势和债务风险如何？")
    query_ticker = st.text_input("查询股票代码", value=ticker)
    if st.button("检索财报上下文", use_container_width=True):
        with st.status("正在混合检索财报上下文...", expanded=True) as status:
            try:
                answer = ask_financial_report(ticker=query_ticker, query=query, top_k=top_k)
            except Exception as exc:
                status.update(label="财报检索失败", state="error")
                st.error(_friendly_error(exc))
                return
            status.update(label="财报检索完成", state="complete")
        _render_financial_context_answer(answer)


def _render_position_memory(position_memory: PositionMemory) -> None:
    st.subheader("本地仓位记忆")
    st.write(f"当前保存路径：`{position_memory.path}`")

    uploaded_positions = st.file_uploader("上传当前持仓CSV", type=["csv"])
    if uploaded_positions is not None:
        uploaded_frame = pd.read_csv(uploaded_positions)
        position_memory.save(uploaded_frame)
        st.success("已保存上传的当前持仓。")

    saved_positions = position_memory.load()
    editable_positions = saved_positions if not saved_positions.empty else position_memory.template_frame()
    edited_positions = _positions_editor(editable_positions, key="positions_page_editor")

    col_a, col_b = st.columns([1, 1])
    if col_a.button("保存本地仓位CSV", type="primary", use_container_width=True):
        path = position_memory.save(edited_positions)
        st.success(f"已保存到 {path}")

    export_frame = position_memory.normalize(edited_positions)
    col_b.download_button(
        "导出仓位CSV",
        export_frame.to_csv(index=False).encode("utf-8-sig"),
        file_name="current_positions.csv",
        mime="text/csv",
        use_container_width=True,
    )

    weights = position_memory.normalize(edited_positions)
    if not weights.empty:
        st.subheader("当前仓位权重")
        weight_series = pd.Series(position_memory.normalize(edited_positions)["weight"].values, index=weights["ticker"])
        if weight_series.sum() > 0:
            weight_series = weight_series / weight_series.sum()
            st.plotly_chart(_pie_chart(weight_series), use_container_width=True)


def _render_client_profile_page() -> None:
    st.subheader("客户画像问卷")
    st.write("用自然语言和简单选项生成 BL/组合优化会用到的约束：目标波动率、最低预期收益、单只股票上限。")

    col_a, col_b, col_c, col_d = st.columns(4)
    risk_level = col_a.selectbox("风险类型", options=["保守型", "稳健型", "平衡型", "成长型", "激进型"], index=1)
    horizon = col_b.selectbox("投资期限", options=["1年以内", "1-3年", "3-5年", "5年以上"], index=2)
    liquidity_need = col_c.selectbox("流动性需求", options=["高", "中", "低"], index=1)
    max_drawdown = col_d.slider("最大可承受回撤", min_value=0.05, max_value=0.40, value=0.15, step=0.01)
    profile_text = st.text_area(
        "补充描述",
        value="我是保守型投资者，期望跑赢通胀即可，不希望账户大幅波动。",
        height=100,
    )
    use_llm = st.checkbox("使用 LLM 结构化解析客户画像", value=False)

    if st.button("生成客户约束", type="primary", use_container_width=True):
        with st.status("正在解析客户画像...", expanded=True) as status:
            try:
                result = parse_client_profile_questionnaire(
                    risk_level=risk_level,
                    horizon=horizon,
                    liquidity_need=liquidity_need,
                    max_drawdown_tolerance=max_drawdown,
                    natural_language_profile=profile_text,
                    use_llm=use_llm,
                )
            except Exception as exc:
                status.update(label="客户画像解析失败", state="error")
                st.error(_friendly_error(exc))
                return
            status.update(label="客户画像解析完成", state="complete")
        _render_client_profile_result(result)
    else:
        latest = load_latest_client_profile()
        if latest is not None:
            st.info("显示最近一次客户画像解析结果。")
            _render_client_profile_result(latest)


def _render_single_ticker_technical_page() -> None:
    st.subheader("个股技术分析")
    st.write("输入任意 A 股代码，查看收盘价、布林带、MA50/MA200、RSI、MACD 和成交量信号。")

    col_a, col_b, col_c, col_d = st.columns(4)
    ticker = col_a.text_input("股票代码", value="600519.SH")
    market = col_b.selectbox("市场", options=["cn", "us"], index=0)
    lookback_days = col_c.slider("观察天数", min_value=120, max_value=1000, value=365, step=30)
    force_refresh = col_d.checkbox("强制刷新", value=False)
    end_date = st.text_input("截止日期，可留空使用今天", value="")

    if st.button("计算技术指标", type="primary", use_container_width=True):
        with st.status("正在拉取行情并计算技术指标...", expanded=True) as status:
            try:
                report = run_single_ticker_technical_analysis(
                    ticker=ticker,
                    market=market,
                    lookback_days=int(lookback_days),
                    end_date=end_date.strip() or None,
                    force_refresh=force_refresh,
                )
            except Exception as exc:
                status.update(label="技术分析失败", state="error")
                st.error(_friendly_error(exc))
                return
            status.update(label="技术分析完成", state="complete")
        _render_technical_analysis_report(report)
    else:
        latest = load_latest_technical_analysis_report()
        if latest is not None:
            st.info("显示最近一次技术分析结果。")
            _render_technical_analysis_report(latest)


def _render_investment_assistant(
    assistant: ChinaInvestmentAssistant,
    position_memory: PositionMemory,
) -> None:
    st.subheader("投资方案生成")
    latest_universe = load_latest_a_share_universe()
    if latest_universe is not None:
        st.caption(f"已读取股票池选择器：{len(latest_universe.items)} 只，来源 {','.join(latest_universe.sources)}。")
    else:
        st.caption("尚未生成股票池选择器结果，将使用项目内置核心股票池。")

    profile_text = st.text_area(
        "请用一句话描述你的风险偏好",
        value="我是保守型投资者，期望跑赢通胀即可",
        height=100,
    )
    col_a, col_b, col_c, col_d = st.columns(4)
    max_slider_limit = max(30, len(latest_universe.items) if latest_universe is not None else 30)
    max_candidates = col_a.slider("股票池覆盖数量", min_value=5, max_value=max_slider_limit, value=min(30, max_slider_limit), step=5)
    selected_count = col_b.slider("最终持仓数量", min_value=5, max_value=15, value=10, step=1)
    use_llm_report = col_c.checkbox("使用大模型生成研报", value=True)
    drift_threshold = col_d.slider("漂移触发阈值", min_value=0.01, max_value=0.20, value=0.05, step=0.01, format="%.2f")

    saved_positions = position_memory.load()
    if saved_positions.empty:
        st.warning("尚未保存本地仓位。系统会使用演示仓位进行漂移监控。")
    else:
        with st.expander("当前会用于漂移监控的本地仓位", expanded=False):
            st.dataframe(saved_positions, use_container_width=True, hide_index=True)

    run_button = st.button("生成投资方案", type="primary", use_container_width=True)
    if run_button:
        current_weights = position_memory.load_weights()
        with st.status("正在运行投研流水线...", expanded=True) as status:
            st.write("1. 检查 A 股数据源并构建股票池")
            st.write("2. 计算技术指标和候选评分")
            st.write("3. 读取本地仓位 CSV，生成目标权重、漂移警报和纸面订单")
            st.write("4. 调用大模型生成 HTML 持仓研报")
            result = asyncio.run(
                assistant.run(
                    profile_text=profile_text,
                    max_candidates=max_candidates,
                    selected_count=selected_count,
                    use_llm_report=use_llm_report,
                    current_weights=current_weights or None,
                    universe=_universe_result_to_candidates(latest_universe) if latest_universe is not None else None,
                    drift_threshold=drift_threshold,
                )
            )
            status.update(label="投资方案已生成", state="complete")
        _render_result(result)
    else:
        latest = assistant.load_latest_result()
        if latest is not None:
            st.info("显示最近一次生成的投资方案。")
            _render_result(latest)


def _render_black_litterman_page() -> None:
    st.subheader("Black-Litterman 模型")
    st.write("这里把计划书第三阶段落到 GUI：A 股市值权重、市场均衡先验、P/Q/Omega、后验权重和有效前沿对比。")

    latest_universe = load_latest_a_share_universe()
    default_tickers = _default_tickers_from_universe(
        latest_universe,
        fallback="600519.SH,000858.SZ,000333.SZ,300750.SZ,600036.SH",
        limit=8,
    )
    if latest_universe is not None:
        st.caption(f"默认使用股票池选择器前 8 只：来源 {','.join(latest_universe.sources)}。可在下方手动修改。")
    tickers_text = st.text_area("股票池代码，用英文逗号分隔", value=default_tickers, height=80)
    tickers = _parse_tickers(tickers_text)

    col_a, col_b, col_c, col_d = st.columns(4)
    years = col_a.slider("历史年限", min_value=1, max_value=5, value=2, step=1)
    end_date = col_b.text_input("截止日期", value="2024-12-31")
    risk_aversion = col_c.number_input("风险厌恶系数", min_value=0.1, max_value=10.0, value=2.5, step=0.1)
    risk_free_rate = col_d.number_input("无风险利率", min_value=0.0, max_value=0.1, value=0.02, step=0.005)

    latest_profile = load_latest_client_profile()
    default_profile_text = (
        latest_profile.natural_language_profile
        if latest_profile is not None and latest_profile.natural_language_profile
        else "我是保守型投资者，期望跑赢通胀即可"
    )
    profile_text = st.text_input("客户画像", value=default_profile_text)
    if latest_profile is not None:
        st.caption(
            "已读取客户画像问卷约束："
            f"目标波动≤{(latest_profile.constraints.max_volatility or 0):.1%}，"
            f"最低收益≥{(latest_profile.constraints.min_expected_return or 0):.1%}，"
            f"单股≤{latest_profile.constraints.max_single_asset_weight:.1%}"
        )
    view_mode = st.radio("观点模式", options=["绝对观点", "相对观点"], horizontal=True)
    view_frame = _default_view_frame(tickers) if view_mode == "绝对观点" else _default_relative_view_frame(tickers)
    edited_views = st.data_editor(
        view_frame,
        num_rows="dynamic",
        use_container_width=True,
        key="bl_views_editor",
        column_config={
            "ticker": st.column_config.TextColumn("股票代码"),
            "long_ticker": st.column_config.TextColumn("看多股票"),
            "short_ticker": st.column_config.TextColumn("看空股票"),
            "expected_excess_return": st.column_config.NumberColumn("观点超额收益", format="%.2%"),
            "relative_excess_return": st.column_config.NumberColumn("相对超额收益", format="%.2%"),
            "confidence": st.column_config.NumberColumn("置信度", min_value=0.0, max_value=1.0, format="%.0%"),
            "source": st.column_config.SelectboxColumn("观点来源", options=["人工", "基本面", "情绪", "技术面", "LLM融合"]),
        },
    )

    if st.button("计算 Black-Litterman 组合", type="primary", use_container_width=True):
        if len(tickers) < 2:
            st.error("至少需要两个股票代码。")
            return

        with st.status("正在计算 BL 模型...", expanded=True) as status:
            try:
                st.write("1. 拉取 A 股历史价格")
                st.write("2. 使用 TuShare daily_basic 获取总市值")
                prior = build_market_equilibrium_prior(
                    tickers,
                    market="cn",
                    years=years,
                    risk_aversion=risk_aversion,
                    risk_free_rate=risk_free_rate,
                    end_date=end_date,
                )

                st.write("3. 解析观点矩阵 P、观点向量 Q 和 Idzorek 置信度")
                views = _views_frame_to_bl_views(edited_views, prior.tickers, mode=view_mode)
                bl_model = build_black_litterman_model_with_idzorek(
                    views,
                    prior.covariance_matrix,
                    prior.implied_prior_returns,
                    risk_aversion=risk_aversion,
                )
                omega = build_idzorek_omega(
                    views,
                    prior.covariance_matrix,
                    prior.implied_prior_returns,
                    risk_aversion=risk_aversion,
                )
                posterior_returns = bl_model.bl_returns()
                posterior_covariance = bl_model.bl_cov()

                st.write("4. 结合客户约束求解最终权重")
                constraints = latest_profile.constraints if latest_profile is not None else parse_client_constraints_from_text(profile_text)
                optimized = optimize_bl_portfolio_weights(
                    posterior_returns,
                    posterior_covariance,
                    constraints=constraints,
                )
                status.update(label="BL 模型计算完成", state="complete")
            except Exception as exc:
                status.update(label="BL 模型计算失败", state="error")
                st.error(_friendly_error(exc))
                return

        _render_bl_result(prior, views, posterior_returns, posterior_covariance, optimized, omega)


def _render_agent_workflow_page() -> None:
    st.subheader("多智能体工作流")
    st.write("单标的分析链路：基本面分析师 -> 情绪分析师 -> 技术面分析师 -> 投资组合经理。")

    col_a, col_b, col_c = st.columns(3)
    ticker = col_a.text_input("股票代码", value="600519.SH")
    market = col_b.selectbox("市场", options=["cn", "us"], index=0)
    use_llm = col_c.checkbox("使用 LLM", value=True)

    run_button = st.button("运行多智能体分析", type="primary", use_container_width=True)
    if run_button:
        with st.status("正在运行多智能体工作流...", expanded=True) as status:
            st.write("1. 基本面分析师读取财报并形成价值观点")
            st.write("2. 情绪分析师读取最近新闻并给出情绪分数")
            st.write("3. 技术面分析师计算 RSI、MACD、布林带和均线")
            st.write("4. 投资组合经理融合信号并生成最终决策")
            result = run_single_ticker_agent_analysis(ticker=ticker, market=market, use_llm=use_llm)
            status.update(label="多智能体分析完成", state="complete")
        _render_agent_analysis_result(result)
    else:
        latest = SingleTickerAgentAnalysisRunner().load_latest_result()
        if latest is not None:
            st.info("显示最近一次多智能体分析结果。")
            _render_agent_analysis_result(latest)


def _render_agent_interaction_center() -> None:
    st.subheader("智能体交互中心")
    st.write("用自然语言让系统选择并调用已有智能体。当前支持多智能体个股分析、技术分析、财报问答、财务工具、一键全流程和轻量搜索 Agent。")

    default_prompt = "请分析 600519.SH，运行多智能体工作流，并给出最终投资建议。"
    user_request = st.text_area("你想让智能体做什么？", value=default_prompt, height=110)
    col_a, col_b, col_c = st.columns([0.8, 0.8, 0.8])
    default_ticker = col_a.text_input("默认股票代码", value=_extract_first_ticker(user_request) or "600519.SH")
    market = col_b.selectbox("市场", options=["cn", "us"], index=0, key="agent_center_market")
    use_llm = col_c.checkbox("允许 LLM 解析意图/参与分析", value=True, key="agent_center_use_llm")
    max_tokens = st.slider("本次 LLM 路由最大输出 Token", min_value=100, max_value=800, value=300, step=50)

    if st.button("调用智能体", type="primary", use_container_width=True):
        with st.status("正在解析请求并调用智能体...", expanded=True) as status:
            st.write("1. 解析自然语言请求")
            route = _route_agent_request_with_llm(
                user_request,
                default_ticker=default_ticker,
                market=market,
                use_llm=use_llm,
                max_tokens=max_tokens,
            )
            st.write(f"2. 路由到：{route['action']}")
            try:
                result = _execute_agent_center_route(route, user_request=user_request, use_llm=use_llm)
            except Exception as exc:
                status.update(label="智能体调用失败", state="error")
                st.error(_friendly_error(exc))
                st.json(route)
                return
            status.update(label="智能体调用完成", state="complete")
        _render_agent_center_result(route, result)
    else:
        latest = SingleTickerAgentAnalysisRunner().load_latest_result()
        if latest is not None:
            with st.expander("最近一次多智能体结果", expanded=False):
                _render_agent_analysis_result(latest)


def _route_agent_request_with_llm(
    request: str,
    *,
    default_ticker: str,
    market: str,
    use_llm: bool,
    max_tokens: int,
) -> dict[str, Any]:
    fallback = _route_agent_request_by_rules(request, default_ticker=default_ticker, market=market)
    if not use_llm:
        return fallback

    cfg = get_llm_config()
    if not cfg.api_key:
        fallback["warnings"] = ["OPENAI_API_KEY 未配置，已使用规则路由。"]
        return fallback

    try:
        from openai import OpenAI

        prompt = (
            "你是A股智能投顾系统的安全路由器。只返回JSON，不要markdown。\n"
            "可选action只能是：agent_analysis, technical_analysis, rag_qa, financial_tools, full_workflow, search_agent。\n"
            "字段：action, ticker, market, question, reason。\n"
            f"默认ticker：{default_ticker}\n默认market：{market}\n用户请求：{request}"
        )
        client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
        response = client.chat.completions.create(
            model=cfg.chat_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content or "{}"
        record = record_llm_usage(
            feature="智能体交互中心路由",
            model=cfg.chat_model,
            prompt_text=prompt,
            completion_text=content,
            metadata={"default_ticker": default_ticker, "market": market},
        )
        SQLiteStore().save_llm_usage_record(record)
        route = _safe_json_loads(content)
        if not isinstance(route, dict):
            raise ValueError("LLM route is not a JSON object")
        route["action"] = _normalize_agent_action(str(route.get("action") or fallback["action"]))
        route["ticker"] = str(route.get("ticker") or default_ticker).upper().strip()
        route["market"] = str(route.get("market") or market).lower()
        route["question"] = str(route.get("question") or request)
        route.setdefault("reason", "LLM 路由完成。")
        route.setdefault("warnings", [])
        return route
    except Exception as exc:
        fallback["warnings"] = [f"LLM 路由失败，已使用规则路由：{_friendly_error(exc)}"]
        return fallback


def _route_agent_request_by_rules(request: str, *, default_ticker: str, market: str) -> dict[str, Any]:
    text = request.lower()
    if any(word in request for word in ["搜索", "查找", "资料", "新闻资料", "外部资料", "deepresearch", "DeepResearch"]):
        action = "search_agent"
    elif any(word in request for word in ["财报", "年报", "研报", "问答", "RAG"]):
        action = "rag_qa"
    elif any(word in request for word in ["技术", "RSI", "MACD", "均线"]):
        action = "technical_analysis"
    elif any(word in request for word in ["一键", "全流程", "部署", "投资方案"]):
        action = "full_workflow"
    elif any(word in request for word in ["工具", "新闻", "财务报表"]):
        action = "financial_tools"
    elif "search" in text:
        action = "search_agent"
    else:
        action = "agent_analysis"
    return {
        "action": action,
        "ticker": (_extract_first_ticker(request) or default_ticker).upper().strip(),
        "market": market,
        "question": request,
        "reason": "规则路由完成。",
        "warnings": [],
    }


def _execute_agent_center_route(route: dict[str, Any], *, user_request: str, use_llm: bool) -> dict[str, Any]:
    action = _normalize_agent_action(str(route.get("action") or "agent_analysis"))
    ticker = str(route.get("ticker") or "600519.SH").upper().strip()
    market = str(route.get("market") or "cn").lower()
    if market not in {"cn", "us"}:
        market = "cn"

    if action == "agent_analysis":
        result = run_single_ticker_agent_analysis(ticker=ticker, market=market, use_llm=use_llm)
        return {"type": action, "payload": result.to_dict()}
    if action == "technical_analysis":
        result = run_single_ticker_technical_analysis(ticker=ticker, market=market)
        return {"type": action, "payload": result.to_dict()}
    if action == "rag_qa":
        answer = ask_financial_report(str(route.get("question") or user_request), ticker=ticker, top_k=5)
        return {"type": action, "payload": answer.to_dict()}
    if action == "financial_tools":
        trace = _run_financial_tool_debug_trace(
            ticker=ticker,
            market=market,
            limit=2,
            include_statements=True,
            include_news=True,
            use_bound_llm=False,
        )
        return {"type": action, "payload": {"trace": trace}}
    if action == "full_workflow":
        run = run_full_advisory_workflow(
            profile_text=user_request,
            max_candidates=30,
            selected_count=10,
            include_backtest=False,
            expert_mode=False,
        )
        return {"type": action, "payload": run.to_dict()}
    if action == "search_agent":
        result = run_search_agent(
            str(route.get("question") or user_request),
            ticker=ticker,
            market=market,
            max_results=5,
            use_llm=use_llm,
        )
        return {"type": action, "payload": result.to_dict()}
    raise ValueError(f"Unsupported agent center action: {action}")


def _render_agent_center_result(route: dict[str, Any], result: dict[str, Any]) -> None:
    if route.get("warnings"):
        for warning in route["warnings"]:
            st.warning(warning)
    st.subheader("路由结果")
    st.json(route)

    result_type = result.get("type")
    payload = result.get("payload") or {}
    if result_type == "agent_analysis":
        _render_agent_analysis_result(AgentAnalysisResult(**{
            "ticker": payload["ticker"],
            "market": payload["market"],
            "use_llm": payload["use_llm"],
            "generated_at": payload["generated_at"],
            "final_decision": payload["final_decision"],
            "node_events": [AgentNodeEvent(**event) for event in payload.get("node_events", [])],
            "context": payload.get("context", []),
            "warnings": payload.get("warnings", []),
            "raw_state": payload.get("raw_state", {}),
        }))
    elif result_type == "technical_analysis":
        _render_technical_analysis_report(TechnicalAnalysisReport(**payload))
    elif result_type == "rag_qa":
        st.subheader("财报问答结果")
        st.write(payload.get("answer", ""))
        st.json(payload)
    elif result_type == "financial_tools":
        _render_tool_debug_trace(payload.get("trace", []))
    elif result_type == "full_workflow":
        st.subheader("一键全流程结果")
        cols = st.columns(4)
        cols[0].metric("运行ID", payload.get("run_id", ""))
        cols[1].metric("状态", _run_status_label(str(payload.get("status", "unknown"))))
        cols[2].metric("候选池", payload.get("max_candidates", 0))
        cols[3].metric("持仓数", payload.get("selected_count", 0))
        steps = pd.DataFrame(payload.get("steps") or [])
        if not steps.empty:
            st.dataframe(steps, use_container_width=True, hide_index=True)
        st.json(payload)
    elif result_type == "search_agent":
        _render_search_agent_result(SearchAgentResult.from_dict(payload))
    else:
        st.json(result)


def _normalize_agent_action(value: str) -> str:
    allowed = {"agent_analysis", "technical_analysis", "rag_qa", "financial_tools", "full_workflow", "search_agent"}
    normalized = value.strip().lower()
    return normalized if normalized in allowed else "agent_analysis"


def _render_search_agent_page() -> None:
    st.subheader("搜索Agent")
    st.write("轻量版 DeepResearch 风格链路：搜索外部资料 -> 读取网页 -> 汇总证据。默认优先 Serper；未配置时尝试 Jina Search/Reader。")
    col_a, col_b, col_c = st.columns([0.9, 0.6, 0.6])
    ticker = col_a.text_input("关联股票代码", value="600519.SH")
    market = col_b.selectbox("市场", options=["cn", "us"], index=0, key="search_agent_market")
    max_results = col_c.slider("搜索结果数", min_value=1, max_value=8, value=5)
    query = st.text_area("搜索问题", value="请搜索贵州茅台最近的业绩、分红和主要风险，并给出证据摘要。", height=110)
    use_llm = st.checkbox("使用 LLM 总结搜索结果", value=True, key="search_agent_use_llm")

    if st.button("运行搜索Agent", type="primary", use_container_width=True):
        with st.status("正在搜索与阅读外部资料...", expanded=True) as status:
            try:
                st.write("1. 搜索外部资料")
                st.write("2. 尝试读取网页正文")
                st.write("3. 汇总证据并生成中文摘要")
                result = run_search_agent(query, ticker=ticker, market=market, max_results=max_results, use_llm=use_llm)
            except Exception as exc:
                status.update(label="搜索Agent失败", state="error")
                st.error(_friendly_error(exc))
                return
            status.update(label="搜索Agent完成", state="complete")
        _render_search_agent_result(result)
    else:
        latest = load_latest_search_agent_result()
        if latest is not None:
            st.info("显示最近一次搜索Agent结果。")
            _render_search_agent_result(latest)


def _render_search_agent_result(result: SearchAgentResult) -> None:
    cols = st.columns(4)
    cols[0].metric("Provider", result.provider)
    cols[1].metric("来源数", len(result.sources))
    cols[2].metric("股票", result.ticker or "-")
    cols[3].metric("市场", result.market)
    for warning in result.warnings:
        st.warning(warning)
    st.subheader("搜索摘要")
    st.markdown(result.answer)
    sources = pd.DataFrame([source.to_dict() for source in result.sources])
    st.subheader("证据来源")
    if sources.empty:
        st.info("没有可展示的来源。")
    else:
        st.dataframe(sources.drop(columns=["content"], errors="ignore"), use_container_width=True, hide_index=True)
        st.download_button(
            "下载搜索结果 JSON",
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2, default=str).encode("utf-8"),
            file_name="search_agent_result.json",
            mime="application/json",
            use_container_width=True,
        )


def _render_persona_page() -> None:
    st.subheader("智能体 Persona")
    st.write("这里展示多智能体系统的系统提示词版本、职责边界和结构化输出要求，便于检查模型行为。")
    try:
        from fin_agent_sakura.agents.personas import PERSONA_PROMPT_REGISTRY, ValueInvestorAnalysis
    except Exception as exc:
        st.error(f"Persona 模块暂不可用：{_friendly_error(exc)}")
        return
    rows = [
        {
            "智能体": name,
            "版本": payload["version"],
            "提示词长度": len(payload["system_prompt"]),
        }
        for name, payload in PERSONA_PROMPT_REGISTRY.items()
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    selected = st.selectbox("选择智能体", options=list(PERSONA_PROMPT_REGISTRY.keys()))
    payload = PERSONA_PROMPT_REGISTRY[selected]
    st.caption(f"Prompt 版本：{payload['version']}")
    st.code(payload["system_prompt"], language="markdown")

    if selected == "价值投资者智能体":
        st.subheader("Pydantic 结构化输出字段")
        st.json(
            {
                "quality_checks": ["gross_margin_trend", "long_term_debt_to_equity", "leverage_risk"],
                "revenue_forecasts": ["bear", "base", "bull", "operating_margin", "capex_to_revenue", "working_capital_to_revenue"],
                "wacc_assumptions": ["risk_free_rate", "equity_risk_premium", "beta", "wacc"],
                "intrinsic_value_range": ["bear_case", "base_case", "bull_case", "terminal_growth_rate", "terminal_value", "sensitivity_analysis", "margin_of_safety"],
                "conclusion": ["strong_buy", "buy", "hold", "avoid", "insufficient_data"],
            }
        )
        validation_payload = _sample_value_investor_payload()
        try:
            validated = ValueInvestorAnalysis.model_validate(validation_payload)
        except Exception as exc:
            st.error(f"Pydantic 校验失败：{_friendly_error(exc)}")
        else:
            st.success("Pydantic 校验通过：价值投资者结构化输出 schema 可用。")
            st.json(validated.model_dump())


def _render_risk_gate_page(assistant: ChinaInvestmentAssistant) -> None:
    st.subheader("风险断路器")
    st.write("硬编码风控独立于大模型逻辑：只根据历史价格计算 proposed portfolio 的 VaR 和最大回撤。")

    latest = assistant.load_latest_result()
    _render_daily_monitor_schedule()
    st.divider()
    if latest is None:
        st.info("还没有投资方案。请先在“投资方案生成”页生成纸面订单。")
        return

    col_a, col_b, col_c, col_d = st.columns(4)
    max_var = col_a.number_input("最大 VaR", min_value=0.001, max_value=0.2, value=0.035, step=0.005, format="%.3f")
    max_drawdown = col_b.number_input("最大回撤", min_value=0.01, max_value=0.8, value=0.25, step=0.01, format="%.2f")
    var_confidence = col_c.number_input("VaR 置信度", min_value=0.80, max_value=0.99, value=0.95, step=0.01, format="%.2f")
    lookback_days = col_d.number_input("历史窗口天数", min_value=90, max_value=1500, value=365, step=30)

    if not latest.trade_orders:
        st.success("最近一次投资方案没有纸面订单，暂无需要风控拦截的交易。")
        report = load_latest_risk_gate_report()
        if report is not None:
            st.caption("最近一次风险报告")
            _render_risk_gate_report(report.to_dict())
        return

    st.caption("待评估纸面订单")
    st.dataframe(pd.DataFrame(latest.trade_orders), use_container_width=True, hide_index=True)

    if st.button("运行风险断路器", type="primary", use_container_width=True):
        with st.status("正在计算 VaR 与最大回撤...", expanded=True) as status:
            report = evaluate_paper_orders_risk(
                current_weights=latest.current_weights,
                orders=latest.trade_orders,
                market="cn",
                max_var=max_var,
                max_drawdown=max_drawdown,
                var_confidence=var_confidence,
                lookback_days=int(lookback_days),
            )
            status.update(label="风险断路器评估完成", state="complete" if report.approved else "error")
        _render_risk_gate_report(report.to_dict())
    elif latest.risk_gate:
        st.info("显示投资方案生成时自动保存的风险断路器结果。")
        _render_risk_gate_report(latest.risk_gate)
    else:
        report = load_latest_risk_gate_report()
        if report is not None:
            st.info("显示最近一次手动风险断路器结果。")
            _render_risk_gate_report(report.to_dict())


def _render_daily_monitor_schedule() -> None:
    st.subheader("每日漂移检查")
    st.caption("保存本地每日检查计划；本地版不会自动悄悄运行，点击“立即执行一次”会用本地仓位和最近目标权重写入漂移日志。")
    schedule = load_daily_monitor_schedule()
    latest_result = load_latest_daily_monitor_result()

    col_a, col_b, col_c, col_d, col_e = st.columns([0.65, 0.9, 0.9, 0.7, 0.7])
    enabled = col_a.toggle("启用计划", value=schedule.enabled if schedule else False)
    client_id = col_b.text_input("客户ID", value=schedule.client_id if schedule else "paper_client")
    portfolio_id = col_c.text_input("组合ID", value=schedule.portfolio_id if schedule else "default")
    run_time = col_d.text_input("每日时间", value=schedule.run_time if schedule else "09:30")
    threshold = col_e.number_input(
        "漂移阈值",
        min_value=0.005,
        max_value=0.50,
        value=float(schedule.drift_threshold if schedule else 0.05),
        step=0.005,
        format="%.3f",
    )

    save_col, run_col = st.columns(2)
    if save_col.button("保存每日检查计划", use_container_width=True):
        try:
            saved = save_daily_monitor_schedule(
                enabled=enabled,
                client_id=client_id,
                portfolio_id=portfolio_id,
                run_time=run_time,
                drift_threshold=float(threshold),
            )
        except Exception as exc:
            st.error(_friendly_error(exc))
        else:
            st.success("每日检查计划已保存。")
            st.json(saved.to_dict())

    if run_col.button("立即执行一次漂移检查", type="primary", use_container_width=True):
        with st.status("正在读取本地仓位与目标权重...", expanded=True) as status:
            try:
                result = run_daily_monitor_check_once(
                    client_id=client_id,
                    portfolio_id=portfolio_id,
                    drift_threshold=float(threshold),
                )
            except Exception as exc:
                status.update(label="每日漂移检查失败", state="error")
                st.error(_friendly_error(exc))
            else:
                status.update(
                    label="每日漂移检查完成",
                    state="complete" if result.status in {"success", "warning"} else "error",
                )
                _render_daily_monitor_result(result)
    elif latest_result is not None:
        with st.expander("最近一次每日漂移检查", expanded=False):
            _render_daily_monitor_result(latest_result)


def _render_daily_monitor_result(result: DailyMonitorCheckResult) -> None:
    cols = st.columns(4)
    cols[0].metric("检查状态", _node_status_badge(result.status))
    cols[1].metric("触发事件", len(result.events))
    cols[2].metric("当前持仓数", len(result.current_weights))
    cols[3].metric("目标持仓数", len(result.target_weights))
    st.caption(f"生成时间：{result.generated_at} · 日志：{result.log_path}")
    for warning in result.warnings:
        st.warning(warning)
    events = pd.DataFrame(result.events)
    if events.empty:
        st.success("本次检查没有发现超过阈值的资产偏离。")
    else:
        st.dataframe(
            events,
            use_container_width=True,
            hide_index=True,
            column_config={
                "current_weight": st.column_config.NumberColumn("当前权重", format="%.2%"),
                "target_weight": st.column_config.NumberColumn("目标权重", format="%.2%"),
                "drift": st.column_config.NumberColumn("偏离度", format="%.2%"),
                "threshold": st.column_config.NumberColumn("阈值", format="%.2%"),
            },
        )
        st.download_button(
            "下载本次每日检查事件 CSV",
            events.to_csv(index=False).encode("utf-8-sig"),
            file_name="daily_monitor_events.csv",
            mime="text/csv",
            use_container_width=True,
        )


def _render_backtest_page() -> None:
    st.subheader("历史回测")
    st.write("事件驱动回测：遍历历史价格切片，在调仓日调用策略生成目标权重，并与等权买入持有基准对比。")

    latest_universe = load_latest_a_share_universe()
    default_tickers = _default_tickers_from_universe(
        latest_universe,
        fallback="600519.SH,000858.SZ,000333.SZ,300750.SZ,600036.SH",
        limit=8,
    )
    if latest_universe is not None:
        st.caption(f"默认使用股票池选择器前 8 只：来源 {','.join(latest_universe.sources)}。可在下方手动修改。")
    tickers_text = st.text_area("股票池代码，用英文逗号分隔", value=default_tickers, height=80)
    tickers = _parse_tickers(tickers_text)

    col_a, col_b, col_c = st.columns(3)
    start_date = col_a.text_input("开始日期", value="2022-01-01")
    end_date = col_b.text_input("结束日期", value="")
    strategy_label = col_c.selectbox("策略", options=["动量Top N", "等权再平衡"], index=0)

    col_d, col_e, col_f, col_g = st.columns(4)
    rebalance_days = col_d.number_input("调仓间隔天数", min_value=5, max_value=252, value=21, step=1)
    transaction_cost_bps = col_e.number_input("交易成本 bps", min_value=0.0, max_value=100.0, value=5.0, step=1.0)
    top_n = col_f.number_input("动量Top N", min_value=1, max_value=20, value=min(3, max(1, len(tickers))), step=1)
    lookback_days = col_g.number_input("动量观察天数", min_value=10, max_value=252, value=60, step=5)
    col_h, col_i, col_j = st.columns(3)
    slippage_bps = col_h.number_input("滑点 bps", min_value=0.0, max_value=100.0, value=2.0, step=1.0)
    include_index_benchmark = col_i.checkbox("对比沪深300", value=True)
    index_benchmark_ticker = col_j.text_input("指数代码", value="000300.SH")
    uploaded_news = st.file_uploader("可选：上传历史新闻 CSV（需包含 date,text 两列）", type=["csv"])
    news_csv_path = None
    if uploaded_news is not None:
        try:
            news_csv_path = save_backtest_news_csv(uploaded_news)
            st.success(f"已保存历史新闻CSV：{news_csv_path}")
        except Exception as exc:
            st.error(_friendly_error(exc))

    if st.button("运行历史回测", type="primary", use_container_width=True):
        if len(tickers) < 2:
            st.error("至少需要两个股票代码。")
            return
        strategy = "momentum_top_n" if strategy_label == "动量Top N" else "equal_weight"
        with st.status("正在拉取历史价格并运行回测...", expanded=True) as status:
            st.write("1. 拉取 A 股历史收盘价矩阵")
            st.write("2. 遍历历史切片并按调仓频率生成目标权重")
            st.write("3. 扣除交易成本并计算净值")
            st.write("4. 生成等权买入持有基准")
            try:
                report = run_a_share_backtest(
                    tickers=tickers,
                    market="cn",
                    strategy=strategy,
                    start_date=start_date,
                    end_date=end_date.strip() or None,
                    rebalance_frequency_days=int(rebalance_days),
                    transaction_cost_bps=float(transaction_cost_bps),
                    slippage_bps=float(slippage_bps),
                    momentum_lookback_days=int(lookback_days),
                    top_n=int(top_n),
                    news_csv_path=news_csv_path,
                    include_index_benchmark=include_index_benchmark,
                    index_benchmark_ticker=index_benchmark_ticker,
                )
            except Exception as exc:
                status.update(label="历史回测失败", state="error")
                st.error(_friendly_error(exc))
                return
            status.update(label="历史回测完成", state="complete")
        _render_backtest_report(report)
    else:
        latest = load_latest_backtest_report()
        if latest is not None:
            st.info("显示最近一次历史回测结果。")
            _render_backtest_report(latest)


def _render_todo_acceptance() -> None:
    st.subheader("项目 TODO 验收")
    st.write("完整清单已生成在：`docs/项目完整实现TODO.md`")
    include_live = st.checkbox("运行真实一键全流程 smoke test", value=False)
    if st.button("运行 GUI smoke test", type="primary", use_container_width=True):
        with st.status("正在运行 GUI 验收检查...", expanded=True) as status:
            st.write("1. 检查 dashboard import")
            st.write("2. 检查 LLM/TuShare 配置")
            st.write("3. 检查关键产物文件")
            if include_live:
                st.write("4. 运行低成本一键全流程")
            report = run_gui_smoke_tests(include_live_workflow=include_live)
            status.update(label="GUI smoke test 完成", state="complete" if report.passed else "error")
        _render_smoke_test_report(report)
    else:
        latest_smoke = load_latest_smoke_test_report()
        if latest_smoke is not None:
            with st.expander("最近一次 GUI smoke test", expanded=False):
                _render_smoke_test_report(latest_smoke)

    rows = [
        {"阶段": "1.1 数据访问层", "当前状态": "部分完成", "GUI入口": "A股数据源测试"},
        {"阶段": "1.2 技术指标", "当前状态": "部分完成", "GUI入口": "投资方案生成"},
        {"阶段": "1.3 RAG", "当前状态": "部分完成", "GUI入口": "财报知识库"},
        {"阶段": "2.1 LangGraph", "当前状态": "部分完成", "GUI入口": "多智能体工作流"},
        {"阶段": "2.2 Tool Calling", "当前状态": "部分完成", "GUI入口": "多智能体工作流"},
        {"阶段": "2.3 Persona", "当前状态": "部分完成", "GUI入口": "多智能体工作流"},
        {"阶段": "3.1 BL先验", "当前状态": "部分完成", "GUI入口": "BL模型"},
        {"阶段": "3.2 P/Q/Omega", "当前状态": "部分完成", "GUI入口": "BL模型"},
        {"阶段": "3.3 权重优化", "当前状态": "部分完成", "GUI入口": "投资方案生成"},
        {"阶段": "4.1 漂移监控", "当前状态": "部分完成", "GUI入口": "投资方案生成"},
        {"阶段": "4.2 择时规则", "当前状态": "部分完成", "GUI入口": "投资方案生成"},
        {"阶段": "4.3 风险断路器", "当前状态": "部分完成", "GUI入口": "风险断路器"},
        {"阶段": "5.1 回测", "当前状态": "部分完成", "GUI入口": "历史回测"},
        {"阶段": "5.2 Web GUI", "当前状态": "部分完成", "GUI入口": "当前控制中心"},
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_result_center_page() -> None:
    st.subheader("结果中心")
    st.write("这里汇总最近一次全流程运行产生的研报、权重、订单、风险报告、回测报告和漂移日志。")

    latest_run = load_latest_investment_run()
    if latest_run is None:
        st.info("还没有一键全流程运行记录。请先回到首页点击“一键运行全流程”。")
        return

    _render_investment_run(latest_run, expert_mode=True)
    st.divider()
    _render_artifact_downloads(list_investment_run_artifacts(latest_run.run_id))
    st.divider()
    st.subheader("SQLite 本地审计记录")
    counts = SQLiteStore().table_counts()
    st.json(counts)
    runs = SQLiteStore().load_runs()
    if not runs.empty:
        st.dataframe(runs, use_container_width=True, hide_index=True)


def _render_data_source_health_report(report: DataSourceHealthReport) -> None:
    status_text = "正常" if report.is_healthy else "异常"
    cols = st.columns(5)
    cols[0].metric("整体状态", status_text)
    cols[1].metric("模式", {"auto": "自动兜底", "tushare": "TuShare", "akshare": "AkShare"}[report.mode])
    cols[2].metric("行情标的", report.price_ticker)
    cols[3].metric("财报标的", report.statement_ticker)
    cols[4].metric("检查项", len(report.checks))

    if report.warnings:
        with st.expander("系统提示", expanded=True):
            for warning in report.warnings:
                st.warning(warning)

    rows = [
        {
            "数据源": check.provider,
            "接口": check.interface,
            "标的": check.ticker,
            "状态": "成功" if check.status == "success" else "失败" if check.status == "failed" else "跳过",
            "返回行数": check.row_count,
            "耗时秒": check.elapsed_seconds,
            "错误": check.error or "",
        }
        for check in report.checks
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    preview_tabs = [check for check in report.checks if check.preview]
    if preview_tabs:
        st.subheader("数据预览")
        tabs = st.tabs([f"{check.provider}-{check.interface}" for check in preview_tabs])
        for tab, check in zip(tabs, preview_tabs, strict=True):
            with tab:
                st.caption(f"{check.ticker}，字段：{', '.join(check.columns[:10])}")
                st.dataframe(pd.DataFrame(check.preview), use_container_width=True, hide_index=True)

    with st.expander("原始健康报告"):
        st.json(report.to_dict())


def _render_a_share_universe_result(result: AShareUniverseResult) -> None:
    frame = pd.DataFrame([item.to_dict() for item in result.items])
    cols = st.columns(5)
    cols[0].metric("股票数量", len(result.items))
    cols[1].metric("来源", ",".join(result.sources))
    cols[2].metric("缓存", "命中" if result.cache_hit else "刷新")
    cols[3].metric("缓存年龄", _format_age(result.cache_age_seconds))
    cols[4].metric("最大数量", result.max_count)

    if result.warnings:
        with st.expander("股票池提示", expanded=True):
            for warning in result.warnings:
                st.warning(warning)

    if frame.empty:
        st.warning("股票池为空。")
        return

    st.dataframe(frame, use_container_width=True, hide_index=True)
    st.download_button(
        "下载当前股票池 CSV",
        frame.to_csv(index=False).encode("utf-8-sig"),
        file_name="a_share_universe.csv",
        mime="text/csv",
        use_container_width=True,
    )
    with st.expander("原始股票池 JSON"):
        st.json(result.to_dict())


def _render_client_profile_result(result: ClientProfileResult) -> None:
    constraints = result.constraints
    cols = st.columns(5)
    cols[0].metric("风险类型", result.risk_level)
    cols[1].metric("目标波动上限", f"{(constraints.max_volatility or 0):.2%}")
    cols[2].metric("最低期望收益", f"{(constraints.min_expected_return or 0):.2%}")
    cols[3].metric("单股上限", f"{constraints.max_single_asset_weight:.2%}")
    cols[4].metric("解析方式", "LLM" if result.used_llm else "规则")

    if result.warnings:
        with st.expander("解析提示", expanded=True):
            for warning in result.warnings:
                st.warning(warning)

    st.success(result.explanation)
    st.dataframe(
        pd.DataFrame(
            [
                {"约束": "max_volatility", "含义": "目标年化波动率上限", "数值": constraints.max_volatility},
                {"约束": "min_expected_return", "含义": "最低期望年化收益", "数值": constraints.min_expected_return},
                {"约束": "max_single_asset_weight", "含义": "单只股票最大持仓", "数值": constraints.max_single_asset_weight},
                {"约束": "risk_free_rate", "含义": "无风险利率", "数值": constraints.risk_free_rate},
            ]
        ),
        use_container_width=True,
        hide_index=True,
        column_config={"数值": st.column_config.NumberColumn("数值", format="%.2%")},
    )
    with st.expander("原始客户画像 JSON"):
        st.json(result.to_dict())


def _render_technical_analysis_report(report: TechnicalAnalysisReport) -> None:
    latest = report.latest
    signals = report.signals
    history = pd.DataFrame(report.history)
    cols = st.columns(6)
    cols[0].metric("股票", report.ticker)
    cols[1].metric("收盘价", _fmt_number(latest.get("close")))
    cols[2].metric("RSI 14", _fmt_number(latest.get("rsi_14")))
    cols[3].metric("趋势", _translate_signal(signals.get("trend")))
    cols[4].metric("动量", _translate_signal(signals.get("momentum")))
    cols[5].metric("执行建议", _translate_signal(signals.get("execution_signal")))

    st.success(signals.get("summary", "技术指标计算完成。"))

    if history.empty:
        st.warning("没有可展示的历史指标。")
        return

    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    tabs = st.tabs(["价格与均线", "RSI", "MACD", "指标表", "原始报告"])
    with tabs[0]:
        st.plotly_chart(_technical_price_chart(history), use_container_width=True)
    with tabs[1]:
        st.plotly_chart(_technical_rsi_chart(history), use_container_width=True)
    with tabs[2]:
        st.plotly_chart(_technical_macd_chart(history), use_container_width=True)
    with tabs[3]:
        st.dataframe(
            history.tail(80),
            use_container_width=True,
            hide_index=True,
            column_config={
                "close": st.column_config.NumberColumn("收盘价", format="%.2f"),
                "rsi_14": st.column_config.NumberColumn("RSI", format="%.2f"),
                "volume_ratio": st.column_config.NumberColumn("量比", format="%.2f"),
            },
        )
        st.download_button(
            "下载技术指标 CSV",
            history.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{report.ticker}_technical_indicators.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with tabs[4]:
        st.json(report.to_dict())


def _run_financial_tool_debug_trace(
    *,
    ticker: str,
    market: str,
    limit: int,
    include_statements: bool,
    include_news: bool,
    use_bound_llm: bool,
) -> list[dict[str, Any]]:
    from fin_agent_sakura.tools import create_financial_tool_calling_llm, get_financial_statements, get_recent_news

    trace: list[dict[str, Any]] = []
    if include_statements:
        payload = {
            "ticker": ticker,
            "market": market,
            "statement": "all",
            "period": "annual",
            "limit": limit,
        }
        trace.append(_invoke_tool_for_debug("get_financial_statements", get_financial_statements, payload))

    if include_news:
        payload = {"ticker": ticker, "market": market, "limit": limit}
        trace.append(_invoke_tool_for_debug("get_recent_news", get_recent_news, payload))

    if use_bound_llm:
        started = time.perf_counter()
        try:
            llm = create_financial_tool_calling_llm(max_tokens=300)
            response = llm.invoke(
                f"请判断是否需要调用工具来分析 {ticker}，只做一次最小化测试，不要输出长报告。"
            )
            trace.append(
                {
                    "tool": "bound_llm",
                    "input": {"ticker": ticker, "market": market},
                    "status": "success",
                    "elapsed_seconds": round(time.perf_counter() - started, 2),
                    "output": {
                        "content": getattr(response, "content", ""),
                        "tool_calls": getattr(response, "tool_calls", []),
                    },
                    "error": "",
                }
            )
        except Exception as exc:
            trace.append(
                {
                    "tool": "bound_llm",
                    "input": {"ticker": ticker, "market": market},
                    "status": "failed",
                    "elapsed_seconds": round(time.perf_counter() - started, 2),
                    "output": {},
                    "error": _friendly_error(exc),
                }
            )

    return trace


def _invoke_tool_for_debug(tool_name: str, tool_obj: Any, payload: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        raw = tool_obj.invoke(payload)
        parsed = _safe_json_loads(raw)
        return {
            "tool": tool_name,
            "input": payload,
            "status": "success",
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "output": parsed,
            "error": "",
        }
    except Exception as exc:
        return {
            "tool": tool_name,
            "input": payload,
            "status": "failed",
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "output": {},
            "error": _friendly_error(exc),
        }


def _render_tool_debug_trace(trace: list[dict[str, Any]]) -> None:
    if not trace:
        st.info("请选择至少一个工具。")
        return

    rows = [
        {
            "工具": item["tool"],
            "状态": "成功" if item["status"] == "success" else "失败",
            "耗时秒": item["elapsed_seconds"],
            "输入": item["input"],
            "错误": item["error"],
        }
        for item in trace
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    tabs = st.tabs([item["tool"] for item in trace])
    for tab, item in zip(tabs, trace, strict=True):
        with tab:
            if item["error"]:
                st.error(item["error"])
            st.json(item)


def _render_investment_run(run: InvestmentRun, *, expert_mode: bool, allow_expanders: bool = True) -> None:
    status_label = _run_status_label(run.status)
    cols = st.columns(5)
    cols[0].metric("运行状态", status_label)
    cols[1].metric("步骤数", len(run.steps))
    cols[2].metric("产物数", len(run.artifacts))
    cols[3].metric("候选池", run.max_candidates)
    cols[4].metric("持仓数", run.selected_count)
    st.caption(f"运行ID：{run.run_id}；生成时间：{run.generated_at}")

    if run.warnings:
        if allow_expanders:
            with st.expander("全流程警告", expanded=True):
                for warning in run.warnings:
                    st.warning(warning)
        else:
            for warning in run.warnings:
                st.warning(warning)

    steps = pd.DataFrame([step.to_dict() for step in run.steps])
    if steps.empty:
        st.info("没有可展示的运行步骤。")
    else:
        display = steps[["name", "status", "summary", "elapsed_seconds", "error"]].copy()
        display["status"] = display["status"].map(_run_status_label)
        st.subheader("步骤进度")
        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "name": "步骤",
                "status": "状态",
                "summary": "中文结论",
                "elapsed_seconds": st.column_config.NumberColumn("耗时", format="%.2f 秒"),
                "error": "错误",
            },
        )

    if expert_mode and allow_expanders:
        with st.expander("专家模式：运行 JSON", expanded=False):
            st.json(run.to_dict())


def _render_artifact_downloads(artifacts: list[InvestmentArtifact]) -> None:
    if not artifacts:
        st.info("当前运行没有可下载产物。")
        return

    st.subheader("统一下载")
    frame = pd.DataFrame([artifact.to_dict() for artifact in artifacts])
    st.dataframe(
        frame,
        use_container_width=True,
        hide_index=True,
        column_config={
            "name": "产物",
            "artifact_type": "类型",
            "path": "路径",
            "source_step": "来源步骤",
            "generated_at": "生成时间",
            "description": "说明",
        },
    )

    for artifact in artifacts:
        path = Path(artifact.path)
        if not path.exists() or path.is_dir():
            continue
        data = path.read_bytes()
        st.download_button(
            f"下载：{artifact.name}",
            data,
            file_name=path.name,
            mime=_artifact_mime(artifact),
            use_container_width=True,
        )


def _run_status_label(status: str) -> str:
    return {
        "pending": "等待",
        "running": "运行中",
        "success": "成功",
        "failed": "失败",
        "skipped": "跳过",
    }.get(status, status)


def _artifact_mime(artifact: InvestmentArtifact) -> str:
    if artifact.artifact_type == "csv":
        return "text/csv"
    if artifact.artifact_type == "html":
        return "text/html"
    if artifact.artifact_type == "json":
        return "application/json"
    return "application/octet-stream"


def _sample_value_investor_payload() -> dict[str, Any]:
    return {
        "ticker": "600519.SH",
        "quality_checks": {
            "gross_margin_trend": "stable",
            "gross_margin_evidence": "示例：近年毛利率保持高位。",
            "long_term_debt_to_equity": 0.05,
            "leverage_risk": "low",
            "leverage_evidence": "示例：长期债务压力较低。",
        },
        "revenue_forecasts": [
            {
                "case": "bear",
                "forecast_horizon_years": 5,
                "revenue_growth_rates": [0.03, 0.03, 0.025, 0.025, 0.02],
                "operating_margin": 0.45,
                "capex_to_revenue": 0.04,
                "working_capital_to_revenue": 0.03,
                "rationale": "保守情形下增长放缓。",
            },
            {
                "case": "base",
                "forecast_horizon_years": 5,
                "revenue_growth_rates": [0.06, 0.06, 0.05, 0.05, 0.04],
                "operating_margin": 0.50,
                "capex_to_revenue": 0.035,
                "working_capital_to_revenue": 0.025,
                "rationale": "基准情形下维持稳健增长。",
            },
            {
                "case": "bull",
                "forecast_horizon_years": 5,
                "revenue_growth_rates": [0.09, 0.08, 0.07, 0.06, 0.05],
                "operating_margin": 0.53,
                "capex_to_revenue": 0.03,
                "working_capital_to_revenue": 0.02,
                "rationale": "乐观情形下需求和盈利质量更强。",
            },
        ],
        "wacc_assumptions": {
            "risk_free_rate": 0.025,
            "equity_risk_premium": 0.055,
            "beta": 0.9,
            "cost_of_equity": 0.0745,
            "after_tax_cost_of_debt": 0.03,
            "target_debt_weight": 0.05,
            "target_equity_weight": 0.95,
            "tax_rate": 0.25,
            "wacc": 0.072,
            "confidence": 0.65,
            "notes": "示例 WACC，仅用于 schema 校验。",
        },
        "intrinsic_value_range": {
            "bear_case_value_per_share": 1200.0,
            "base_case_value_per_share": 1600.0,
            "bull_case_value_per_share": 2100.0,
            "currency": "CNY",
            "margin_of_safety": 0.12,
            "terminal_growth_rate": 0.025,
            "terminal_value": 1_500_000_000_000.0,
            "sensitivity_analysis": {
                "wacc_plus_1pct": "估值下调约 12%",
                "terminal_growth_minus_1pct": "估值下调约 8%",
            },
            "valuation_method": "DCF proxy",
        },
        "conclusion": "hold",
        "confidence": 0.62,
        "key_risks": ["估值偏高", "消费需求波动"],
        "reasoning_summary": "示例结构化输出用于验证 schema。",
    }


def _render_smoke_test_report(report: SmokeTestReport) -> None:
    cols = st.columns(3)
    cols[0].metric("验收结果", "通过" if report.passed else "失败")
    cols[1].metric("检查项", len(report.items))
    cols[2].metric("生成时间", report.generated_at[:19])
    frame = pd.DataFrame([item.to_dict() for item in report.items])
    if frame.empty:
        st.info("没有 smoke test 检查项。")
        return
    display = frame[["name", "status", "summary", "elapsed_seconds", "error"]].copy()
    display["status"] = display["status"].map({"success": "成功", "failed": "失败", "skipped": "跳过"}).fillna(display["status"])
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "name": "检查项",
            "status": "状态",
            "summary": "中文结论",
            "elapsed_seconds": st.column_config.NumberColumn("耗时", format="%.2f 秒"),
            "error": "错误",
        },
    )
    with st.expander("smoke test 原始 JSON", expanded=False):
        st.json(report.to_dict())


def _render_result(result: ChinaInvestmentResult) -> None:
    selected = pd.DataFrame(result.selected)
    weights = pd.Series(result.target_weights, dtype="float64")
    current = pd.Series(result.current_weights, dtype="float64")
    alerts = pd.DataFrame(result.drift_alerts)
    orders = pd.DataFrame(result.trade_orders)

    cols = st.columns(5)
    cols[0].metric("运行模式", "真实数据" if result.mode == "live_data" else "离线兜底")
    cols[1].metric("股票池数量", result.universe_size)
    cols[2].metric("最终持仓", len(result.selected))
    cols[3].metric("漂移警报", len(result.drift_alerts))
    cols[4].metric("纸面订单", len(result.trade_orders))

    if result.warnings:
        with st.expander("系统警告", expanded=True):
            for warning in result.warnings:
                st.warning(warning)

    left, right = st.columns([0.9, 1.1])
    with left:
        st.subheader("目标资产配置")
        st.plotly_chart(_pie_chart(weights), use_container_width=True)

    with right:
        st.subheader("候选评分")
        st.plotly_chart(_score_chart(selected), use_container_width=True)

    st.subheader("候选股技术标签")
    technical_columns = [
        column
        for column in [
            "ticker",
            "name",
            "sector",
            "trend_label",
            "rsi_label",
            "rsi_14",
            "volume_signal",
            "volume_ratio",
            "score",
        ]
        if column in selected.columns
    ]
    st.dataframe(
        selected[technical_columns],
        use_container_width=True,
        hide_index=True,
        column_config={
            "ticker": "股票代码",
            "name": "名称",
            "sector": "行业",
            "trend_label": "趋势标签",
            "rsi_label": "RSI标签",
            "rsi_14": st.column_config.NumberColumn("RSI", format="%.2f"),
            "volume_signal": "成交量信号",
            "volume_ratio": st.column_config.NumberColumn("量比", format="%.2f"),
            "score": st.column_config.NumberColumn("综合评分", format="%.4f"),
        },
    )

    st.subheader("目标权重 vs 当前权重")
    allocation = pd.DataFrame(
        {
            "ticker": weights.index,
            "target_weight": weights.values,
            "current_weight": current.reindex(weights.index).fillna(0).values,
        }
    )
    allocation["drift"] = allocation["current_weight"] - allocation["target_weight"]
    threshold = _latest_drift_threshold(alerts)
    allocation["abs_drift"] = allocation["drift"].abs()
    allocation["drift_status"] = allocation["abs_drift"].map(
        lambda value: "超过阈值" if threshold is not None and float(value) > threshold else "正常"
    )
    st.dataframe(
        allocation,
        use_container_width=True,
        hide_index=True,
        column_config={
            "ticker": "股票代码",
            "target_weight": st.column_config.NumberColumn("目标权重", format="%.2%"),
            "current_weight": st.column_config.NumberColumn("当前权重", format="%.2%"),
            "drift": st.column_config.NumberColumn("偏离度", format="%.2%"),
            "abs_drift": st.column_config.NumberColumn("绝对偏离", format="%.2%"),
            "drift_status": "漂移状态",
        },
    )

    tab_report, tab_orders, tab_alerts, tab_raw = st.tabs(["持仓研报", "纸面订单", "警报控制台", "原始结果"])
    with tab_report:
        st.components.v1.html(result.research_report_html, height=520, scrolling=True)
    with tab_orders:
        if orders.empty:
            st.success("当前没有需要执行的纸面订单。")
        else:
            if result.risk_gate:
                _render_risk_gate_report(result.risk_gate)
                st.divider()
            _render_order_explanations(orders, result.risk_gate)
            risk_approved = bool(result.risk_gate and result.risk_gate.get("decision") == "approved")
            if risk_approved:
                st.download_button(
                    "下载已批准纸面订单 CSV",
                    orders.to_csv(index=False).encode("utf-8-sig"),
                    file_name="approved_paper_trade_orders.csv",
                    mime="text/csv",
                )
            else:
                st.warning("订单尚未通过风险断路器，或已被风控拒绝。请到“风险断路器”页查看。")
                st.download_button(
                    "下载研究记录 CSV",
                    orders.to_csv(index=False).encode("utf-8-sig"),
                    file_name="research_only_trade_orders.csv",
                    mime="text/csv",
                )
    with tab_alerts:
        if alerts.empty:
            st.success("当前没有超过阈值的资产偏离。")
        else:
            _render_drift_alerts(alerts)
        _render_rebalance_event_log()
    with tab_raw:
        st.json(result.to_dict())


def _build_passive_health_checks(position_memory: PositionMemory) -> list[HealthCheck]:
    llm = get_llm_config()
    tushare = get_tushare_config()
    position_frame = position_memory.load()
    rag_count = _safe_count_indexed_reports()
    latest_universe = load_latest_a_share_universe()
    latest_risk = load_latest_risk_gate_report()
    latest_backtest = load_latest_backtest_report()
    latest_run = load_latest_investment_run()
    return [
        HealthCheck("LLM", "normal" if llm.api_key else "error", llm.chat_model if llm.api_key else "未配置 API Key"),
        HealthCheck("TuShare", "normal" if tushare.token else "error", tushare.http_url or "未配置 Token"),
        HealthCheck("RAG", "normal" if rag_count > 0 else "warning", f"{rag_count} 只股票已建索引" if rag_count > 0 else "尚未导入财报"),
        HealthCheck(
            "股票池",
            "normal" if latest_universe is not None else "warning",
            f"{len(latest_universe.items)} 只候选" if latest_universe is not None else "尚未生成",
        ),
        HealthCheck(
            "本地仓位",
            "normal" if not position_frame.empty else "warning",
            f"{len(position_frame)} 条持仓" if not position_frame.empty else "尚未保存",
        ),
        HealthCheck("BL", "normal" if Path("data/processed/cache").exists() else "warning", "BL 页面可运行"),
        HealthCheck(
            "风控",
            "normal" if latest_risk is not None else "warning",
            latest_risk.decision if latest_risk is not None else "尚未评估",
        ),
        HealthCheck(
            "回测",
            "normal" if latest_backtest is not None else "warning",
            f"{latest_backtest.start_date} 至 {latest_backtest.end_date}" if latest_backtest is not None else "尚未运行",
        ),
        HealthCheck(
            "GUI导入",
            "normal",
            f"最近全流程 {latest_run.run_id}" if latest_run is not None else "dashboard import 正常",
        ),
    ]


def _render_plan_status_lights() -> None:
    rows = [
        {"任务": "1.1 数据访问层", "状态": "已接入GUI", "入口": "A股数据源测试"},
        {"任务": "1.2 技术指标", "状态": "已接入GUI", "入口": "个股技术分析/投资方案生成"},
        {"任务": "1.3 RAG", "状态": "已接入GUI", "入口": "财报知识库"},
        {"任务": "2.1 LangGraph", "状态": "已接入GUI", "入口": "多智能体工作流"},
        {"任务": "2.2 Tool Calling", "状态": "已接入GUI", "入口": "LLM测试/多智能体工作流"},
        {"任务": "2.3 Persona", "状态": "已接入GUI", "入口": "智能体Persona"},
        {"任务": "3.1 BL先验", "状态": "已接入GUI", "入口": "BL模型"},
        {"任务": "3.2 P/Q/Omega", "状态": "部分完成", "入口": "BL模型"},
        {"任务": "3.3 权重优化", "状态": "已接入GUI", "入口": "客户画像问卷/BL模型"},
        {"任务": "4.1 漂移监控", "状态": "已接入GUI", "入口": "投资方案生成"},
        {"任务": "4.2 择时规则", "状态": "已接入GUI", "入口": "投资方案生成"},
        {"任务": "4.3 风险断路器", "状态": "已接入GUI", "入口": "风险断路器/结果中心"},
        {"任务": "5.1 回测", "状态": "已接入GUI", "入口": "历史回测"},
        {"任务": "5.2 Web GUI", "状态": "已接入GUI", "入口": "首页/结果中心"},
    ]
    frame = pd.DataFrame(rows)
    st.dataframe(frame, use_container_width=True, hide_index=True)


def _render_cost_control_summary() -> None:
    usage = summarize_llm_usage()
    sqlite_counts = SQLiteStore().table_counts()
    cols = st.columns(4)
    cols[0].metric("LLM调用次数", usage["calls"])
    cols[1].metric("估算Token", usage["total_tokens"])
    cols[2].metric("估算成本USD", f"${usage['estimated_cost_usd']:.4f}")
    cols[3].metric("SQLite运行记录", sqlite_counts.get("investment_runs", 0))
    with st.expander("成本与运行记录明细", expanded=False):
        usage_frame = load_llm_usage()
        if usage_frame.empty:
            st.info("还没有记录到 LLM 调用。")
        else:
            st.dataframe(usage_frame.tail(50), use_container_width=True, hide_index=True)
        runs = SQLiteStore().load_runs()
        if not runs.empty:
            st.caption("SQLite 中最近的 InvestmentRun")
            st.dataframe(runs, use_container_width=True, hide_index=True)
        cache_dirs = _cache_directory_status()
        st.caption("本地缓存目录")
        st.dataframe(pd.DataFrame(cache_dirs), use_container_width=True, hide_index=True)


def _cache_directory_status() -> list[dict[str, Any]]:
    paths = [
        ("行情/市值缓存", Path("data/processed/cache")),
        ("技术指标缓存", Path("data/processed/technical_indicators")),
        ("财报PDF", Path("data/reports")),
        ("RAG索引", Path("data/rag_index")),
        ("LLM用量日志", Path("data/processed/llm_usage_log.jsonl")),
    ]
    rows = []
    for name, path in paths:
        if path.is_dir():
            count = len([item for item in path.rglob("*") if item.is_file()])
            size = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
        elif path.exists():
            count = 1
            size = path.stat().st_size
        else:
            count = 0
            size = 0
        rows.append({"缓存": name, "路径": str(path), "存在": path.exists(), "文件数": count, "大小KB": round(size / 1024, 2)})
    return rows


def _safe_count_indexed_reports() -> int:
    try:
        return len(list_indexed_financial_reports())
    except Exception:
        return 0


def _render_health_cards(checks: list[HealthCheck]) -> None:
    cols = st.columns(len(checks))
    for col, check in zip(cols, checks, strict=True):
        col.metric(check.name, _status_label(check.status))
        col.caption(check.detail)


def _render_bl_result(
    prior: Any,
    views: Any,
    posterior_returns: pd.Series,
    posterior_covariance: pd.DataFrame,
    optimized: Any,
    omega: pd.DataFrame | None = None,
) -> None:
    prior_table = pd.DataFrame(
        {
            "ticker": prior.tickers,
            "market_cap": prior.market_caps.reindex(prior.tickers).values,
            "market_weight": prior.market_weights.reindex(prior.tickers).values,
            "prior_return": prior.implied_prior_returns.reindex(prior.tickers).values,
            "posterior_return": posterior_returns.reindex(prior.tickers).values,
            "optimized_weight": optimized.weights.reindex(prior.tickers).fillna(0).values,
        }
    )

    metrics = st.columns(4)
    metrics[0].metric("资产数量", len(prior.tickers))
    metrics[1].metric("目标收益", f"{optimized.expected_return:.2%}")
    metrics[2].metric("目标波动", f"{optimized.volatility:.2%}")
    metrics[3].metric("Sharpe", f"{optimized.sharpe_ratio:.2f}")
    if getattr(optimized, "fallback_used", False):
        st.warning("组合优化使用了 fallback 策略：原始约束可能不可达，已回退到最小波动组合。")
    if getattr(optimized, "diagnostics", None):
        with st.expander("约束冲突解释", expanded=True):
            for item in optimized.diagnostics or []:
                st.write(item)

    tab_alloc, tab_views, tab_matrix, tab_frontier = st.tabs(["权重与收益", "观点矩阵", "协方差矩阵", "有效前沿"])
    with tab_alloc:
        left, right = st.columns([0.95, 1.05])
        with left:
            st.plotly_chart(_pie_chart(optimized.weights[optimized.weights > 0]), use_container_width=True)
        with right:
            st.dataframe(
                prior_table,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "ticker": "股票代码",
                    "market_cap": st.column_config.NumberColumn("总市值", format="%.2f"),
                    "market_weight": st.column_config.NumberColumn("市值权重", format="%.2%"),
                    "prior_return": st.column_config.NumberColumn("先验收益", format="%.2%"),
                    "posterior_return": st.column_config.NumberColumn("后验收益", format="%.2%"),
                    "optimized_weight": st.column_config.NumberColumn("优化权重", format="%.2%"),
                },
            )
        st.plotly_chart(_return_comparison_chart(prior_table), use_container_width=True)

    with tab_views:
        st.caption("P 矩阵")
        st.dataframe(views.picking_matrix, use_container_width=True)
        st.caption("Q 观点向量")
        st.dataframe(views.views_vector.rename("expected_excess_return").to_frame(), use_container_width=True)
        st.caption("观点置信度")
        confidence_frame = pd.DataFrame(
            {
                "view": views.views_vector.index,
                "ticker": views.view_tickers,
                "confidence": views.confidences,
                "source": views.view_sources or ["人工"] * len(views.confidences),
            }
        )
        st.dataframe(
            confidence_frame,
            use_container_width=True,
            hide_index=True,
            column_config={"confidence": st.column_config.NumberColumn("置信度", format="%.0%")},
        )
        if omega is not None:
            st.caption("Omega 不确定性矩阵")
            st.dataframe(omega, use_container_width=True)
            st.plotly_chart(_heatmap(omega, "Omega 不确定性矩阵"), use_container_width=True)

    with tab_matrix:
        st.plotly_chart(_heatmap(prior.covariance_matrix, "先验协方差矩阵"), use_container_width=True)
        st.plotly_chart(_heatmap(posterior_covariance, "后验协方差矩阵"), use_container_width=True)

    with tab_frontier:
        frontier = _build_frontier_points(
            prior.implied_prior_returns,
            prior.covariance_matrix,
            posterior_returns,
            posterior_covariance,
        )
        st.plotly_chart(_frontier_chart(frontier), use_container_width=True)


def _render_agent_analysis_result(result: AgentAnalysisResult) -> None:
    decision = result.final_decision
    status_counts = pd.Series([event.status for event in result.node_events]).value_counts()
    cols = st.columns(5)
    cols[0].metric("标的", result.ticker)
    cols[1].metric("市场", result.market)
    cols[2].metric("最终建议", str(decision.get("action", "hold")))
    cols[3].metric("置信度", f"{float(decision.get('confidence', 0.0)):.0%}")
    cols[4].metric("风险评分", f"{float(decision.get('risk_score', 0.0)):.0%}")

    if result.warnings:
        with st.expander("系统警告", expanded=True):
            for warning in result.warnings:
                st.warning(warning)

    _render_agent_flow(result)

    st.subheader("节点运行状态")
    event_rows = [
        {
            "节点": event.node,
            "状态": event.status,
            "耗时秒": event.elapsed_seconds,
            "摘要": event.summary,
            "错误": event.error or "",
        }
        for event in result.node_events
    ]
    st.dataframe(pd.DataFrame(event_rows), use_container_width=True, hide_index=True)

    st.subheader("最终决策")
    st.json(decision)

    tabs = st.tabs([event.node for event in result.node_events] + ["原始状态"])
    for tab, event in zip(tabs[:-1], result.node_events, strict=True):
        with tab:
            if event.error:
                st.error(event.error)
            st.write(event.summary)
            st.json(event.result)
    with tabs[-1]:
        st.json(result.raw_state)

    if not status_counts.empty:
        st.caption("节点状态统计")
        st.dataframe(status_counts.rename("count").to_frame(), use_container_width=True)


def _render_agent_flow(result: AgentAnalysisResult) -> None:
    st.subheader("工作流流程图")
    events_by_node = {event.node: event for event in result.node_events}
    flow_nodes = [
        ("数据源", "三张财报 / 行情 / 新闻 / RAG"),
        ("基本面分析师", "财报 + RAG"),
        ("情绪分析师", "最近新闻"),
        ("技术面分析师", "RSI / MACD / 均线"),
        ("投资组合经理", "信号融合"),
        ("风控", "RiskManager 下一出口"),
    ]
    cols = st.columns(len(flow_nodes))
    for index, (name, caption) in enumerate(flow_nodes):
        event = events_by_node.get(name)
        if name == "数据源":
            status = "success" if result.node_events else "unknown"
            elapsed = ""
        elif name == "风控":
            status = "warning"
            elapsed = "独立页面执行"
        else:
            status = event.status if event else "unknown"
            elapsed = f"{event.elapsed_seconds:.2f}s" if event else "等待"
        label = _node_status_badge(status)
        cols[index].metric(name, label)
        cols[index].caption(f"{caption} · {elapsed}")
        if index < len(flow_nodes) - 1:
            cols[index].caption("→")


def _node_status_badge(status: str) -> str:
    return {
        "success": "成功",
        "warning": "警告",
        "error": "失败",
        "failed": "失败",
        "unknown": "等待",
    }.get(status, status)


def _render_risk_gate_report(report: dict[str, Any]) -> None:
    approved = report.get("decision") == "approved"
    cols = st.columns(4)
    cols[0].metric("风控结论", "批准" if approved else "拒绝")
    cols[1].metric("VaR", f"{float(report.get('portfolio_var', 0.0)):.2%}")
    cols[2].metric("最大回撤", f"{float(report.get('max_drawdown', 0.0)):.2%}")
    cols[3].metric("报警数量", len(report.get("alerts") or []))

    if approved:
        st.success("风险断路器已批准。本结果仍为纸面订单，真实交易前需要人工确认。")
    else:
        st.error("风险断路器已拒绝。禁止将本批订单作为可执行指令导出。")

    if report.get("warnings"):
        with st.expander("风控警告", expanded=True):
            for warning in report["warnings"]:
                st.warning(warning)

    left, right = st.columns([0.9, 1.1])
    with left:
        proposed = pd.Series(report.get("proposed_weights") or {}, dtype="float64")
        if not proposed.empty:
            st.subheader("风控后组合权重")
            st.plotly_chart(_pie_chart(proposed), use_container_width=True)
    with right:
        alerts = pd.DataFrame(report.get("alerts") or [])
        st.subheader("报警日志")
        if alerts.empty:
            st.success("没有触发风险报警。")
        else:
            st.dataframe(alerts, use_container_width=True, hide_index=True)

    orders = pd.DataFrame(report.get("orders") or [])
    if not orders.empty:
        st.subheader("订单导出")
        if approved:
            st.download_button(
                "下载已批准纸面订单 CSV",
                orders.to_csv(index=False).encode("utf-8-sig"),
                file_name="approved_paper_trade_orders.csv",
                mime="text/csv",
            )
        else:
            st.download_button(
                "下载被拒绝订单报告 CSV",
                orders.to_csv(index=False).encode("utf-8-sig"),
                file_name="rejected_order_report.csv",
                mime="text/csv",
            )
    with st.expander("原始风控报告"):
        st.json(report)


def _render_order_explanations(orders: pd.DataFrame, risk_gate: dict[str, Any] | None) -> None:
    frame = orders.copy()
    if "risk_gate_decision" not in frame.columns:
        frame["risk_gate_decision"] = (risk_gate or {}).get("decision", "not_run")
    if "execution_label" not in frame.columns:
        frame["execution_label"] = frame["risk_gate_decision"].map(
            lambda value: "风控拒绝" if value == "rejected" else "需要人工确认" if value == "approved" else "立即执行"
        )
    columns = [
        column
        for column in [
            "ticker",
            "action",
            "execution_label",
            "target_weight_delta",
            "suggested_batches",
            "batch_weight_delta",
            "technical_signal",
            "technical_rsi_14",
            "technical_momentum",
            "sentiment_score",
            "sentiment_confidence",
            "risk_gate_decision",
            "reason",
        ]
        if column in frame.columns
    ]
    st.subheader("订单决策解释")
    st.dataframe(
        frame[columns],
        use_container_width=True,
        hide_index=True,
        column_config={
            "ticker": "股票代码",
            "action": "方向",
            "execution_label": "执行状态",
            "target_weight_delta": st.column_config.NumberColumn("目标调仓幅度", format="%.2%"),
            "suggested_batches": st.column_config.NumberColumn("建议批次", format="%d"),
            "batch_weight_delta": st.column_config.NumberColumn("单批幅度", format="%.2%"),
            "technical_signal": "技术信号",
            "technical_rsi_14": st.column_config.NumberColumn("RSI", format="%.2f"),
            "technical_momentum": "技术动量",
            "sentiment_score": st.column_config.NumberColumn("情绪分", format="%.2f"),
            "sentiment_confidence": st.column_config.NumberColumn("情绪置信度", format="%.0%"),
            "risk_gate_decision": "风控结果",
            "reason": "规则原因",
        },
    )


def _render_drift_alerts(alerts: pd.DataFrame) -> None:
    frame = alerts.copy()
    columns = [
        column
        for column in [
            "ticker",
            "action",
            "execution_label",
            "current_weight",
            "target_weight",
            "drift",
            "threshold",
            "technical_signal",
            "sentiment_score",
            "decision",
            "reason",
        ]
        if column in frame.columns
    ]
    st.dataframe(
        frame[columns],
        use_container_width=True,
        hide_index=True,
        column_config={
            "ticker": "股票代码",
            "action": "方向",
            "execution_label": "处理状态",
            "current_weight": st.column_config.NumberColumn("当前权重", format="%.2%"),
            "target_weight": st.column_config.NumberColumn("目标权重", format="%.2%"),
            "drift": st.column_config.NumberColumn("偏离度", format="%.2%"),
            "threshold": st.column_config.NumberColumn("阈值", format="%.2%"),
            "technical_signal": "技术信号",
            "sentiment_score": st.column_config.NumberColumn("情绪分", format="%.2f"),
            "decision": "规则结果",
            "reason": "规则原因",
        },
    )


def _render_rebalance_event_log() -> None:
    log = load_rebalance_event_log()
    if log.empty:
        st.info("还没有历史再平衡事件日志。运行一次投资方案后会自动记录。")
        return

    log = log.copy()
    log["generated_at"] = pd.to_datetime(log["generated_at"], errors="coerce")
    log["drift"] = pd.to_numeric(log.get("drift"), errors="coerce")
    st.subheader("历史漂移曲线")
    st.plotly_chart(_drift_history_chart(log), use_container_width=True)

    st.subheader("再平衡事件日志")
    display = log.sort_values("generated_at", ascending=False).head(200)
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.download_button(
        "下载再平衡事件日志 CSV",
        log.to_csv(index=False).encode("utf-8-sig"),
        file_name="rebalance_events.csv",
        mime="text/csv",
    )


def _latest_drift_threshold(alerts: pd.DataFrame) -> float | None:
    if alerts.empty or "threshold" not in alerts.columns:
        return None
    values = pd.to_numeric(alerts["threshold"], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.iloc[-1])


def _drift_history_chart(log: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    frame = log.dropna(subset=["generated_at", "drift"])
    if frame.empty:
        fig.update_layout(height=360, title="暂无可绘制的历史漂移")
        return fig
    for ticker, item in frame.groupby("ticker", sort=False):
        fig.add_trace(
            go.Scatter(
                x=item["generated_at"],
                y=item["drift"].abs(),
                mode="lines+markers",
                name=str(ticker),
                hovertemplate="%{x}<br>绝对漂移 %{y:.2%}",
            )
        )
    fig.update_layout(
        height=380,
        margin=dict(l=8, r=8, t=16, b=8),
        yaxis_title="绝对漂移",
        yaxis_tickformat=".0%",
        xaxis_title="生成时间",
    )
    return fig


def _render_backtest_report(report: BacktestRunReport) -> None:
    cols = st.columns(6)
    cols[0].metric("累计收益", f"{report.cumulative_return:.2%}")
    cols[1].metric("年化收益", f"{report.annualized_return:.2%}")
    cols[2].metric("年化波动", f"{report.annualized_volatility:.2%}")
    cols[3].metric("Sharpe", f"{report.sharpe_ratio:.2f}")
    cols[4].metric("最大回撤", f"{report.max_drawdown:.2%}")
    cols[5].metric("基准收益", f"{report.benchmark_cumulative_return:.2%}")
    if report.index_benchmark_cumulative_return is not None:
        st.metric("沪深300基准收益", f"{report.index_benchmark_cumulative_return:.2%}")

    if report.warnings:
        with st.expander("回测警告", expanded=True):
            for warning in report.warnings:
                st.warning(warning)

    st.subheader("策略 vs 基准净值")
    st.plotly_chart(_backtest_curve_chart(report), use_container_width=True)

    left, right = st.columns([0.95, 1.05])
    with left:
        st.subheader("回测参数")
        st.json(
            {
                "tickers": report.tickers,
                "market": report.market,
                "strategy": report.strategy,
                "start_date": report.start_date,
                "end_date": report.end_date,
            }
        )
    with right:
        st.subheader("基准指标")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "组合": "策略",
                        "累计收益": report.cumulative_return,
                        "Sharpe": report.sharpe_ratio,
                        "最大回撤": report.max_drawdown,
                    },
                    {
                        "组合": "等权买入持有",
                        "累计收益": report.benchmark_cumulative_return,
                        "Sharpe": report.benchmark_sharpe_ratio,
                        "最大回撤": report.benchmark_max_drawdown,
                    },
                    *(
                        [
                            {
                                "组合": "沪深300基准",
                                "累计收益": report.index_benchmark_cumulative_return,
                                "Sharpe": report.index_benchmark_sharpe_ratio,
                                "最大回撤": report.index_benchmark_max_drawdown,
                            }
                        ]
                        if report.index_benchmark_cumulative_return is not None
                        else []
                    ),
                ]
            ),
            use_container_width=True,
            hide_index=True,
            column_config={
                "累计收益": st.column_config.NumberColumn("累计收益", format="%.2%"),
                "Sharpe": st.column_config.NumberColumn("Sharpe", format="%.2f"),
                "最大回撤": st.column_config.NumberColumn("最大回撤", format="%.2%"),
            },
        )

    snapshots = pd.DataFrame(report.snapshots)
    st.subheader("历史调仓记录")
    rebalance_rows = snapshots[snapshots["turnover"] > 0].copy()
    if rebalance_rows.empty:
        st.info("没有检测到调仓记录。")
    else:
        st.dataframe(
            rebalance_rows[["date", "portfolio_value", "turnover", "transaction_cost", "weights"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "portfolio_value": st.column_config.NumberColumn("组合净值", format="%.2f"),
                "turnover": st.column_config.NumberColumn("换手率", format="%.2%"),
                "transaction_cost": st.column_config.NumberColumn("交易成本", format="%.2f"),
            },
        )

    col_a, col_b = st.columns(2)
    col_a.download_button(
        "下载净值曲线 CSV",
        _backtest_curve_frame(report).to_csv(index=False).encode("utf-8-sig"),
        file_name="backtest_equity_curve.csv",
        mime="text/csv",
        use_container_width=True,
    )
    col_b.download_button(
        "下载调仓记录 CSV",
        snapshots.to_csv(index=False).encode("utf-8-sig"),
        file_name="backtest_snapshots.csv",
        mime="text/csv",
        use_container_width=True,
    )
    html_path = Path("data/processed/backtest_latest.html")
    if html_path.exists():
        st.download_button(
            "下载回测报告 HTML",
            html_path.read_bytes(),
            file_name="backtest_report.html",
            mime="text/html",
            use_container_width=True,
        )


def _render_financial_index_info(info: FinancialReportIndexInfo) -> None:
    cols = st.columns(3)
    cols[0].metric("股票代码", info.ticker)
    cols[1].metric("分块数量", info.chunk_count)
    cols[2].metric("PDF来源数", len(info.sources))

    if info.sources:
        st.caption("来源文件")
        st.write(", ".join(info.sources))

    summaries = pd.DataFrame(info.page_summaries)
    if not summaries.empty:
        with st.expander("页摘要预览", expanded=False):
            st.dataframe(summaries, use_container_width=True, hide_index=True)


def _render_financial_context_answer(answer: FinancialContextAnswer) -> None:
    st.caption(f"股票：{answer.ticker}；问题：{answer.query}")
    if not answer.contexts:
        st.info("没有检索到相关段落。")
        return
    matches = answer.matches or [
        {"context": context, "score": None, "source": "", "page": "", "chunk_id": ""}
        for context in answer.contexts
    ]
    summary_rows = [
        {
            "rank": index,
            "score": match.get("score"),
            "source": match.get("source"),
            "page": match.get("page"),
            "chunk_id": match.get("chunk_id"),
        }
        for index, match in enumerate(matches, start=1)
    ]
    st.dataframe(
        pd.DataFrame(summary_rows),
        use_container_width=True,
        hide_index=True,
        column_config={
            "rank": "排名",
            "score": st.column_config.NumberColumn("相似度", format="%.2f"),
            "source": "来源文件",
            "page": "页码",
            "chunk_id": "Chunk ID",
        },
    )
    for index, match in enumerate(matches, start=1):
        score = match.get("score")
        score_text = f" · 相似度 {float(score):.2f}" if score is not None else ""
        source = match.get("source") or "unknown"
        page = match.get("page") or "unknown"
        with st.expander(f"相关段落 {index} · {source} 第 {page} 页{score_text}", expanded=index == 1):
            st.text(str(match.get("context") or "")[:6000])


def _positions_editor(frame: pd.DataFrame, *, key: str) -> pd.DataFrame:
    return st.data_editor(
        frame,
        num_rows="dynamic",
        use_container_width=True,
        key=key,
        column_config={
            "ticker": st.column_config.TextColumn("股票代码"),
            "name": st.column_config.TextColumn("名称"),
            "shares": st.column_config.NumberColumn("股数", min_value=0.0),
            "market_value": st.column_config.NumberColumn("市值", min_value=0.0),
            "weight": st.column_config.NumberColumn("权重", min_value=0.0, max_value=100.0),
        },
    )


def _status_label(status: HealthStatus) -> str:
    return {
        "normal": "正常",
        "warning": "待完善",
        "error": "异常",
        "unknown": "未知",
    }[status]


def _friendly_error(exc: Exception) -> str:
    message = str(exc).strip()
    if not message:
        message = type(exc).__name__
    return f"{type(exc).__name__}: {message}"


def _safe_json_loads(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        import json

        return json.loads(value)
    except Exception:
        return value


def _extract_first_ticker(value: str) -> str | None:
    import re

    text = value.upper()
    a_share = re.search(r"(?<!\d)(?:SH|SZ|BJ)?(\d{6})(?:\.(?:SH|SZ|BJ))?(?!\d)", text)
    if a_share:
        raw = a_share.group(0)
        code = a_share.group(1)
        if "." in raw:
            return raw
        if raw.startswith(("SH", "SZ", "BJ")):
            return f"{code}.{raw[:2]}"
        if code.startswith(("6", "9")):
            return f"{code}.SH"
        if code.startswith(("4", "8")):
            return f"{code}.BJ"
        return f"{code}.SZ"

    us_ticker = re.search(r"\b[A-Z]{1,5}(?:\.[A-Z])?\b", text)
    if us_ticker:
        token = us_ticker.group(0)
        if token not in {"A", "I", "LLM", "RAG", "RSI", "MACD", "BL", "GUI"}:
            return token
    return None


def _parse_tickers(value: str) -> list[str]:
    tickers = [part.strip().upper() for part in value.replace("\n", ",").split(",")]
    return [ticker for ticker in tickers if ticker]


def _default_tickers_from_universe(result: AShareUniverseResult | None, *, fallback: str, limit: int) -> str:
    if result is None or not result.tickers:
        return fallback
    return ",".join(result.tickers[:limit])


def _universe_result_to_candidates(result: AShareUniverseResult) -> list[StockCandidate]:
    return [
        StockCandidate(ticker=item.ticker, name=item.name or item.ticker, sector=item.industry or item.board or "未知")
        for item in result.items
    ]


def _default_view_frame(tickers: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": ticker,
                "expected_excess_return": 0.03 + idx * 0.005,
                "confidence": max(0.35, 0.75 - idx * 0.07),
                "source": "人工",
            }
            for idx, ticker in enumerate(tickers)
        ]
    )


def _default_relative_view_frame(tickers: list[str]) -> pd.DataFrame:
    if len(tickers) < 2:
        return pd.DataFrame(columns=["long_ticker", "short_ticker", "relative_excess_return", "confidence", "source"])
    return pd.DataFrame(
        [
            {
                "long_ticker": tickers[0],
                "short_ticker": tickers[1],
                "relative_excess_return": 0.03,
                "confidence": 0.65,
                "source": "人工",
            }
        ]
    )


def _views_frame_to_payload(frame: pd.DataFrame, tickers: list[str]) -> dict[str, dict[str, float]]:
    payload: dict[str, dict[str, float]] = {}
    universe = set(tickers)
    for _, row in frame.iterrows():
        ticker = str(row.get("ticker", "")).strip().upper()
        if not ticker or ticker not in universe:
            continue
        payload[ticker] = {
            "expected_excess_return": float(row.get("expected_excess_return", 0.0)),
            "confidence": float(row.get("confidence", 0.5)),
        }
    if not payload:
        raise ValueError("至少需要一个有效观点。")
    return payload


def _views_frame_to_bl_views(frame: pd.DataFrame, tickers: list[str], *, mode: str) -> BlackLittermanViews:
    if mode == "相对观点":
        return _relative_views_frame_to_bl_views(frame, tickers)
    payload = _views_frame_to_payload(frame, tickers)
    views = build_absolute_views_from_llm(payload, tickers)
    sources = [
        str(row.get("source") or "人工")
        for _, row in frame.iterrows()
        if str(row.get("ticker", "")).strip().upper() in views.view_tickers
    ]
    return _attach_view_metadata(views, sources=sources)


def _relative_views_frame_to_bl_views(frame: pd.DataFrame, tickers: list[str]) -> BlackLittermanViews:
    universe = [ticker.upper().strip() for ticker in tickers if ticker.strip()]
    ticker_to_column = {ticker: index for index, ticker in enumerate(universe)}
    rows: list[list[float]] = []
    q_values: list[float] = []
    confidences: list[float] = []
    view_tickers: list[str] = []
    sources: list[str] = []
    for _, row in frame.iterrows():
        long_ticker = str(row.get("long_ticker", "")).strip().upper()
        short_ticker = str(row.get("short_ticker", "")).strip().upper()
        if not long_ticker or not short_ticker:
            continue
        if long_ticker not in ticker_to_column or short_ticker not in ticker_to_column:
            continue
        if long_ticker == short_ticker:
            continue
        picking = [0.0] * len(universe)
        picking[ticker_to_column[long_ticker]] = 1.0
        picking[ticker_to_column[short_ticker]] = -1.0
        rows.append(picking)
        q_values.append(float(row.get("relative_excess_return", 0.0)))
        confidences.append(float(row.get("confidence", 0.5)))
        view_tickers.append(f"{long_ticker}>{short_ticker}")
        sources.append(str(row.get("source") or "人工"))
    if not rows:
        raise ValueError("相对观点至少需要一行有效的看多股票、看空股票和相对收益。")
    view_index = [f"relative_view_{idx}_{ticker}" for idx, ticker in enumerate(view_tickers)]
    views = BlackLittermanViews(
        tickers=universe,
        view_tickers=view_tickers,
        picking_matrix=pd.DataFrame(rows, index=view_index, columns=universe, dtype="float64"),
        views_vector=pd.Series(q_values, index=view_index, name="views", dtype="float64"),
        confidences=confidences,
        view_sources=sources,
    )
    return views


def _attach_view_metadata(views: BlackLittermanViews, *, sources: list[str]) -> BlackLittermanViews:
    metadata_sources = sources[: len(views.confidences)]
    if len(metadata_sources) < len(views.confidences):
        metadata_sources.extend(["人工"] * (len(views.confidences) - len(metadata_sources)))
    object.__setattr__(views, "view_sources", metadata_sources)
    return views


def _pie_chart(weights: pd.Series) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Pie(
                labels=weights.index,
                values=weights.values,
                hole=0.42,
                textinfo="label+percent",
                sort=False,
            )
        ]
    )
    fig.update_layout(height=420, margin=dict(l=8, r=8, t=8, b=8), showlegend=False)
    return fig


def _return_comparison_chart(table: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="先验收益",
            x=table["ticker"],
            y=table["prior_return"],
            marker_color="#5B677A",
        )
    )
    fig.add_trace(
        go.Bar(
            name="后验收益",
            x=table["ticker"],
            y=table["posterior_return"],
            marker_color="#2A9D8F",
        )
    )
    fig.update_layout(
        height=360,
        barmode="group",
        margin=dict(l=8, r=8, t=16, b=8),
        yaxis_tickformat=".0%",
        yaxis_title="年化收益",
    )
    return fig


def _score_chart(selected: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=selected["score"],
            y=selected["ticker"] + " " + selected["name"],
            orientation="h",
            marker_color="#2A9D8F",
        )
    )
    fig.update_layout(
        height=420,
        margin=dict(l=8, r=8, t=8, b=8),
        xaxis_title="综合评分",
        yaxis_title="",
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def _heatmap(matrix: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=matrix.values,
            x=matrix.columns,
            y=matrix.index,
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(tickformat=".2f"),
        )
    )
    fig.update_layout(height=420, title=title, margin=dict(l=8, r=8, t=48, b=8))
    return fig


def _build_frontier_points(
    prior_returns: pd.Series,
    prior_covariance: pd.DataFrame,
    posterior_returns: pd.Series,
    posterior_covariance: pd.DataFrame,
) -> pd.DataFrame:
    frames = []
    for label, returns, covariance in [
        ("市场先验", prior_returns, prior_covariance),
        ("BL后验", posterior_returns, posterior_covariance),
    ]:
        aligned_returns = returns.dropna().astype("float64")
        aligned_covariance = covariance.loc[aligned_returns.index, aligned_returns.index].astype("float64")
        min_ret = float(aligned_returns.min())
        max_ret = float(aligned_returns.max())
        targets = pd.Series([min_ret + (max_ret - min_ret) * idx / 16 for idx in range(2, 15)])
        for target in targets:
            try:
                from pypfopt.efficient_frontier import EfficientFrontier

                ef = EfficientFrontier(aligned_returns, aligned_covariance)
                ef.efficient_return(float(target))
                exp_ret, vol, sharpe = ef.portfolio_performance()
                frames.append(
                    {
                        "frontier": label,
                        "expected_return": float(exp_ret),
                        "volatility": float(vol),
                        "sharpe": float(sharpe),
                    }
                )
            except Exception:
                continue
    return pd.DataFrame(frames)


def _frontier_chart(frontier: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if frontier.empty:
        fig.update_layout(height=420, title="有效前沿暂无可用点")
        return fig

    colors = {"市场先验": "#5B677A", "BL后验": "#2A9D8F"}
    for label, frame in frontier.groupby("frontier", sort=False):
        fig.add_trace(
            go.Scatter(
                x=frame["volatility"],
                y=frame["expected_return"],
                mode="lines+markers",
                name=label,
                marker=dict(size=7),
                line=dict(width=3, color=colors.get(label, "#2A9D8F")),
                customdata=frame["sharpe"],
                hovertemplate="波动 %{x:.2%}<br>收益 %{y:.2%}<br>Sharpe %{customdata:.2f}",
            )
        )
    fig.update_layout(
        height=460,
        margin=dict(l=8, r=8, t=16, b=8),
        xaxis_title="年化波动率",
        yaxis_title="年化预期收益",
        xaxis_tickformat=".0%",
        yaxis_tickformat=".0%",
    )
    return fig


def _backtest_curve_frame(report: BacktestRunReport) -> pd.DataFrame:
    strategy = pd.DataFrame(report.equity_curve)
    benchmark = pd.DataFrame(report.benchmark_curve)
    if strategy.empty or benchmark.empty:
        return pd.DataFrame(columns=["date", "strategy_value", "benchmark_value"])
    merged = strategy.merge(benchmark, on="date", how="outer").sort_values("date")
    index_benchmark = pd.DataFrame(report.index_benchmark_curve)
    if not index_benchmark.empty:
        merged = merged.merge(index_benchmark, on="date", how="outer").sort_values("date")
    merged["strategy_value"] = pd.to_numeric(merged["strategy_value"], errors="coerce")
    merged["benchmark_value"] = pd.to_numeric(merged["benchmark_value"], errors="coerce")
    if "index_benchmark_value" in merged.columns:
        merged["index_benchmark_value"] = pd.to_numeric(merged["index_benchmark_value"], errors="coerce")
    return merged.ffill()


def _backtest_curve_chart(report: BacktestRunReport) -> go.Figure:
    curve = _backtest_curve_frame(report)
    fig = go.Figure()
    if curve.empty:
        fig.update_layout(height=420, title="暂无净值曲线")
        return fig

    fig.add_trace(
        go.Scatter(
            x=curve["date"],
            y=curve["strategy_value"],
            mode="lines",
            name="策略",
            line=dict(color="#2A9D8F", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=curve["date"],
            y=curve["benchmark_value"],
            mode="lines",
            name="等权买入持有",
            line=dict(color="#5B677A", width=2),
        )
    )
    if "index_benchmark_value" in curve.columns:
        fig.add_trace(
            go.Scatter(
                x=curve["date"],
                y=curve["index_benchmark_value"],
                mode="lines",
                name="沪深300",
                line=dict(color="#8E6C88", width=2, dash="dash"),
            )
        )
    fig.update_layout(
        height=460,
        margin=dict(l=8, r=8, t=16, b=8),
        xaxis_title="日期",
        yaxis_title="组合价值",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _technical_price_chart(history: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if {"open", "high", "low", "close"}.issubset(history.columns):
        fig.add_trace(
            go.Candlestick(
                x=history["date"],
                open=pd.to_numeric(history["open"], errors="coerce"),
                high=pd.to_numeric(history["high"], errors="coerce"),
                low=pd.to_numeric(history["low"], errors="coerce"),
                close=pd.to_numeric(history["close"], errors="coerce"),
                name="K线",
                increasing_line_color="#2A9D8F",
                decreasing_line_color="#C44536",
            )
        )
    else:
        fig.add_trace(go.Scatter(x=history["date"], y=history["close"], mode="lines", name="收盘价"))

    for column, name, color in [
        ("ma_50", "MA50", "#5B677A"),
        ("ma_200", "MA200", "#8E6C88"),
        ("bb_upper", "布林上轨", "#E9C46A"),
        ("bb_middle", "布林中轨", "#A8A8A8"),
        ("bb_lower", "布林下轨", "#E9C46A"),
    ]:
        if column in history.columns:
            fig.add_trace(
                go.Scatter(
                    x=history["date"],
                    y=pd.to_numeric(history[column], errors="coerce"),
                    mode="lines",
                    name=name,
                    line=dict(width=1.6, color=color, dash="dot" if column.startswith("bb_") else None),
                )
            )
    fig.update_layout(
        height=520,
        margin=dict(l=8, r=8, t=16, b=8),
        xaxis_title="日期",
        yaxis_title="价格",
        xaxis_rangeslider_visible=False,
    )
    return fig


def _technical_rsi_chart(history: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history["date"],
            y=pd.to_numeric(history.get("rsi_14"), errors="coerce"),
            mode="lines",
            name="RSI 14",
            line=dict(color="#2A9D8F", width=2.4),
        )
    )
    fig.add_hline(y=70, line_dash="dash", line_color="#C44536", annotation_text="超买")
    fig.add_hline(y=30, line_dash="dash", line_color="#457B9D", annotation_text="超卖")
    fig.update_layout(height=360, margin=dict(l=8, r=8, t=16, b=8), yaxis_range=[0, 100])
    return fig


def _technical_macd_chart(history: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=history["date"],
            y=pd.to_numeric(history.get("macd_hist"), errors="coerce"),
            name="MACD柱",
            marker_color="#A8A8A8",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=history["date"],
            y=pd.to_numeric(history.get("macd"), errors="coerce"),
            mode="lines",
            name="MACD",
            line=dict(color="#2A9D8F", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=history["date"],
            y=pd.to_numeric(history.get("macd_signal"), errors="coerce"),
            mode="lines",
            name="Signal",
            line=dict(color="#C44536", width=2),
        )
    )
    fig.update_layout(height=380, margin=dict(l=8, r=8, t=16, b=8), xaxis_title="日期")
    return fig


def _progress_chart(progress: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=progress["done"],
            y=progress["stage"],
            orientation="h",
            marker_color=["#2A9D8F", "#E9C46A", "#E9C46A", "#F4A261", "#F4A261"],
            text=[f"{value:.0%}" for value in progress["done"]],
            textposition="auto",
        )
    )
    fig.update_layout(
        height=360,
        margin=dict(l=8, r=8, t=8, b=8),
        xaxis=dict(range=[0, 1], tickformat=".0%"),
        yaxis_title="",
        xaxis_title="完成度",
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def _fmt_number(value: Any) -> str:
    try:
        number = float(value)
        return f"{number:.2f}"
    except (TypeError, ValueError):
        return "N/A"


def _format_age(age_seconds: float | None) -> str:
    if age_seconds is None:
        return "N/A"
    if age_seconds < 60:
        return f"{age_seconds:.0f}秒"
    if age_seconds < 3600:
        return f"{age_seconds / 60:.1f}分"
    if age_seconds < 86400:
        return f"{age_seconds / 3600:.1f}小时"
    return f"{age_seconds / 86400:.1f}天"


def _translate_signal(value: Any) -> str:
    mapping = {
        "uptrend": "趋势向上",
        "downtrend": "趋势向下",
        "mixed": "震荡",
        "unknown": "未知",
        "bullish": "偏多",
        "bearish": "偏空",
        "buy_watch": "可关注买入",
        "defer_buy": "推迟买入",
        "risk_reduce": "降低风险",
        "hold": "观望",
        "expanding": "放量",
        "contracting": "缩量",
        "normal": "正常",
        "unavailable": "不可用",
    }
    return mapping.get(str(value), str(value))


if __name__ == "__main__":
    main()
