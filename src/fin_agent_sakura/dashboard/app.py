"""Streamlit investment-advisory dashboard."""

from __future__ import annotations

import math
from dataclasses import asdict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fin_agent_sakura.portfolio import parse_client_constraints_from_text


def main() -> None:
    st.set_page_config(page_title="Sakura 投顾仪表盘", layout="wide")
    st.title("Sakura 智能投顾仪表盘")

    weights = _sample_weights()
    frontier = _sample_frontier()
    alerts = _sample_alerts()

    with st.sidebar:
        st.header("客户风险问卷")
        profile_text = st.text_area(
            "风险偏好描述",
            value="我是保守型投资者，期望跑赢通胀即可",
            height=110,
        )
        investment_horizon = st.selectbox("投资期限", ["1-3 年", "3-5 年", "5 年以上"], index=1)
        liquidity_need = st.select_slider("流动性需求", options=["低", "中", "高"], value="中")
        loss_tolerance = st.slider("可接受最大回撤", min_value=5, max_value=40, value=15, step=1)
        constraints = parse_client_constraints_from_text(profile_text)
        st.caption("解析后的约束")
        st.json(
            {
                **asdict(constraints),
                "investment_horizon": investment_horizon,
                "liquidity_need": liquidity_need,
                "loss_tolerance_pct": loss_tolerance,
            }
        )

    metric_cols = st.columns(4)
    metric_cols[0].metric("组合资产数", str(len(weights)))
    metric_cols[1].metric("目标年化收益", f"{constraints.min_expected_return or 0:.1%}")
    metric_cols[2].metric("目标波动率上限", f"{constraints.max_volatility or 0:.1%}")
    metric_cols[3].metric("单资产上限", f"{constraints.max_single_asset_weight:.1%}")

    left, right = st.columns([0.95, 1.05])
    with left:
        st.subheader("投资组合资产配置")
        st.plotly_chart(_portfolio_pie(weights), use_container_width=True)

    with right:
        st.subheader("BL 调整前后有效前沿")
        st.plotly_chart(_efficient_frontier_chart(frontier), use_container_width=True)

    report_col, alert_col = st.columns([1.15, 0.85])
    with report_col:
        st.subheader("大语言模型持仓研报")
        st.components.v1.html(_sample_research_report_html(), height=460, scrolling=True)

    with alert_col:
        st.subheader("每日资产偏离度警报")
        _render_alert_console(alerts)


def _sample_weights() -> pd.Series:
    return pd.Series(
        {
            "AAPL": 0.24,
            "MSFT": 0.22,
            "NVDA": 0.18,
            "GOOGL": 0.14,
            "BRK.B": 0.12,
            "Cash": 0.10,
        },
        dtype="float64",
    )


def _sample_frontier() -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for label, return_shift, vol_shift in [
        ("MVO Prior", 0.00, 0.015),
        ("BL Posterior", 0.012, -0.004),
    ]:
        for step in range(28):
            volatility = 0.06 + step * 0.006 + vol_shift
            expected_return = 0.025 + math.sqrt(step + 1) * 0.018 + return_shift
            rows.append(
                {
                    "frontier": label,
                    "volatility": volatility,
                    "expected_return": expected_return,
                    "sharpe": expected_return / volatility,
                }
            )
    return pd.DataFrame(rows)


def _sample_alerts() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "current_weight": 0.31,
                "target_weight": 0.24,
                "drift": 0.07,
                "status": "需要再平衡",
            },
            {
                "ticker": "MSFT",
                "current_weight": 0.20,
                "target_weight": 0.22,
                "drift": -0.02,
                "status": "正常",
            },
            {
                "ticker": "NVDA",
                "current_weight": 0.11,
                "target_weight": 0.18,
                "drift": -0.07,
                "status": "等待择时确认",
            },
        ]
    )


def _portfolio_pie(weights: pd.Series) -> go.Figure:
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
    fig.update_layout(margin=dict(l=8, r=8, t=8, b=8), height=410, showlegend=False)
    return fig


def _efficient_frontier_chart(frontier: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    colors = {"MVO Prior": "#5B677A", "BL Posterior": "#2A9D8F"}
    for label, frame in frontier.groupby("frontier", sort=False):
        fig.add_trace(
            go.Scatter(
                x=frame["volatility"],
                y=frame["expected_return"],
                mode="lines+markers",
                name=label,
                marker=dict(size=6),
                line=dict(width=3, color=colors[label]),
                hovertemplate="波动率 %{x:.1%}<br>收益 %{y:.1%}<br>Sharpe %{customdata:.2f}",
                customdata=frame["sharpe"],
            )
        )
    fig.update_layout(
        height=410,
        margin=dict(l=8, r=8, t=8, b=8),
        xaxis_title="年化波动率",
        yaxis_title="年化预期收益",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")
    return fig


def _sample_research_report_html() -> str:
    return """
    <section style="font-family: Arial, sans-serif; line-height: 1.55; padding: 4px 10px;">
      <h2 style="margin: 0 0 8px 0;">AAPL 持仓研报摘要</h2>
      <p><strong>结论：</strong>维持核心持仓，短期因权重漂移偏高建议分批再平衡。</p>
      <p><strong>基本面：</strong>收入结构稳健，服务业务毛利率对整体盈利质量形成支撑；长期债务股本比处于可管理区间。</p>
      <p><strong>估值：</strong>DCF 基准情形显示内在价值区间仍有安全边际，但当前仓位已高于 BL 目标权重。</p>
      <p><strong>技术面：</strong>价格处于 50 日均线上方，趋势仍偏强；若 RSI 进入超买区间，则延后新增买入。</p>
      <p><strong>风险：</strong>硬件换机周期、监管政策、汇率和估值倍数收缩是主要风险源。</p>
    </section>
    """


def _render_alert_console(alerts: pd.DataFrame) -> None:
    threshold = st.slider("偏离度阈值", min_value=1, max_value=15, value=5, step=1) / 100
    filtered = alerts.assign(abs_drift=alerts["drift"].abs())
    urgent = filtered[filtered["abs_drift"] > threshold]
    if urgent.empty:
        st.success("当前无资产超过偏离度阈值")
    else:
        st.warning(f"{len(urgent)} 个资产超过阈值，需要检查再平衡")

    st.dataframe(
        filtered[["ticker", "current_weight", "target_weight", "drift", "status"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "ticker": "资产",
            "current_weight": st.column_config.NumberColumn("当前权重", format="%.1f%%"),
            "target_weight": st.column_config.NumberColumn("目标权重", format="%.1f%%"),
            "drift": st.column_config.NumberColumn("偏离度", format="%.1f%%"),
            "status": "状态",
        },
    )


if __name__ == "__main__":
    main()

