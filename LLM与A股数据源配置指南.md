# LLM 与 A 股数据源配置指南

这份文档说明如何把当前项目真正接入大模型，并配置 A 股数据源用于实际研究流程。

## 1. 激活环境

项目已经创建了 conda 环境：

```powershell
conda activate fin-agent-sakura
```

如果需要重新安装核心依赖：

```powershell
pip install -e ".[market-data,rag,agents,tools,portfolio,dashboard]"
```

说明：`pandas-ta` 在当前 Python 3.11 + pip 索引下没有可用包，但项目的 `TechnicalIndicators(engine="pandas")` 已经有纯 pandas 后备实现，可以先正常使用。

## 2. 配置 OpenAI LLM

当前项目里有两个主要 LLM 入口：

- `fin_agent_sakura.tools.create_financial_tool_calling_llm`
- `fin_agent_sakura.agents.build_value_investor_chain`

你需要至少配置：

```powershell
$env:OPENAI_BASE_URL="https://aiapi.clzjwl.cn/v1"
$env:OPENAI_API_KEY="sk-你的OpenAI API Key"
$env:OPENAI_CHAT_MODEL="gpt-5.5"
$env:OPENAI_EMBEDDING_MODEL="text-embedding-3-small"
```

也可以在项目根目录创建 `.env`：

```text
OPENAI_BASE_URL=https://aiapi.clzjwl.cn/v1
OPENAI_API_KEY=sk-你的OpenAI API Key
OPENAI_CHAT_MODEL=gpt-5.5
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

项目会自动读取 `.env`。`.env` 已被 `.gitignore` 忽略，不会进入 Git。

推荐模型选择：

- `gpt-5.5`：如果你的 OpenAI 账号/API 项目已经有该模型权限，可以按你的要求直接使用这个模型名。
- `gpt-5.2`：当前 OpenAI 官方模型列表中可见的高能力 GPT-5 系列模型之一，可作为 `gpt-5.5` 不可用时的备选。
- `gpt-4.1-mini`：适合作为默认 tool calling 模型，速度和成本较平衡。
- `gpt-4.1`：更强的非推理模型，适合更复杂的研报分析。
- `gpt-5-mini` 或更高版本：如果你的账号可用，适合更复杂的多步骤代理任务。

项目代码会通过 [config.py](../src/fin_agent_sakura/config.py) 读取：

```python
OPENAI_BASE_URL
OPENAI_API_KEY
OPENAI_CHAT_MODEL
OPENAI_EMBEDDING_MODEL
```

因此建议显式设置 `OPENAI_CHAT_MODEL` 和 `OPENAI_BASE_URL`，不要依赖默认值。如果运行时报 `model not found`，说明当前网关或 API 项目没有该模型权限，请把 `OPENAI_CHAT_MODEL` 临时切到你账号可用的模型。

## 3. 配置 Embedding 模型

RAG 模块默认使用：

```python
text-embedding-3-small
```

配置位置在 [financial_context.py](../src/fin_agent_sakura/rag/financial_context.py)：

```python
FinancialRAGConfig(embedding_model="text-embedding-3-small")
```

如果你处理大量中文年报，且更重视召回质量，可以改成：

```python
from fin_agent_sakura.rag import FinancialRAGConfig

config = FinancialRAGConfig(
    embedding_model="text-embedding-3-large",
)
```

`text-embedding-3-small` 成本更低，适合先跑通；`text-embedding-3-large` 召回质量更强。

## 4. 配置 A 股数据源

项目目前的 A 股数据路径：

- OHLCV：优先使用 AkShare，不需要 token。
- 财务报表：如果设置了 `TUSHARE_TOKEN`，优先使用 TuShare；否则回退 AkShare 新浪财报接口。

### 4.1 AkShare

AkShare 已经安装在 `fin-agent-sakura` 环境中。它通常不需要 API token。

代码入口：

```python
from fin_agent_sakura.data import MarketDataClientFactory

client = MarketDataClientFactory.get_client("cn")
```

支持 ticker 写法：

- `600519.SH`
- `000001.SZ`
- `600519`
- `SZ000001`

建议统一使用 TuShare 风格：

```text
600519.SH
000001.SZ
300750.SZ
```

### 4.2 TuShare

TuShare 用于更稳定地拉取 A 股财务报表。你需要到 TuShare 官网申请 token，然后设置环境变量：

```powershell
$env:TUSHARE_TOKEN="你的TuShare token"
```

设置后，以下方法会优先使用 TuShare：

```python
await client.fetch_balance_sheet("600519.SH")
await client.fetch_cash_flow("600519.SH")
await client.fetch_income_statement("600519.SH")
```

如果没有设置 `TUSHARE_TOKEN`，项目会回退 AkShare。

## 5. A 股数据源 Smoke Test

新建一个临时脚本，例如 `scripts/a_share_smoke_test.py`：

```python
import asyncio

from fin_agent_sakura.data import MarketDataClientFactory


async def main() -> None:
    client = MarketDataClientFactory.get_client("cn")

    prices = await client.fetch_ohlcv(
        "600519.SH",
        start="2024-01-01",
        end="2024-12-31",
    )
    print("OHLCV:")
    print(prices.head())

    income = await client.fetch_income_statement(
        "600519.SH",
        period="annual",
        limit=3,
    )
    print("Income statement:")
    print(income.head())


asyncio.run(main())
```

运行：

```powershell
conda activate fin-agent-sakura
python scripts/a_share_smoke_test.py
```

如果 OHLCV 成功但财报失败，优先检查：

1. 是否设置了 `TUSHARE_TOKEN`。
2. TuShare token 权限是否足够。
3. AkShare 的网页接口是否临时变动。

## 6. 配置财务工具调用 LLM

项目已经把财报函数封装成 LangChain tool：

```python
from fin_agent_sakura.tools import create_financial_tool_calling_llm

llm = create_financial_tool_calling_llm()
response = llm.invoke(
    "请调用工具获取 600519.SH 的最近三年利润表，市场为 cn，然后总结收入和利润趋势。"
)

print(response)
```

注意：当前 `get_recent_news` 只实现了美股新闻，A 股新闻会返回 warning。A 股新闻后续建议接入：

- AkShare 东方财富新闻接口
- 财联社/同花顺/雪球等合规数据源
- 你自己的新闻数据库

## 7. 配置价值投资者智能体

价值投资者智能体使用 Pydantic 结构化输出，适合做基本面分析。

```python
from langchain_openai import ChatOpenAI

from fin_agent_sakura.agents import build_value_investor_chain
from fin_agent_sakura.tools.financial_tools import get_financial_statements


financial_json = get_financial_statements.invoke(
    {
        "ticker": "600519.SH",
        "market": "cn",
        "statement": "all",
        "period": "annual",
        "limit": 5,
    }
)

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
)

chain = build_value_investor_chain(llm)

result = chain.invoke(
    {
        "ticker": "600519.SH",
        "financial_json": financial_json,
        "context": "这是 A 股白酒行业龙头，需要重点检查收入质量、毛利率趋势、长期债务股本比和估值安全边际。",
    }
)

print(result.model_dump_json(indent=2, ensure_ascii=False))
```

输出会被解析为 `ValueInvestorAnalysis`，包含：

- 毛利率趋势检查
- 长期债务股本比检查
- 收入预测
- WACC 假设
- 内在价值区间
- 结论与置信度

## 8. 配置 A 股年报 RAG

如果你有本地 PDF 年报，可以先建立索引：

```python
from fin_agent_sakura.rag import (
    FinancialRAGConfig,
    ingest_financial_report,
    retrieve_financial_context,
)

config = FinancialRAGConfig(
    embedding_model="text-embedding-3-small",
)

ingest_financial_report(
    "reports/600519_2024_annual_report.pdf",
    ticker="600519.SH",
    config=config,
)

contexts = retrieve_financial_context(
    query="公司的收入增长、毛利率变化、渠道库存和现金流风险是什么？",
    ticker="600519.SH",
    config=config,
)

print("\n\n".join(contexts))
```

索引默认写入：

```text
data/rag_index/
```

该目录已经被 `.gitignore` 忽略。

## 9. 推荐的实际使用顺序

建议按下面顺序打通 A 股研究流程：

1. 激活环境并设置环境变量：

```powershell
conda activate fin-agent-sakura
$env:OPENAI_BASE_URL="https://aiapi.clzjwl.cn/v1"
$env:OPENAI_API_KEY="sk-..."
$env:OPENAI_CHAT_MODEL="gpt-5.5"
$env:OPENAI_EMBEDDING_MODEL="text-embedding-3-small"
$env:TUSHARE_TOKEN="你的TuShare token"
```

2. 测试 A 股行情：

```python
client = MarketDataClientFactory.get_client("cn")
prices = await client.fetch_ohlcv("600519.SH", start="2024-01-01")
```

3. 测试 A 股财报：

```python
income = await client.fetch_income_statement("600519.SH")
```

4. 测试 tool calling：

```python
llm = create_financial_tool_calling_llm(model="gpt-4.1-mini")
```

5. 测试价值投资者智能体：

```python
chain = build_value_investor_chain(llm)
```

6. 可选：导入本地 PDF 年报，启用 RAG。

7. 接入 LangGraph 工作流，把基本面、技术面、风险管理和组合经理串起来。

## 9.1 非技术用户的一键投顾界面

项目现在提供一个面向普通用户的 Streamlit GUI。它会完成：

1. 读取你的风险偏好。
2. 构建一个较宽的 A 股候选池。
3. 优先尝试调用 AkShare/TuShare 拉取真实 A 股数据。
4. 如果当前网络或代理导致 A 股数据源失败，则自动切换到离线兜底数据，并在界面中显示警告。
5. 生成候选评分、目标权重、偏离度警报、纸面订单和大模型研报。

启动方式：

```powershell
conda activate fin-agent-sakura
python -m streamlit run src/fin_agent_sakura/dashboard/app.py --server.port 8501
```

然后打开：

```text
http://localhost:8501
```

使用步骤：

1. 在左侧输入风险偏好，例如“我是保守型投资者，期望跑赢通胀即可”。
2. 选择股票池覆盖数量和最终持仓数量。
3. 勾选“使用大模型生成研报”。
4. 点击“生成投资方案”。
5. 查看目标资产配置、候选评分、持仓研报、纸面订单和警报控制台。

结果会保存到：

```text
data/processed/china_investment_result.json
```

注意：这一步是“纸面部署”，不会自动连接券商、不会自动下单。

你也可以从命令行运行：

```powershell
python scripts/run_china_investment_assistant.py
```

如果输出：

```text
mode offline_fallback
```

说明当前 A 股实时数据源没有连通，系统使用了离线兜底数据继续演示完整流程。如果输出：

```text
mode live_data
```

说明本次使用了真实行情数据。

## 9.2 从纸面部署到真实投资

当前系统已经可以生成：

- A 股候选池
- 目标权重
- 偏离度警报
- 纸面交易订单
- LLM 研报

但它还没有接入真实券商交易 API。真实投资前，建议采用以下安全流程：

1. 使用 GUI 生成纸面订单。
2. 下载或复制纸面订单。
3. 人工检查：
   - 是否使用了真实数据模式 `live_data`
   - 是否有数据源警告
   - 是否符合你的风险承受能力
   - 是否满足流动性和仓位限制
4. 手动在券商 App 或交易终端下单。
5. 下单后把真实持仓权重录入后续数据库模块，再由 `PortfolioMonitor` 做每日漂移监控。

后续如果要自动化真实交易，需要再实现一个 Broker Adapter，例如：

- QMT / MiniQMT
- 掘金量化
- 券商柜台 API
- 其他合规交易接口

在接入真实下单前，务必保留 `RiskManager` 作为最终出口风控，任何 LLM 生成的建议都不能绕过硬风控。

## 10. 常见问题

### 10.1 OpenAI 报 authentication failed

检查：

```powershell
echo $env:OPENAI_API_KEY
```

确认 key 没有多余空格或换行。

### 10.2 OpenAI 报 model not found

说明你的账号可能没有该模型权限。换成：

```powershell
$env:OPENAI_CHAT_MODEL="gpt-5.2"
```

或在代码里显式传入你账号可用的模型名。

### 10.3 TuShare 报权限不足

TuShare 的部分财务接口需要积分或权限。你可以：

- 提高 TuShare 账号权限。
- 临时不设置 `TUSHARE_TOKEN`，让项目走 AkShare fallback。
- 为目标数据源单独写适配器。

### 10.4 A 股新闻为空

这是当前项目的已知限制：`get_recent_news` 目前只支持美股 yfinance 新闻。A 股新闻需要后续实现中文新闻源适配器。

### 10.5 技术指标没有 pandas-ta

使用：

```python
TechnicalIndicators(engine="pandas")
```

这是当前最稳的方式。

## 11. 当前项目中需要重点记住的环境变量

```text
OPENAI_API_KEY       OpenAI API Key
OPENAI_BASE_URL      OpenAI-compatible API base URL
OPENAI_CHAT_MODEL    LangChain ChatOpenAI 默认模型
OPENAI_EMBEDDING_MODEL Embedding 模型
TUSHARE_TOKEN        TuShare A 股数据 token
FMP_API_KEY          美股 FMP 财报 token，可选
```

## 12. 官方参考

- OpenAI model list: https://platform.openai.com/docs/models
- GPT-4.1 mini: https://platform.openai.com/docs/models/gpt-4.1-mini
- text-embedding-3-small: https://platform.openai.com/docs/models/text-embedding-3-small
- text-embedding-3-large: https://platform.openai.com/docs/models/text-embedding-3-large
