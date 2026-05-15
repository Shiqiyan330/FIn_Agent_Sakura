"""Lightweight web search agent inspired by DeepResearch-style workflows."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote, urlparse

import pandas as pd
import requests

from fin_agent_sakura.config import get_llm_config, load_dotenv
from fin_agent_sakura.storage import SQLiteStore, record_llm_usage


SearchProvider = Literal["jina", "serper", "fallback"]
DEFAULT_SEARCH_AGENT_OUTPUT = Path("data/processed/search_agent_latest.json")


@dataclass(frozen=True, slots=True)
class SearchSource:
    """One external evidence source returned by the search agent."""

    title: str
    url: str
    snippet: str = ""
    content: str = ""
    provider: SearchProvider = "jina"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SearchAgentResult:
    """Persistable search-agent result for GUI display."""

    query: str
    ticker: str | None
    market: str
    generated_at: str
    answer: str
    sources: list[SearchSource]
    provider: SearchProvider
    warnings: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "ticker": self.ticker,
            "market": self.market,
            "generated_at": self.generated_at,
            "answer": self.answer,
            "sources": [source.to_dict() for source in self.sources],
            "provider": self.provider,
            "warnings": self.warnings,
            "raw": self.raw,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SearchAgentResult":
        return cls(
            query=str(payload.get("query") or ""),
            ticker=payload.get("ticker"),
            market=str(payload.get("market") or "cn"),
            generated_at=str(payload.get("generated_at") or ""),
            answer=str(payload.get("answer") or ""),
            sources=[SearchSource(**item) for item in payload.get("sources", [])],
            provider=payload.get("provider", "fallback"),
            warnings=list(payload.get("warnings") or []),
            raw=dict(payload.get("raw") or {}),
        )


def run_search_agent(
    query: str,
    *,
    ticker: str | None = None,
    market: str = "cn",
    max_results: int = 5,
    use_llm: bool = True,
    output_path: str | Path = DEFAULT_SEARCH_AGENT_OUTPUT,
) -> SearchAgentResult:
    """Run a lightweight search-read-summarize loop and persist the result."""

    load_dotenv()
    warnings: list[str] = []
    resolved_query = _compose_query(query, ticker=ticker, market=market)
    sources: list[SearchSource] = []
    provider: SearchProvider = "fallback"
    raw: dict[str, Any] = {}

    try:
        sources, raw = _search_with_serper(resolved_query, max_results=max_results)
        provider = "serper"
    except Exception as exc:
        warnings.append(f"Serper搜索不可用，尝试Jina搜索：{type(exc).__name__}: {exc}")

    if not sources:
        try:
            sources, raw = _search_with_jina(resolved_query, max_results=max_results)
            provider = "jina"
        except Exception as exc:
            warnings.append(f"Jina搜索不可用，使用本地降级摘要：{type(exc).__name__}: {exc}")

    if not sources:
        sources = [
            SearchSource(
                title="未获得外部搜索结果",
                url="",
                snippet="请配置 SERPER_KEY_ID 或检查 Jina Reader/Search 网络访问。",
                provider="fallback",
            )
        ]
        provider = "fallback"

    enriched = _enrich_sources_with_reader(sources, max_pages=min(max_results, 3), warnings=warnings)
    answer = _summarize_search_result(resolved_query, enriched, use_llm=use_llm, warnings=warnings)
    result = SearchAgentResult(
        query=resolved_query,
        ticker=ticker,
        market=market,
        generated_at=pd.Timestamp.now().isoformat(),
        answer=answer,
        sources=enriched,
        provider=provider,
        warnings=warnings,
        raw=raw,
    )
    _save_result(result, output_path)
    SQLiteStore().save_state_record("search_agent_result", ticker or "general", result.to_dict())
    return result


def load_latest_search_agent_result(
    path: str | Path = DEFAULT_SEARCH_AGENT_OUTPUT,
) -> SearchAgentResult | None:
    result_path = Path(path)
    if not result_path.exists():
        return None
    return SearchAgentResult.from_dict(json.loads(result_path.read_text(encoding="utf-8")))


def _search_with_serper(query: str, *, max_results: int) -> tuple[list[SearchSource], dict[str, Any]]:
    api_key = os.getenv("SERPER_KEY_ID") or os.getenv("SERPER_API_KEY")
    if not api_key:
        raise RuntimeError("SERPER_KEY_ID is not configured")
    response = requests.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        json={"q": query, "num": max_results},
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    sources = [
        SearchSource(
            title=str(item.get("title") or ""),
            url=str(item.get("link") or ""),
            snippet=str(item.get("snippet") or ""),
            provider="serper",
        )
        for item in payload.get("organic", [])[:max_results]
    ]
    return sources, payload


def _search_with_jina(query: str, *, max_results: int) -> tuple[list[SearchSource], dict[str, Any]]:
    url = f"https://s.jina.ai/{quote(query)}"
    response = requests.get(url, headers=_jina_headers(), timeout=35)
    response.raise_for_status()
    text = response.text
    sources = _parse_jina_search_text(text, max_results=max_results)
    return sources, {"text": text[:12000], "endpoint": url}


def _parse_jina_search_text(text: str, *, max_results: int) -> list[SearchSource]:
    sources: list[SearchSource] = []
    blocks = re.split(r"\n(?=Title:\s*)", text)
    for block in blocks:
        title = _extract_field(block, "Title")
        url = _extract_field(block, "URL")
        content = _extract_field(block, "Content")
        if not title and not url:
            continue
        sources.append(
            SearchSource(
                title=title or url,
                url=url,
                snippet=content[:800],
                content=content[:4000],
                provider="jina",
            )
        )
        if len(sources) >= max_results:
            break
    if sources:
        return sources

    urls = re.findall(r"https?://[^\s)>\]]+", text)
    for idx, url in enumerate(urls[:max_results], start=1):
        sources.append(SearchSource(title=f"Jina搜索结果 {idx}", url=url, snippet="", provider="jina"))
    return sources


def _enrich_sources_with_reader(
    sources: list[SearchSource],
    *,
    max_pages: int,
    warnings: list[str],
) -> list[SearchSource]:
    enriched: list[SearchSource] = []
    skipped_count = 0
    failed_count = 0
    for idx, source in enumerate(sources):
        if idx >= max_pages or not source.url or source.content:
            enriched.append(source)
            continue
        if _should_skip_reader(source.url):
            skipped_count += 1
            enriched.append(source)
            continue
        try:
            content = _read_url_with_jina(source.url)
        except Exception:
            failed_count += 1
            enriched.append(source)
        else:
            enriched.append(
                SearchSource(
                    title=source.title,
                    url=source.url,
                    snippet=source.snippet,
                    content=content[:6000],
                    provider=source.provider,
                )
            )
    readable_count = sum(1 for source in enriched if source.content)
    if failed_count and readable_count == 0:
        warnings.append(
            f"搜索结果中有 {failed_count} 个网页正文读取失败，已保留搜索摘要继续分析。"
        )
    if skipped_count and readable_count == 0:
        warnings.append(
            f"已跳过 {skipped_count} 个动态或登录限制网页正文读取，保留搜索摘要继续分析。"
        )
    return enriched


def _read_url_with_jina(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http/https URLs can be read")
    response = requests.get(f"https://r.jina.ai/{url}", headers=_jina_headers(), timeout=35)
    response.raise_for_status()
    return response.text


def _should_skip_reader(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    blocked_hosts = (
        "xueqiu.com",
        "weibo.com",
        "twitter.com",
        "x.com",
        "facebook.com",
        "linkedin.com",
    )
    return any(host == blocked or host.endswith(f".{blocked}") for blocked in blocked_hosts)


def _summarize_search_result(
    query: str,
    sources: list[SearchSource],
    *,
    use_llm: bool,
    warnings: list[str],
) -> str:
    evidence = "\n\n".join(
        f"[{idx}] {source.title}\nURL: {source.url}\n{(source.content or source.snippet)[:2500]}"
        for idx, source in enumerate(sources, start=1)
    )
    fallback = _deterministic_summary(query, sources)
    if not use_llm:
        return fallback

    cfg = get_llm_config()
    if not cfg.api_key:
        warnings.append("OPENAI_API_KEY 未配置，搜索结果使用规则摘要。")
        return fallback

    try:
        from openai import OpenAI

        prompt = (
            "你是A股投研搜索Agent。请基于证据回答，必须区分事实、推断和待验证事项。"
            "不要给出直接下单建议。请用中文输出：摘要、关键证据、风险、后续核验清单。\n"
            f"问题：{query}\n\n证据：\n{evidence[:14000]}"
        )
        client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
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
        content = "".join(chunks).strip() or fallback
        record = record_llm_usage(
            feature="搜索Agent总结",
            model=cfg.chat_model,
            prompt_text=prompt,
            completion_text=content,
            metadata={"query": query, "source_count": len(sources)},
        )
        SQLiteStore().save_llm_usage_record(record)
        return content
    except Exception as exc:
        warnings.append(f"LLM搜索总结失败，已使用规则摘要：{type(exc).__name__}: {exc}")
        return fallback


def _deterministic_summary(query: str, sources: list[SearchSource]) -> str:
    lines = [f"搜索问题：{query}", "", "关键来源："]
    for idx, source in enumerate(sources[:5], start=1):
        snippet = (source.snippet or source.content or "").strip().replace("\n", " ")
        lines.append(f"{idx}. {source.title} - {snippet[:180]}")
    lines.append("")
    lines.append("提示：以上为搜索证据摘要，真实投资前仍需核验公告、交易所披露和财报原文。")
    return "\n".join(lines)


def _compose_query(query: str, *, ticker: str | None, market: str) -> str:
    clean = query.strip()
    if ticker and ticker.upper() not in clean.upper():
        suffix = "A股" if market == "cn" else "stock"
        clean = f"{ticker} {suffix} {clean}"
    return clean


def _extract_field(block: str, field_name: str) -> str:
    pattern = rf"(?im)^{re.escape(field_name)}:\s*(.+?)(?=\n[A-Z][A-Za-z ]*:\s*|\Z)"
    match = re.search(pattern, block, flags=re.S)
    if not match:
        return ""
    return match.group(1).strip()


def _jina_headers() -> dict[str, str]:
    headers = {"Accept": "text/plain"}
    keys = os.getenv("JINA_API_KEYS") or os.getenv("JINA_API_KEY")
    if keys:
        headers["Authorization"] = f"Bearer {keys.split(',')[0].strip()}"
    return headers


def _save_result(result: SearchAgentResult, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
