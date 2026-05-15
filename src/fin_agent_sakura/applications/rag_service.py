"""GUI-friendly financial report RAG service."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from fin_agent_sakura.rag import (
    FinancialDocumentPipeline,
    FinancialRAGConfig,
    delete_financial_report_index,
    retrieve_financial_context,
    retrieve_financial_context_with_scores,
)


@dataclass(frozen=True, slots=True)
class FinancialReportIndexInfo:
    """Summary of one indexed ticker report collection."""

    ticker: str
    chunk_count: int
    sources: list[str]
    page_summaries: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FinancialContextAnswer:
    """Retrieved context snippets for a financial report question."""

    ticker: str
    query: str
    contexts: list[str]
    matches: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def save_uploaded_report(uploaded_file: Any, *, ticker: str, upload_dir: str | Path = "data/reports") -> Path:
    """Persist an uploaded Streamlit PDF file to local storage."""

    ticker_key = _normalize_ticker(ticker)
    directory = Path(upload_dir) / ticker_key
    directory.mkdir(parents=True, exist_ok=True)
    filename = Path(str(uploaded_file.name)).name
    if not filename.lower().endswith(".pdf"):
        raise ValueError("只能上传 PDF 文件。")
    path = directory / filename
    path.write_bytes(uploaded_file.getbuffer())
    return path


def ingest_uploaded_financial_report(
    pdf_path: str | Path,
    *,
    ticker: str,
    overwrite: bool = True,
    config: FinancialRAGConfig | None = None,
) -> FinancialReportIndexInfo:
    """Index a saved PDF and return index metadata for GUI display."""

    pipeline = FinancialDocumentPipeline(config)
    pipeline.ingest_pdf(pdf_path, ticker, overwrite=overwrite)
    return get_financial_report_index_info(ticker, config=config)


def ask_financial_report(
    *,
    ticker: str,
    query: str,
    top_k: int = 5,
    config: FinancialRAGConfig | None = None,
) -> FinancialContextAnswer:
    """Retrieve relevant report contexts for a GUI query."""

    matches = retrieve_financial_context_with_scores(query=query, ticker=ticker, top_k=top_k, config=config)
    return FinancialContextAnswer(
        ticker=_normalize_ticker(ticker),
        query=query,
        contexts=[match.context for match in matches],
        matches=[match.to_dict() for match in matches],
    )


def delete_indexed_financial_report(
    ticker: str,
    *,
    config: FinancialRAGConfig | None = None,
) -> FinancialReportIndexInfo:
    """Delete the local RAG index for one ticker and return empty metadata."""

    ticker_key = _normalize_ticker(ticker)
    delete_financial_report_index(ticker_key, config=config)
    return get_financial_report_index_info(ticker_key, config=config)


def list_indexed_financial_reports(config: FinancialRAGConfig | None = None) -> list[FinancialReportIndexInfo]:
    """List all indexed tickers under the local chunk store."""

    cfg = config or FinancialRAGConfig()
    chunk_dir = cfg.index_root / "chunks"
    if not chunk_dir.exists():
        return []
    infos = []
    for path in sorted(chunk_dir.glob("*.jsonl")):
        infos.append(get_financial_report_index_info(path.stem, config=cfg))
    return infos


def get_financial_report_index_info(
    ticker: str,
    *,
    config: FinancialRAGConfig | None = None,
) -> FinancialReportIndexInfo:
    """Read chunk metadata and page summaries for an indexed ticker."""

    cfg = config or FinancialRAGConfig()
    ticker_key = _normalize_ticker(ticker)
    path = cfg.index_root / "chunks" / f"{ticker_key}.jsonl"
    if not path.exists():
        return FinancialReportIndexInfo(ticker=ticker_key, chunk_count=0, sources=[], page_summaries=[])

    chunk_count = 0
    sources: set[str] = set()
    summaries_by_page: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            chunk_count += 1
            record = json.loads(line)
            metadata = record.get("metadata", {})
            source = Path(str(metadata.get("source", ""))).name
            page = str(metadata.get("page", "unknown"))
            summary = str(metadata.get("page_summary", ""))
            if source:
                sources.add(source)
            key = (source, page)
            if summary and key not in summaries_by_page:
                summaries_by_page[key] = {
                    "source": source,
                    "page": page,
                    "page_summary": summary,
                }

    return FinancialReportIndexInfo(
        ticker=ticker_key,
        chunk_count=chunk_count,
        sources=sorted(sources),
        page_summaries=list(summaries_by_page.values())[:30],
    )


def _normalize_ticker(ticker: str) -> str:
    cleaned = ticker.strip().upper()
    if not cleaned:
        raise ValueError("ticker 不能为空。")
    return cleaned
