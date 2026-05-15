"""Retrieval-augmented generation utilities for financial documents."""

from fin_agent_sakura.rag.financial_context import (
    FinancialDocumentPipeline,
    FinancialContextMatch,
    FinancialRAGConfig,
    FinancialRAGError,
    delete_financial_report_index,
    ingest_financial_report,
    retrieve_financial_context,
    retrieve_financial_context_with_scores,
)

__all__ = [
    "FinancialDocumentPipeline",
    "FinancialContextMatch",
    "FinancialRAGConfig",
    "FinancialRAGError",
    "delete_financial_report_index",
    "ingest_financial_report",
    "retrieve_financial_context",
    "retrieve_financial_context_with_scores",
]
