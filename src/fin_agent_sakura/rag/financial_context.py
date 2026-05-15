"""PDF annual-report ingestion and hybrid retrieval for financial context."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any, Iterable, Sequence

from fin_agent_sakura.config import get_llm_config


class FinancialRAGError(RuntimeError):
    """Raised when the financial document retrieval pipeline fails."""


@dataclass(frozen=True, slots=True)
class FinancialRAGConfig:
    """Configuration for annual-report indexing and hybrid retrieval."""

    index_root: Path = Path("data/rag_index")
    embedding_model: str = "text-embedding-3-small"
    collection_prefix: str = "financial_reports"
    chunk_size: int = 1_200
    chunk_overlap: int = 180
    page_summary_chars: int = 700
    vector_k: int = 8
    bm25_k: int = 8
    final_k: int = 5
    rrf_k: int = 60


@dataclass(frozen=True, slots=True)
class FinancialContextMatch:
    """One hybrid retrieval match with source metadata and normalized score."""

    context: str
    score: float
    source: str
    page: str
    chunk_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "context": self.context,
            "score": self.score,
            "source": self.source,
            "page": self.page,
            "chunk_id": self.chunk_id,
        }


@dataclass(frozen=True, slots=True)
class _LangChainDeps:
    bm25_retriever: Any
    chroma: Any
    document: Any
    openai_embeddings: Any
    pdf_loader: Any
    text_splitter: Any


class FinancialDocumentPipeline:
    """Build and query a hybrid Chroma + BM25 index over local PDF filings."""

    def __init__(self, config: FinancialRAGConfig | None = None) -> None:
        self.config = config or FinancialRAGConfig()

    def ingest_pdf(self, pdf_path: str | Path, ticker: str, *, overwrite: bool = True) -> int:
        """Read a local annual-report PDF and persist chunked retrieval indexes.

        Returns the number of text chunks indexed.
        """

        deps = _import_langchain_dependencies()
        path = Path(pdf_path)
        if not path.exists() or not path.is_file():
            raise FinancialRAGError(f"PDF file does not exist: {path}")
        if path.suffix.lower() != ".pdf":
            raise FinancialRAGError(f"Expected a PDF file, got: {path.suffix}")

        ticker_key = _normalize_ticker(ticker)
        pages = self._load_pdf_pages(path, ticker_key, deps)
        if not pages:
            raise FinancialRAGError(f"No readable pages found in PDF: {path}")

        splitter = deps.text_splitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ". ", "。", "; ", "；", " ", ""],
        )
        chunks = splitter.split_documents(pages)
        chunks = self._assign_chunk_ids(chunks, ticker_key, path)
        if not chunks:
            raise FinancialRAGError(f"No chunks produced from PDF: {path}")

        self._write_chunk_store(ticker_key, chunks, overwrite=overwrite)
        self._write_vector_store(ticker_key, chunks, deps, overwrite=overwrite)
        return len(chunks)

    def retrieve(self, query: str, ticker: str, *, top_k: int | None = None) -> list[str]:
        """Return the most relevant paragraphs for a ticker and query."""

        return [match.context for match in self.retrieve_with_scores(query, ticker, top_k=top_k)]

    def retrieve_with_scores(
        self,
        query: str,
        ticker: str,
        *,
        top_k: int | None = None,
    ) -> list[FinancialContextMatch]:
        """Return relevant paragraphs with hybrid RRF scores and source metadata."""

        if not query.strip():
            raise FinancialRAGError("query must not be empty")

        deps = _import_langchain_dependencies()
        ticker_key = _normalize_ticker(ticker)
        stored_docs = self._read_chunk_store(ticker_key, deps)
        if not stored_docs:
            raise FinancialRAGError(
                f"No indexed chunks found for {ticker_key}. Ingest a PDF before retrieval."
            )

        vector_docs = self._vector_search(query, ticker_key, deps)
        bm25_docs = self._bm25_search(query, stored_docs, deps)
        fused_docs = self._reciprocal_rank_fusion_with_scores([vector_docs, bm25_docs])

        limit = top_k or self.config.final_k
        if not fused_docs:
            return []
        max_score = max(score for _, score in fused_docs) or 1.0
        return [
            FinancialContextMatch(
                context=_format_context(doc),
                score=float(score / max_score),
                source=Path(str(doc.metadata.get("source", ""))).name,
                page=str(doc.metadata.get("page", "unknown")),
                chunk_id=str(doc.metadata.get("chunk_id") or ""),
            )
            for doc, score in fused_docs[:limit]
        ]

    def delete_index(self, ticker: str) -> None:
        """Delete local chunk store and vector collection for one ticker."""

        deps = _import_langchain_dependencies()
        ticker_key = _normalize_ticker(ticker)
        chunk_path = self._chunk_store_path(ticker_key)
        if chunk_path.exists():
            chunk_path.unlink()

        runtime_config = get_llm_config()
        embeddings = deps.openai_embeddings(
            model=self.config.embedding_model or runtime_config.embedding_model,
            api_key=runtime_config.api_key,
            base_url=runtime_config.base_url,
        )
        store = deps.chroma(
            collection_name=self._collection_name(ticker_key),
            embedding_function=embeddings,
            persist_directory=str(self._vector_dir()),
        )
        try:
            store.delete_collection()
        except Exception:
            pass

    def _load_pdf_pages(self, pdf_path: Path, ticker: str, deps: _LangChainDeps) -> list[Any]:
        loader = deps.pdf_loader(str(pdf_path))
        raw_pages = loader.load()
        pages: list[Any] = []

        for page_number, raw_page in enumerate(raw_pages, start=1):
            text = _normalize_text(raw_page.page_content)
            if not text:
                continue

            loader_page = raw_page.metadata.get("page")
            normalized_page = int(loader_page) + 1 if isinstance(loader_page, int) else page_number
            summary = _extractive_page_summary(text, max_chars=self.config.page_summary_chars)
            metadata = {
                **raw_page.metadata,
                "ticker": ticker,
                "source": str(pdf_path),
                "page": normalized_page,
                "page_summary": summary,
            }
            pages.append(deps.document(page_content=text, metadata=metadata))

        return pages

    def _assign_chunk_ids(self, chunks: Sequence[Any], ticker: str, pdf_path: Path) -> list[Any]:
        assigned: list[Any] = []
        source = str(pdf_path)
        for index, chunk in enumerate(chunks):
            content_hash = sha1(chunk.page_content.encode("utf-8")).hexdigest()[:16]
            chunk_id = f"{ticker}:{index}:{content_hash}"
            chunk.metadata = {
                **chunk.metadata,
                "ticker": ticker,
                "source": source,
                "chunk_index": index,
                "chunk_id": chunk_id,
            }
            assigned.append(chunk)
        return assigned

    def _write_vector_store(
        self,
        ticker: str,
        chunks: Sequence[Any],
        deps: _LangChainDeps,
        *,
        overwrite: bool,
    ) -> None:
        runtime_config = get_llm_config()
        embeddings = deps.openai_embeddings(
            model=self.config.embedding_model or runtime_config.embedding_model,
            api_key=runtime_config.api_key,
            base_url=runtime_config.base_url,
        )
        store = deps.chroma(
            collection_name=self._collection_name(ticker),
            embedding_function=embeddings,
            persist_directory=str(self._vector_dir()),
        )

        if overwrite:
            try:
                store.delete_collection()
            except Exception:
                pass
            store = deps.chroma(
                collection_name=self._collection_name(ticker),
                embedding_function=embeddings,
                persist_directory=str(self._vector_dir()),
            )

        ids = [chunk.metadata["chunk_id"] for chunk in chunks]
        store.add_documents(list(chunks), ids=ids)
        if hasattr(store, "persist"):
            store.persist()

    def _write_chunk_store(self, ticker: str, chunks: Sequence[Any], *, overwrite: bool) -> None:
        path = self._chunk_store_path(ticker)
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if overwrite else "a"
        with path.open(mode, encoding="utf-8") as handle:
            for chunk in chunks:
                record = {
                    "page_content": chunk.page_content,
                    "metadata": chunk.metadata,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _read_chunk_store(self, ticker: str, deps: _LangChainDeps) -> list[Any]:
        path = self._chunk_store_path(ticker)
        if not path.exists():
            return []

        docs: list[Any] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                docs.append(
                    deps.document(
                        page_content=record["page_content"],
                        metadata=record.get("metadata", {}),
                    )
                )
        return docs

    def _vector_search(self, query: str, ticker: str, deps: _LangChainDeps) -> list[Any]:
        runtime_config = get_llm_config()
        embeddings = deps.openai_embeddings(
            model=self.config.embedding_model or runtime_config.embedding_model,
            api_key=runtime_config.api_key,
            base_url=runtime_config.base_url,
        )
        store = deps.chroma(
            collection_name=self._collection_name(ticker),
            embedding_function=embeddings,
            persist_directory=str(self._vector_dir()),
        )
        return store.similarity_search(query, k=self.config.vector_k)

    def _bm25_search(self, query: str, docs: Sequence[Any], deps: _LangChainDeps) -> list[Any]:
        retriever = deps.bm25_retriever.from_documents(list(docs))
        retriever.k = self.config.bm25_k
        if hasattr(retriever, "invoke"):
            return list(retriever.invoke(query))
        return list(retriever.get_relevant_documents(query))

    def _reciprocal_rank_fusion(self, ranked_lists: Iterable[Sequence[Any]]) -> list[Any]:
        return [doc for doc, _ in self._reciprocal_rank_fusion_with_scores(ranked_lists)]

    def _reciprocal_rank_fusion_with_scores(self, ranked_lists: Iterable[Sequence[Any]]) -> list[tuple[Any, float]]:
        scores: dict[str, float] = {}
        docs_by_id: dict[str, Any] = {}

        for ranked_docs in ranked_lists:
            for rank, doc in enumerate(ranked_docs, start=1):
                chunk_id = str(doc.metadata.get("chunk_id") or sha1(doc.page_content.encode()).hexdigest())
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (self.config.rrf_k + rank)
                docs_by_id[chunk_id] = doc

        return [
            (docs_by_id[chunk_id], score)
            for chunk_id, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
        ]

    def _collection_name(self, ticker: str) -> str:
        raw = f"{self.config.collection_prefix}_{ticker}".lower()
        cleaned = re.sub(r"[^a-z0-9_-]", "_", raw)
        cleaned = re.sub(r"_+", "_", cleaned).strip("_")
        return cleaned[:63].ljust(3, "x")

    def _vector_dir(self) -> Path:
        return self.config.index_root / "chroma"

    def _chunk_store_path(self, ticker: str) -> Path:
        return self.config.index_root / "chunks" / f"{ticker}.jsonl"


def ingest_financial_report(
    pdf_path: str | Path,
    ticker: str,
    *,
    config: FinancialRAGConfig | None = None,
    overwrite: bool = True,
) -> int:
    """Index one local annual-report PDF for later retrieval."""

    return FinancialDocumentPipeline(config).ingest_pdf(pdf_path, ticker, overwrite=overwrite)


def retrieve_financial_context(
    query: str,
    ticker: str,
    *,
    top_k: int | None = None,
    config: FinancialRAGConfig | None = None,
) -> list[str]:
    """Retrieve the most relevant annual-report paragraphs for a ticker."""

    return FinancialDocumentPipeline(config).retrieve(query, ticker, top_k=top_k)


def retrieve_financial_context_with_scores(
    query: str,
    ticker: str,
    *,
    top_k: int | None = None,
    config: FinancialRAGConfig | None = None,
) -> list[FinancialContextMatch]:
    """Retrieve relevant annual-report paragraphs with similarity scores."""

    return FinancialDocumentPipeline(config).retrieve_with_scores(query, ticker, top_k=top_k)


def delete_financial_report_index(
    ticker: str,
    *,
    config: FinancialRAGConfig | None = None,
) -> None:
    """Delete all local RAG index artifacts for one ticker."""

    FinancialDocumentPipeline(config).delete_index(ticker)


def _import_langchain_dependencies() -> _LangChainDeps:
    try:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_community.retrievers import BM25Retriever
        from langchain_core.documents import Document
        from langchain_openai import OpenAIEmbeddings
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError as exc:
        raise FinancialRAGError(
            "Install the RAG dependencies with `pip install -e .[rag]` before using this pipeline."
        ) from exc

    try:
        from langchain_chroma import Chroma
    except ImportError:
        try:
            from langchain_community.vectorstores import Chroma
        except ImportError as exc:
            raise FinancialRAGError("Install chromadb/langchain-chroma for vector search.") from exc

    return _LangChainDeps(
        bm25_retriever=BM25Retriever,
        chroma=Chroma,
        document=Document,
        openai_embeddings=OpenAIEmbeddings,
        pdf_loader=PyPDFLoader,
        text_splitter=RecursiveCharacterTextSplitter,
    )


def _normalize_ticker(ticker: str) -> str:
    cleaned = ticker.strip().upper()
    if not cleaned:
        raise FinancialRAGError("ticker must not be empty")
    return re.sub(r"[^A-Z0-9._-]", "_", cleaned)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _extractive_page_summary(text: str, *, max_chars: int) -> str:
    normalized = _normalize_text(text)
    if len(normalized) <= max_chars:
        return normalized

    sentences = re.split(r"(?<=[.!?。！？])\s+", normalized)
    selected: list[str] = []
    current_length = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if current_length + len(sentence) > max_chars:
            break
        selected.append(sentence)
        current_length += len(sentence) + 1

    if selected:
        return " ".join(selected)
    return normalized[:max_chars].rstrip()


def _format_context(doc: Any) -> str:
    page = doc.metadata.get("page", "unknown")
    source = Path(str(doc.metadata.get("source", ""))).name
    summary = doc.metadata.get("page_summary", "")
    prefix = f"[source={source} page={page}]"
    if summary:
        prefix = f"{prefix} page_summary={summary}"
    return f"{prefix}\n{doc.page_content}"
