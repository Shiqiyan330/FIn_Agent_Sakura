"""A-share universe builder with local caching and TuShare fallback."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import pandas as pd

from fin_agent_sakura.config import get_tushare_config
from fin_agent_sakura.storage import CacheStore


UniverseSource = Literal["全A", "主板", "创业板", "科创板", "北交所", "沪深300", "中证500", "中证1000"]

DEFAULT_UNIVERSE_OUTPUT = Path("data/processed/a_share_universe_latest.json")
UNIVERSE_TTL_SECONDS = 7 * 24 * 60 * 60

_INDEX_CODES: dict[str, str] = {
    "沪深300": "000300.SH",
    "中证500": "000905.SH",
    "中证1000": "000852.SH",
}


@dataclass(frozen=True, slots=True)
class AShareUniverseItem:
    """One A-share universe row."""

    ticker: str
    name: str
    market: str = "cn"
    board: str = ""
    industry: str = ""
    list_date: str = ""
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "name": self.name,
            "market": self.market,
            "board": self.board,
            "industry": self.industry,
            "list_date": self.list_date,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class AShareUniverseResult:
    """Persistable A-share universe selection result."""

    generated_at: str
    sources: list[str]
    max_count: int
    items: list[AShareUniverseItem]
    cache_hit: bool = False
    cache_key: str = ""
    cache_age_seconds: float | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def tickers(self) -> list[str]:
        return [item.ticker for item in self.items]

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "sources": self.sources,
            "max_count": self.max_count,
            "items": [item.to_dict() for item in self.items],
            "tickers": self.tickers,
            "cache_hit": self.cache_hit,
            "cache_key": self.cache_key,
            "cache_age_seconds": self.cache_age_seconds,
            "warnings": self.warnings,
        }


def build_a_share_universe(
    sources: Sequence[UniverseSource | str],
    max_count: int = 100,
    *,
    force_refresh: bool = False,
    cache_store: CacheStore | None = None,
    output_path: str | Path = DEFAULT_UNIVERSE_OUTPUT,
) -> AShareUniverseResult:
    """Build an A-share universe from boards and index constituents."""

    clean_sources = _normalize_sources(sources)
    if max_count <= 0:
        raise ValueError("max_count must be positive")

    cache = cache_store or CacheStore()
    cache_key = _cache_key(clean_sources, max_count)
    cache_info = cache.describe(cache_key, ttl_seconds=UNIVERSE_TTL_SECONDS)
    if not force_refresh:
        cached = cache.get_json(cache_key, ttl_seconds=UNIVERSE_TTL_SECONDS)
        if cached is not None:
            result = _result_from_payload(cached, cache_hit=True, cache_age_seconds=cache_info.age_seconds)
            _save_result(result, output_path)
            return result

    warnings: list[str] = []
    try:
        frame = _fetch_tushare_universe(clean_sources)
    except Exception as exc:
        warnings.append(f"TuShare股票池拉取失败，已使用项目内置核心股票池：{type(exc).__name__}: {exc}")
        frame = _fallback_universe_frame(clean_sources)

    if frame.empty:
        warnings.append("股票池结果为空，已使用项目内置核心股票池。")
        frame = _fallback_universe_frame(clean_sources)

    frame = _dedupe_and_limit(frame, max_count)
    items = [AShareUniverseItem(**row) for row in frame.to_dict("records")]
    result = AShareUniverseResult(
        generated_at=pd.Timestamp.utcnow().isoformat(),
        sources=clean_sources,
        max_count=max_count,
        items=items,
        cache_hit=False,
        cache_key=cache_key,
        warnings=warnings,
    )
    cache.set_json(cache_key, result.to_dict())
    _save_result(result, output_path)
    return result


def load_latest_a_share_universe(path: str | Path = DEFAULT_UNIVERSE_OUTPUT) -> AShareUniverseResult | None:
    """Load the latest GUI-built A-share universe."""

    result_path = Path(path)
    if not result_path.exists():
        return None
    return _result_from_payload(json.loads(result_path.read_text(encoding="utf-8")))


def _fetch_tushare_universe(sources: list[str]) -> pd.DataFrame:
    pro = _tushare_pro()
    stock_basic = pro.stock_basic(
        exchange="",
        list_status="L",
        fields="ts_code,symbol,name,area,industry,market,list_date",
    )
    if stock_basic.empty:
        raise ValueError("TuShare stock_basic returned no rows")
    stock_basic = stock_basic.rename(columns={"ts_code": "ticker"})
    stock_basic["ticker"] = stock_basic["ticker"].map(_normalize_tushare_code)
    stock_basic["board"] = stock_basic.apply(_infer_board, axis=1)
    stock_basic["source"] = "stock_basic"

    frames: list[pd.DataFrame] = []
    board_sources = [source for source in sources if source not in _INDEX_CODES]
    index_sources = [source for source in sources if source in _INDEX_CODES]

    if "全A" in board_sources:
        frames.append(stock_basic)
    else:
        wanted_boards = set(board_sources)
        if wanted_boards:
            frames.append(stock_basic[stock_basic["board"].isin(wanted_boards)])

    for index_source in index_sources:
        constituents = _fetch_index_constituents(pro, index_source)
        if constituents.empty:
            continue
        merged = constituents.merge(stock_basic, on="ticker", how="left", suffixes=("", "_basic"))
        merged["name"] = merged["name"].fillna(merged.get("con_name"))
        merged["board"] = merged["board"].fillna(merged["ticker"].map(_infer_board_from_ticker))
        merged["source"] = index_source
        frames.append(merged)

    if not frames:
        return pd.DataFrame(columns=_UNIVERSE_COLUMNS)
    combined = pd.concat(frames, ignore_index=True, copy=False)
    return _normalize_universe_frame(combined)


def _fetch_index_constituents(pro: Any, index_source: str) -> pd.DataFrame:
    index_code = _INDEX_CODES[index_source]
    methods = [
        ("index_weight", {"index_code": index_code}),
        ("index_member", {"index_code": index_code}),
    ]
    for method_name, kwargs in methods:
        method = getattr(pro, method_name, None)
        if method is None:
            continue
        try:
            frame = method(**kwargs)
        except Exception:
            continue
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            ticker_col = "con_code" if "con_code" in frame.columns else "ts_code"
            if ticker_col not in frame.columns:
                continue
            result = frame.rename(columns={ticker_col: "ticker", "trade_date": "list_date"})
            result["ticker"] = result["ticker"].map(_normalize_tushare_code)
            result["source"] = index_source
            return result
    return pd.DataFrame(columns=["ticker", "source"])


_UNIVERSE_COLUMNS = ["ticker", "name", "market", "board", "industry", "list_date", "source"]


def _normalize_universe_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized["ticker"] = normalized["ticker"].map(_normalize_tushare_code)
    normalized["name"] = _column_or_default(normalized, "name")
    normalized["market"] = "cn"
    normalized["board"] = _column_or_default(normalized, "board")
    normalized["industry"] = _column_or_default(normalized, "industry")
    normalized["list_date"] = _column_or_default(normalized, "list_date")
    normalized["source"] = _column_or_default(normalized, "source")
    return normalized[_UNIVERSE_COLUMNS]


def _column_or_default(frame: pd.DataFrame, column: str, default: str = "") -> pd.Series:
    if column in frame.columns:
        return frame[column].fillna(default).astype(str)
    return pd.Series([default] * len(frame), index=frame.index, dtype="object")


def _dedupe_and_limit(frame: pd.DataFrame, max_count: int) -> pd.DataFrame:
    normalized = _normalize_universe_frame(frame)
    normalized = normalized.dropna(subset=["ticker"])
    normalized = normalized[normalized["ticker"].astype(str).str.contains(r"\.(?:SH|SZ|BJ)$", regex=True)]
    normalized = normalized.drop_duplicates("ticker", keep="first")
    normalized = normalized.sort_values(["source", "ticker"], kind="mergesort")
    return normalized.head(max_count).reset_index(drop=True)


def _fallback_universe_frame(sources: list[str]) -> pd.DataFrame:
    from fin_agent_sakura.applications.china_investment_assistant import _default_a_share_universe

    rows = []
    source_label = ",".join(sources) or "fallback"
    for candidate in _default_a_share_universe():
        board = _infer_board_from_ticker(candidate.ticker)
        if _matches_sources(candidate.ticker, board, sources):
            rows.append(
                {
                    "ticker": candidate.ticker,
                    "name": candidate.name,
                    "market": "cn",
                    "board": board,
                    "industry": candidate.sector,
                    "list_date": "",
                    "source": f"fallback:{source_label}",
                }
            )
    if not rows:
        for candidate in _default_a_share_universe():
            rows.append(
                {
                    "ticker": candidate.ticker,
                    "name": candidate.name,
                    "market": "cn",
                    "board": _infer_board_from_ticker(candidate.ticker),
                    "industry": candidate.sector,
                    "list_date": "",
                    "source": f"fallback:{source_label}",
                }
            )
    return pd.DataFrame(rows, columns=_UNIVERSE_COLUMNS)


def _matches_sources(ticker: str, board: str, sources: list[str]) -> bool:
    if "全A" in sources or board in sources:
        return True
    return any(source in _INDEX_CODES for source in sources)


def _tushare_pro() -> Any:
    try:
        import tushare as ts
    except ImportError as exc:
        raise RuntimeError("Install tushare to build A-share universe") from exc

    cfg = get_tushare_config()
    if not cfg.token:
        raise RuntimeError("TUSHARE_TOKEN is required")
    pro = ts.pro_api(cfg.token)
    pro._DataApi__token = cfg.token
    if cfg.http_url:
        pro._DataApi__http_url = cfg.http_url
    return pro


def _infer_board(row: pd.Series) -> str:
    market = str(row.get("market") or "")
    ticker = str(row.get("ticker") or row.get("ts_code") or "")
    if "创业板" in market:
        return "创业板"
    if "科创板" in market:
        return "科创板"
    if "北交所" in market or ticker.endswith(".BJ"):
        return "北交所"
    return _infer_board_from_ticker(ticker)


def _infer_board_from_ticker(ticker: str) -> str:
    code = str(ticker).split(".", maxsplit=1)[0]
    suffix = str(ticker).split(".")[-1] if "." in str(ticker) else ""
    if suffix == "BJ" or code.startswith(("4", "8")):
        return "北交所"
    if code.startswith("300"):
        return "创业板"
    if code.startswith("688"):
        return "科创板"
    return "主板"


def _normalize_tushare_code(value: Any) -> str:
    cleaned = str(value).strip().upper()
    if "." in cleaned:
        code, exchange = cleaned.split(".", maxsplit=1)
        return f"{code}.{exchange}"
    if cleaned.startswith(("SH", "SZ", "BJ")):
        return f"{cleaned[2:]}.{cleaned[:2]}"
    if cleaned.startswith(("6", "9")):
        return f"{cleaned}.SH"
    if cleaned.startswith(("4", "8")):
        return f"{cleaned}.BJ"
    return f"{cleaned}.SZ"


def _normalize_sources(sources: Sequence[UniverseSource | str]) -> list[str]:
    allowed = {"全A", "主板", "创业板", "科创板", "北交所", *_INDEX_CODES.keys()}
    cleaned = [str(source).strip() for source in sources if str(source).strip()]
    if not cleaned:
        cleaned = ["沪深300"]
    unknown = sorted(set(cleaned).difference(allowed))
    if unknown:
        raise ValueError(f"Unsupported A-share universe sources: {', '.join(unknown)}")
    result: list[str] = []
    for source in cleaned:
        if source not in result:
            result.append(source)
    return result


def _cache_key(sources: list[str], max_count: int) -> str:
    joined = "_".join(sources)
    return f"a_share_universe_{joined}_{max_count}"


def _save_result(result: AShareUniverseResult, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def _result_from_payload(
    payload: dict[str, Any],
    *,
    cache_hit: bool | None = None,
    cache_age_seconds: float | None = None,
) -> AShareUniverseResult:
    return AShareUniverseResult(
        generated_at=str(payload.get("generated_at", "")),
        sources=list(payload.get("sources") or []),
        max_count=int(payload.get("max_count", 0) or 0),
        items=[
            AShareUniverseItem(
                ticker=str(item.get("ticker", "")),
                name=str(item.get("name", "")),
                market=str(item.get("market", "cn")),
                board=str(item.get("board", "")),
                industry=str(item.get("industry", "")),
                list_date=str(item.get("list_date", "")),
                source=str(item.get("source", "")),
            )
            for item in payload.get("items", [])
        ],
        cache_hit=bool(payload.get("cache_hit", False) if cache_hit is None else cache_hit),
        cache_key=str(payload.get("cache_key", "")),
        cache_age_seconds=cache_age_seconds if cache_age_seconds is not None else payload.get("cache_age_seconds"),
        warnings=list(payload.get("warnings") or []),
    )
