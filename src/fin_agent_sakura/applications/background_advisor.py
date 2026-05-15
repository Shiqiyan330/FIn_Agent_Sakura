"""Background execution support for conversational advisor turns."""

from __future__ import annotations

import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from fin_agent_sakura.applications.conversational_advisor import (
    AgentCallEvent,
    ConversationalAdvisorSession,
    continue_conversational_advisor_session,
)
from fin_agent_sakura.applications.user_accounts import user_data_dir


TaskStatus = Literal["pending", "running", "success", "failed", "timed_out"]
DEFAULT_TIMEOUT_SECONDS = 20 * 60
_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="advisor-bg")
_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class BackgroundAdvisorTask:
    """Persisted status for one background conversational advisor turn."""

    task_id: str
    user_id: str
    session_id: str
    user_message: str
    status: TaskStatus = "pending"
    created_at: str = field(default_factory=lambda: pd.Timestamp.now().isoformat())
    started_at: str = ""
    updated_at: str = field(default_factory=lambda: pd.Timestamp.now().isoformat())
    completed_at: str = ""
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    error: str = ""
    events: list[dict[str, Any]] = field(default_factory=list)
    result_session: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BackgroundAdvisorTask":
        return cls(
            task_id=str(payload.get("task_id") or uuid.uuid4().hex),
            user_id=str(payload.get("user_id") or "default_user"),
            session_id=str(payload.get("session_id") or ""),
            user_message=str(payload.get("user_message") or ""),
            status=payload.get("status", "pending"),
            created_at=str(payload.get("created_at") or pd.Timestamp.now().isoformat()),
            started_at=str(payload.get("started_at") or ""),
            updated_at=str(payload.get("updated_at") or pd.Timestamp.now().isoformat()),
            completed_at=str(payload.get("completed_at") or ""),
            timeout_seconds=int(payload.get("timeout_seconds") or DEFAULT_TIMEOUT_SECONDS),
            error=str(payload.get("error") or ""),
            events=list(payload.get("events") or []),
            result_session=dict(payload.get("result_session") or {}),
        )


def submit_background_advisor_turn(
    session: ConversationalAdvisorSession,
    user_message: str,
    *,
    use_llm: bool = True,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> BackgroundAdvisorTask:
    """Submit one advisor turn to a process-local background worker."""

    task = BackgroundAdvisorTask(
        task_id=uuid.uuid4().hex,
        user_id=session.user_id,
        session_id=session.session_id,
        user_message=user_message,
        timeout_seconds=timeout_seconds,
    )
    _save_task(task)
    _EXECUTOR.submit(_run_advisor_turn, task.task_id, session, user_message, use_llm)
    return task


def load_background_advisor_task(user_id: str, task_id: str) -> BackgroundAdvisorTask | None:
    path = _task_path(user_id, task_id)
    if not path.exists():
        return None
    return _with_timeout_status(BackgroundAdvisorTask.from_dict(json.loads(path.read_text(encoding="utf-8"))))


def load_latest_background_advisor_task(user_id: str) -> BackgroundAdvisorTask | None:
    latest = _latest_task_path(user_id)
    if not latest.exists():
        return None
    try:
        payload = json.loads(latest.read_text(encoding="utf-8"))
        task_id = str(payload.get("task_id") or "")
    except Exception:
        return None
    if not task_id:
        return None
    return load_background_advisor_task(user_id, task_id)


def is_background_task_active(task: BackgroundAdvisorTask | None) -> bool:
    return task is not None and task.status in {"pending", "running"}


def _run_advisor_turn(
    task_id: str,
    session: ConversationalAdvisorSession,
    user_message: str,
    use_llm: bool,
) -> None:
    started = time.monotonic()
    task = load_background_advisor_task(session.user_id, task_id)
    if task is None:
        return
    _save_task(_replace_task(task, status="running", started_at=pd.Timestamp.now().isoformat(), updated_at=pd.Timestamp.now().isoformat()))

    def on_event(event: AgentCallEvent) -> None:
        current = load_background_advisor_task(session.user_id, task_id)
        if current is None or current.status == "timed_out":
            return
        events = [*current.events, event.to_dict()]
        _save_task(_replace_task(current, events=events[-80:], updated_at=pd.Timestamp.now().isoformat()))

    try:
        result = continue_conversational_advisor_session(
            session,
            user_message,
            use_llm=use_llm,
            progress_callback=on_event,
        )
    except Exception as exc:
        current = load_background_advisor_task(session.user_id, task_id) or task
        if time.monotonic() - started > current.timeout_seconds:
            status: TaskStatus = "timed_out"
            error = f"对话后台任务超过 {current.timeout_seconds // 60} 分钟，已标记超时。"
        else:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
        _save_task(
            _replace_task(
                current,
                status=status,
                error=error,
                completed_at=pd.Timestamp.now().isoformat(),
                updated_at=pd.Timestamp.now().isoformat(),
            )
        )
        return

    current = load_background_advisor_task(session.user_id, task_id) or task
    if time.monotonic() - started > current.timeout_seconds:
        _save_task(
            _replace_task(
                current,
                status="timed_out",
                error=f"对话后台任务超过 {current.timeout_seconds // 60} 分钟，已标记超时；如结果已保存，请刷新最新对话。",
                result_session=result.to_dict(),
                completed_at=pd.Timestamp.now().isoformat(),
                updated_at=pd.Timestamp.now().isoformat(),
            )
        )
        return
    _save_task(
        _replace_task(
            current,
            status="success",
            result_session=result.to_dict(),
            completed_at=pd.Timestamp.now().isoformat(),
            updated_at=pd.Timestamp.now().isoformat(),
        )
    )


def _with_timeout_status(task: BackgroundAdvisorTask) -> BackgroundAdvisorTask:
    if task.status not in {"pending", "running"}:
        return task
    started_text = task.started_at or task.created_at
    try:
        started = pd.Timestamp(started_text).timestamp()
    except Exception:
        return task
    if time.time() - started <= task.timeout_seconds:
        return task
    timed_out = _replace_task(
        task,
        status="timed_out",
        error=f"对话后台任务超过 {task.timeout_seconds // 60} 分钟，已标记超时。",
        completed_at=pd.Timestamp.now().isoformat(),
        updated_at=pd.Timestamp.now().isoformat(),
    )
    _save_task(timed_out)
    return timed_out


def _replace_task(task: BackgroundAdvisorTask, **changes: Any) -> BackgroundAdvisorTask:
    payload = task.to_dict()
    payload.update(changes)
    return BackgroundAdvisorTask.from_dict(payload)


def _save_task(task: BackgroundAdvisorTask) -> None:
    with _LOCK:
        path = _task_path(task.user_id, task.task_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(task.to_dict(), ensure_ascii=False, indent=2, default=str)
        path.write_text(payload, encoding="utf-8")
        _latest_task_path(task.user_id).write_text(
            json.dumps({"task_id": task.task_id, "updated_at": task.updated_at}, ensure_ascii=False),
            encoding="utf-8",
        )


def _task_path(user_id: str, task_id: str) -> Path:
    return user_data_dir(user_id) / "background_tasks" / f"{task_id}.json"


def _latest_task_path(user_id: str) -> Path:
    return user_data_dir(user_id) / "background_tasks" / "latest.json"
