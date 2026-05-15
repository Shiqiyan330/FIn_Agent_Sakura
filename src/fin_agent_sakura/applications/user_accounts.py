"""Local user account records for the conversational advisor."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_USERS_DIR = Path("data/processed/users")
DEFAULT_ACTIVE_USER_PATH = Path("data/processed/active_user.json")


@dataclass(frozen=True, slots=True)
class LocalUserAccount:
    """A lightweight local user account for separating advisory data."""

    user_id: str
    display_name: str
    investment_role: str = "个人投资者"
    goals: str = ""
    created_at: str = field(default_factory=lambda: pd.Timestamp.now().isoformat())
    updated_at: str = field(default_factory=lambda: pd.Timestamp.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LocalUserAccount":
        return cls(
            user_id=str(payload.get("user_id") or "default_user"),
            display_name=str(payload.get("display_name") or "默认用户"),
            investment_role=str(payload.get("investment_role") or "个人投资者"),
            goals=str(payload.get("goals") or ""),
            created_at=str(payload.get("created_at") or pd.Timestamp.now().isoformat()),
            updated_at=str(payload.get("updated_at") or pd.Timestamp.now().isoformat()),
        )


def save_user_account(
    display_name: str,
    *,
    investment_role: str = "个人投资者",
    goals: str = "",
    user_id: str | None = None,
    users_dir: str | Path = DEFAULT_USERS_DIR,
) -> LocalUserAccount:
    """Create or update a local user account."""

    resolved_id = user_id or _new_user_id(display_name)
    existing = load_user_account(resolved_id, users_dir=users_dir)
    account = LocalUserAccount(
        user_id=resolved_id,
        display_name=display_name.strip() or resolved_id,
        investment_role=investment_role.strip() or "个人投资者",
        goals=goals.strip(),
        created_at=existing.created_at if existing is not None else pd.Timestamp.now().isoformat(),
        updated_at=pd.Timestamp.now().isoformat(),
    )
    path = _account_path(account.user_id, users_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(account.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    set_active_user(account.user_id)
    return account


def list_user_accounts(users_dir: str | Path = DEFAULT_USERS_DIR) -> list[LocalUserAccount]:
    """List local user accounts."""

    root = Path(users_dir)
    if not root.exists():
        return []
    accounts = []
    for path in sorted(root.glob("*/profile.json")):
        try:
            accounts.append(LocalUserAccount.from_dict(json.loads(path.read_text(encoding="utf-8"))))
        except Exception:
            continue
    return accounts


def load_user_account(user_id: str, *, users_dir: str | Path = DEFAULT_USERS_DIR) -> LocalUserAccount | None:
    path = _account_path(user_id, users_dir)
    if not path.exists():
        return None
    return LocalUserAccount.from_dict(json.loads(path.read_text(encoding="utf-8")))


def get_or_create_active_user() -> LocalUserAccount:
    active_id = load_active_user_id()
    if active_id:
        account = load_user_account(active_id)
        if account is not None:
            return account
    return save_user_account("默认用户", user_id="default_user")


def set_active_user(user_id: str) -> None:
    DEFAULT_ACTIVE_USER_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_ACTIVE_USER_PATH.write_text(json.dumps({"user_id": user_id}, ensure_ascii=False), encoding="utf-8")


def load_active_user_id() -> str | None:
    if not DEFAULT_ACTIVE_USER_PATH.exists():
        return None
    try:
        payload = json.loads(DEFAULT_ACTIVE_USER_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload.get("user_id")


def user_data_dir(user_id: str) -> Path:
    path = DEFAULT_USERS_DIR / _slug_user_id(user_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _account_path(user_id: str, users_dir: str | Path) -> Path:
    return Path(users_dir) / _slug_user_id(user_id) / "profile.json"


def _slug_user_id(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+", "_", value.strip())
    cleaned = cleaned.strip("_")
    return cleaned or "default_user"


def _new_user_id(display_name: str) -> str:
    slug = _slug_user_id(display_name)
    if slug == "default_user" and display_name.strip() and display_name.strip() != "默认用户":
        return f"user_{uuid.uuid4().hex[:8]}"
    return slug
