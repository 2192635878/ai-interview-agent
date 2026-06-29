from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .state import AgentState, now_text


def _json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False)


def _json_loads(text: str, default: Any) -> Any:
    try:
        return json.loads(text) if text else default
    except json.JSONDecodeError:
        return default


def init_memory_db(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_states (
            session_id INTEGER PRIMARY KEY,
            state_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS user_profiles (
            client_id TEXT PRIMARY KEY,
            preferred_role TEXT,
            total_sessions INTEGER NOT NULL DEFAULT 0,
            average_score REAL NOT NULL DEFAULT 0,
            weaknesses_json TEXT NOT NULL DEFAULT '[]',
            strengths_json TEXT NOT NULL DEFAULT '[]',
            review_plan_json TEXT NOT NULL DEFAULT '[]',
            updated_at TEXT NOT NULL
        )
        """
    )


def save_agent_state(connection: sqlite3.Connection, session_id: int, state: AgentState) -> None:
    connection.execute(
        """
        INSERT INTO agent_states (session_id, state_json, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            state_json = excluded.state_json,
            updated_at = excluded.updated_at
        """,
        (session_id, _json_dumps(state.to_dict()), now_text()),
    )


def load_agent_state(connection: sqlite3.Connection, session_id: int) -> AgentState | None:
    row = connection.execute(
        "SELECT state_json FROM agent_states WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    if row is None:
        return None
    return AgentState.from_dict(_json_loads(row["state_json"], {}))


def load_user_profile(connection: sqlite3.Connection, client_id: str) -> Dict[str, Any]:
    row = connection.execute(
        "SELECT * FROM user_profiles WHERE client_id = ?",
        (client_id,),
    ).fetchone()
    if row is None:
        return {
            "client_id": client_id,
            "preferred_role": "",
            "total_sessions": 0,
            "average_score": 0.0,
            "weaknesses": [],
            "strengths": [],
            "review_plan": [],
        }
    return {
        "client_id": client_id,
        "preferred_role": row["preferred_role"] or "",
        "total_sessions": row["total_sessions"],
        "average_score": row["average_score"],
        "weaknesses": _json_loads(row["weaknesses_json"], []),
        "strengths": _json_loads(row["strengths_json"], []),
        "review_plan": _json_loads(row["review_plan_json"], []),
    }


def _merge_unique(existing: Iterable[str], incoming: Iterable[str], limit: int = 20) -> List[str]:
    merged = []
    for item in [*existing, *incoming]:
        cleaned = str(item).strip()
        if cleaned and cleaned not in merged:
            merged.append(cleaned)
    return merged[:limit]


def update_user_profile(
    connection: sqlite3.Connection,
    client_id: str,
    role: str,
    average_score: float,
    weaknesses: Iterable[str],
    strengths: Iterable[str],
    review_plan: Iterable[str] | None = None,
) -> Dict[str, Any]:
    current = load_user_profile(connection, client_id)
    total_sessions = int(current["total_sessions"]) + 1
    previous_average = float(current["average_score"])
    new_average = round(((previous_average * (total_sessions - 1)) + average_score) / total_sessions, 2)
    merged_weaknesses = _merge_unique(current["weaknesses"], weaknesses)
    merged_strengths = _merge_unique(current["strengths"], strengths)
    merged_plan = _merge_unique(review_plan or [], current.get("review_plan", []), limit=10)
    connection.execute(
        """
        INSERT INTO user_profiles (
            client_id, preferred_role, total_sessions, average_score,
            weaknesses_json, strengths_json, review_plan_json, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(client_id) DO UPDATE SET
            preferred_role = excluded.preferred_role,
            total_sessions = excluded.total_sessions,
            average_score = excluded.average_score,
            weaknesses_json = excluded.weaknesses_json,
            strengths_json = excluded.strengths_json,
            review_plan_json = excluded.review_plan_json,
            updated_at = excluded.updated_at
        """,
        (
            client_id,
            role,
            total_sessions,
            new_average,
            _json_dumps(merged_weaknesses),
            _json_dumps(merged_strengths),
            _json_dumps(merged_plan),
            now_text(),
        ),
    )
    return load_user_profile(connection, client_id)


def extract_profile_update(state: AgentState) -> Dict[str, Any]:
    if state.scores:
        average_score = round(sum(score.average() for score in state.scores) / len(state.scores), 2)
    else:
        average_score = 0.0
    return {
        "average_score": average_score,
        "weaknesses": state.weaknesses,
        "strengths": state.strengths,
    }
