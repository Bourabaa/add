from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from add_pfe.config import ProjectPaths


DEFAULT_DB_PATH = ProjectPaths.default().notifications_db_path

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS notifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feedback_url TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    predicted_org_slug TEXT,
    predicted_org_name TEXT,
    top1_confidence REAL NOT NULL DEFAULT 0,
    top2_confidence REAL NOT NULL DEFAULT 0,
    top1_margin REAL NOT NULL DEFAULT 0,
    routing_decision TEXT NOT NULL,
    routing_reasons TEXT NOT NULL DEFAULT '',
    top_candidates_json TEXT NOT NULL DEFAULT '[]',
    draft_response TEXT NOT NULL DEFAULT '',
    review_status TEXT NOT NULL DEFAULT 'pending_review',
    review_notes TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def connect_db(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    initialize_db(connection)
    return connection


def initialize_db(connection: sqlite3.Connection) -> None:
    connection.executescript(SCHEMA_SQL)
    connection.commit()


def list_notifications(connection: sqlite3.Connection, review_status: str | None = None) -> list[sqlite3.Row]:
    if review_status and review_status != "all":
        cursor = connection.execute(
            "SELECT * FROM notifications WHERE review_status = ? ORDER BY id DESC",
            (review_status,),
        )
    else:
        cursor = connection.execute("SELECT * FROM notifications ORDER BY id DESC")
    return list(cursor.fetchall())


def get_notification(connection: sqlite3.Connection, notification_id: int) -> sqlite3.Row | None:
    cursor = connection.execute("SELECT * FROM notifications WHERE id = ?", (notification_id,))
    return cursor.fetchone()


def update_notification_status(
    connection: sqlite3.Connection,
    notification_id: int,
    review_status: str,
    review_notes: str = "",
) -> None:
    connection.execute(
        """
        UPDATE notifications
        SET review_status = ?, review_notes = ?, updated_at = ?
        WHERE id = ?
        """,
        (review_status, review_notes.strip(), utc_now_iso(), notification_id),
    )
    connection.commit()


def upsert_notification(connection: sqlite3.Connection, payload: dict[str, Any]) -> tuple[int, str]:
    now = utc_now_iso()
    existing = connection.execute(
        "SELECT id, review_status, review_notes, created_at FROM notifications WHERE feedback_url = ?",
        (payload["feedback_url"],),
    ).fetchone()

    top_candidates_json = json.dumps(payload.get("top_candidates", []), ensure_ascii=False)
    routing_reasons = "|".join(payload.get("routing_reasons", []))

    if existing is None:
        cursor = connection.execute(
            """
            INSERT INTO notifications (
                feedback_url,
                title,
                description,
                predicted_org_slug,
                predicted_org_name,
                top1_confidence,
                top2_confidence,
                top1_margin,
                routing_decision,
                routing_reasons,
                top_candidates_json,
                draft_response,
                review_status,
                review_notes,
                created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["feedback_url"],
                payload.get("title", ""),
                payload.get("description", ""),
                payload.get("predicted_org_slug", ""),
                payload.get("predicted_org_name", ""),
                float(payload.get("top1_confidence", 0.0)),
                float(payload.get("top2_confidence", 0.0)),
                float(payload.get("top1_margin", 0.0)),
                payload.get("routing_decision", "manual_review"),
                routing_reasons,
                top_candidates_json,
                payload.get("draft_response", ""),
                payload.get("review_status", "pending_review"),
                payload.get("review_notes", ""),
                now,
                now,
            ),
        )
        connection.commit()
        return int(cursor.lastrowid), "inserted"

    connection.execute(
        """
        UPDATE notifications
        SET
            title = ?,
            description = ?,
            predicted_org_slug = ?,
            predicted_org_name = ?,
            top1_confidence = ?,
            top2_confidence = ?,
            top1_margin = ?,
            routing_decision = ?,
            routing_reasons = ?,
            top_candidates_json = ?,
            draft_response = ?,
            review_status = ?,
            review_notes = ?,
            created_at = ?,
            updated_at = ?
        WHERE feedback_url = ?
        """,
        (
            payload.get("title", ""),
            payload.get("description", ""),
            payload.get("predicted_org_slug", ""),
            payload.get("predicted_org_name", ""),
            float(payload.get("top1_confidence", 0.0)),
            float(payload.get("top2_confidence", 0.0)),
            float(payload.get("top1_margin", 0.0)),
            payload.get("routing_decision", "manual_review"),
            routing_reasons,
            top_candidates_json,
            payload.get("draft_response", ""),
            payload.get("review_status", existing["review_status"]),
            payload.get("review_notes", existing["review_notes"]),
            existing["created_at"],
            now,
            payload["feedback_url"],
        ),
    )
    connection.commit()
    return int(existing["id"]), "updated"


def deserialize_top_candidates(row: sqlite3.Row | dict[str, Any]) -> list[dict[str, Any]]:
    raw_value = row["top_candidates_json"] if isinstance(row, sqlite3.Row) else row.get("top_candidates_json", "[]")
    try:
        return json.loads(raw_value or "[]")
    except json.JSONDecodeError:
        return []

