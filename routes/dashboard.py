from __future__ import annotations

import json
from typing import Any

from flask import Blueprint, redirect, render_template, request, url_for

from add_pfe.config import REVIEW_STATUSES, STATUS_FILTERS
from add_pfe.notifications.repository import (
    DEFAULT_DB_PATH,
    connect_db,
    deserialize_top_candidates,
    get_notification,
    list_notifications,
    update_notification_status,
)

dashboard_bp = Blueprint("dashboard", __name__)


def db_connection():
    return connect_db(DEFAULT_DB_PATH)


def resolve_status_filter(raw_status: str | None) -> str:
    status = raw_status or "all"
    return status if status in STATUS_FILTERS else "all"


def render_notifications_page(selected: Any, status: str) -> str:
    connection = db_connection()
    try:
        notifications = list_notifications(connection, review_status=status)
    finally:
        connection.close()

    if selected is None and notifications:
        selected = notifications[0]

    return render_template(
        "notifications.html",
        notifications=notifications,
        selected=selected,
        top_candidates_json=json.dumps(deserialize_top_candidates(selected), ensure_ascii=False, indent=2) if selected else "[]",
        statuses=STATUS_FILTERS,
        review_statuses=REVIEW_STATUSES,
        current_status=status,
    )


@dashboard_bp.get("/")
def index() -> object:
    return redirect(url_for("dashboard.list_notifications_view"))


@dashboard_bp.get("/notifications")
def list_notifications_view() -> str:
    return render_notifications_page(selected=None, status=resolve_status_filter(request.args.get("status")))


@dashboard_bp.get("/notifications/<int:notification_id>")
def notification_detail(notification_id: int) -> str:
    status = resolve_status_filter(request.args.get("status"))
    connection = db_connection()
    try:
        selected = get_notification(connection, notification_id)
    finally:
        connection.close()
    return render_notifications_page(selected=selected, status=status)


@dashboard_bp.post("/notifications/<int:notification_id>/status")
def update_status(notification_id: int) -> object:
    review_status = request.form.get("review_status", "pending_review")
    review_notes = request.form.get("review_notes", "")
    status_filter = resolve_status_filter(request.form.get("status_filter"))
    if review_status not in REVIEW_STATUSES:
        review_status = "pending_review"

    connection = db_connection()
    try:
        update_notification_status(
            connection,
            notification_id=notification_id,
            review_status=review_status,
            review_notes=review_notes,
        )
    finally:
        connection.close()

    return redirect(url_for("dashboard.notification_detail", notification_id=notification_id, status=status_filter))
