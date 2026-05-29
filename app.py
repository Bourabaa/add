from __future__ import annotations

import json
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from flask import Flask, jsonify, redirect, render_template, request, url_for

from add_pfe.config import REVIEW_STATUSES, STATUS_FILTERS, load_dashboard_config
from add_pfe.notifications.repository import (
    DEFAULT_DB_PATH,
    connect_db,
    deserialize_top_candidates,
    get_notification,
    list_notifications,
    update_notification_status,
)
from add_pfe.integration.payloads import route_feedback_payloads
if TYPE_CHECKING:
    from add_pfe.routing.service import RoutingService


app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False


def db_connection():
    return connect_db(DEFAULT_DB_PATH)


@lru_cache(maxsize=1)
def get_routing_service() -> "RoutingService":
    # The matcher is loaded lazily so the dashboard stays responsive on startup.
    from add_pfe.routing.service import RoutingService

    return RoutingService.from_project_defaults()


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


@app.get("/")
def index() -> object:
    return redirect(url_for("list_notifications_view"))


@app.get("/healthz")
def healthz() -> Any:
    return jsonify({"status": "ok"})


@app.get("/notifications")
def list_notifications_view() -> str:
    return render_notifications_page(selected=None, status=resolve_status_filter(request.args.get("status")))


@app.get("/notifications/<int:notification_id>")
def notification_detail(notification_id: int) -> str:
    status = resolve_status_filter(request.args.get("status"))
    connection = db_connection()
    try:
        selected = get_notification(connection, notification_id)
    finally:
        connection.close()
    return render_notifications_page(selected=selected, status=status)


@app.post("/notifications/<int:notification_id>/status")
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

    return redirect(url_for("notification_detail", notification_id=notification_id, status=status_filter))


@app.post("/api/route")
def route_request_api() -> tuple[Any, int] | Any:
    payload = request.get_json(silent=True) or {}
    title = str(payload.get("title", ""))
    description = str(payload.get("description", ""))
    request_text = str(payload.get("request_text", ""))
    top_k = int(payload.get("top_k", 3) or 3)

    if not any(item.strip() for item in [title, description, request_text]):
        return jsonify({"error": "Provide request_text or title/description."}), 400

    routing_service = get_routing_service()
    if request_text.strip():
        result = routing_service.route_text(request_text=request_text, top_k=top_k)
    else:
        result = routing_service.route_feedback(title=title, description=description, top_k=top_k)
    return jsonify(result)


@app.post("/api/route-batch")
def route_batch_api() -> tuple[Any, int] | Any:
    payload = request.get_json(silent=True) or {}
    organizations = payload.get("organizations") or []
    feedbacks = payload.get("feedbacks") or []
    top_k = int(payload.get("top_k", 3) or 3)

    if not isinstance(organizations, list) or not organizations:
        return jsonify({"error": "Provide a non-empty organizations list."}), 400
    if not isinstance(feedbacks, list) or not feedbacks:
        return jsonify({"error": "Provide a non-empty feedbacks list."}), 400

    try:
        result = route_feedback_payloads(
            organizations=organizations,
            feedbacks=feedbacks,
            top_k=top_k,
        )
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    return jsonify(result)


if __name__ == "__main__":
    dashboard_config = load_dashboard_config()
    app.run(host=dashboard_config.host, port=dashboard_config.port, debug=dashboard_config.debug)
