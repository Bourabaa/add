from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any

from flask import Blueprint, jsonify, request

from add_pfe.integration.payloads import route_feedback_payloads

if TYPE_CHECKING:
    from add_pfe.routing.service import RoutingService

api_bp = Blueprint("api", __name__)


@lru_cache(maxsize=1)
def get_routing_service() -> "RoutingService":
    # The matcher is loaded lazily so the dashboard stays responsive on startup.
    from add_pfe.routing.service import RoutingService

    return RoutingService.from_project_defaults()


@api_bp.post("/api/route")
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


@api_bp.post("/api/route-batch")
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
