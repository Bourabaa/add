from __future__ import annotations

from typing import Any

from flask import Blueprint, jsonify

health_bp = Blueprint("health", __name__)


@health_bp.get("/healthz")
def healthz() -> Any:
    return jsonify({"status": "ok"})
