from __future__ import annotations

from flask import Flask

from routes.api import api_bp
from routes.dashboard import dashboard_bp
from routes.health import health_bp


def register_blueprints(app: Flask) -> None:
    app.register_blueprint(health_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(api_bp)
