"""Flask application entry point.

Responsibilities of this file are limited to:
  1. Creating the Flask app instance.
  2. Registering Blueprints.
  3. Running the development server when invoked directly.

All route logic lives in routes/.
"""

from __future__ import annotations

from flask import Flask

from add_pfe.config import load_dashboard_config
from routes.health import health_bp
from routes.dashboard import dashboard_bp
from routes.api import api_bp


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["JSON_AS_ASCII"] = False

    # ── Blueprints ──────────────────────────────────────────────────────────
    app.register_blueprint(health_bp)       # GET  /healthz
    app.register_blueprint(dashboard_bp)    # GET/POST /notifications/*  + GET /
    app.register_blueprint(api_bp)          # POST /api/route  /api/route-batch

    return app


app = create_app()


if __name__ == "__main__":
    cfg = load_dashboard_config()
    app.run(host=cfg.host, port=cfg.port, debug=cfg.debug)