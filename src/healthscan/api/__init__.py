"""Flask application factory."""

from __future__ import annotations

import logging
from pathlib import Path

from flask import Flask

from healthscan.api.registry import ModelRegistry
from healthscan.api.routes import register_routes
from healthscan.config import MODELS_DIR, STATIC_DIR

log = logging.getLogger(__name__)


def create_app(models_dir: Path | str | None = None) -> Flask:
    """Build a configured Flask app.

    The app serves its own frontend from ``static/``, so it is same-origin and
    needs no CORS layer.

    Artifacts are loaded here rather than at import time so tests can point
    the app at a temporary directory of fixtures.
    """
    app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")

    registry = ModelRegistry(models_dir or MODELS_DIR)
    registry.load()
    app.extensions["registry"] = registry

    register_routes(app, registry)
    return app
