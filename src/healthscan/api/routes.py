"""HTTP routes.

Every failure path returns JSON, including 404s, 405s, and unexpected
exceptions - an API client should never have to parse an HTML error page.
"""

from __future__ import annotations

import logging

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.exceptions import HTTPException

from healthscan.api.registry import InferenceError, ModelRegistry, RegistryError
from healthscan.api.validation import ValidationError, parse_features
from healthscan.config import DEFAULT_MODEL

log = logging.getLogger(__name__)


def register_routes(app: Flask, registry: ModelRegistry) -> None:
    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.get("/healthz")
    def healthz():
        """Liveness probe. Reports readiness without failing the check.

        Returns 200 whenever the process is serving; ``ready`` is false if no
        artifacts were loaded, which is a deployment problem rather than a
        reason to have the platform restart a healthy container.
        """
        return jsonify(
            {
                "status": "ok",
                "ready": registry.is_ready,
                "diseases": sorted(registry.diseases),
            }
        )

    @app.get("/api/diseases")
    def get_diseases():
        return jsonify(registry.catalog())

    @app.get("/api/metrics")
    def get_metrics():
        return jsonify(registry.metrics or {})

    @app.post("/api/predict")
    def predict():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            raise ValidationError("Request body must be valid JSON.")

        disease = payload.get("disease")
        if not isinstance(disease, str) or not disease:
            raise ValidationError("A disease must be selected.", {"disease": "Required."})

        model_type = payload.get("model_type") or DEFAULT_MODEL
        if not isinstance(model_type, str):
            raise ValidationError("Model must be a string.", {"model_type": "Invalid."})

        artifacts = registry.get(disease)
        registry.get_model(disease, model_type)

        values = parse_features(disease, artifacts.features, payload)
        return jsonify(registry.predict(disease, model_type, values))

    # ------------------------------------------------------------------
    # Error handlers
    # ------------------------------------------------------------------

    @app.errorhandler(ValidationError)
    def handle_validation_error(exc: ValidationError):
        return jsonify({"error": exc.message, "fields": exc.errors}), 400

    @app.errorhandler(RegistryError)
    def handle_registry_error(exc: RegistryError):
        return jsonify({"error": str(exc)}), 404

    @app.errorhandler(InferenceError)
    def handle_inference_error(exc: InferenceError):
        # Detail is already logged with a traceback; don't leak it to clients.
        log.error("inference error: %s", exc)
        return jsonify({"error": "Prediction failed. Please try again."}), 500

    @app.errorhandler(HTTPException)
    def handle_http_error(exc: HTTPException):
        return jsonify({"error": exc.description}), exc.code

    @app.errorhandler(Exception)
    def handle_unexpected_error(exc: Exception):
        log.exception("unhandled error: %s", exc)
        return jsonify({"error": "Internal server error."}), 500
