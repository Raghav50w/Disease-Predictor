"""The web API.

    uvicorn healthscan.main:app --reload
"""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from healthscan.config import DEFAULT_MODEL, STATIC_DIR
from healthscan.registry import ModelRegistry
from healthscan.schemas import PredictRequest, validate_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="HealthScanner", docs_url="/docs")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

registry = ModelRegistry()
registry.load()


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/healthz")
def healthz():
    """Liveness probe for Render."""
    return {
        "status": "ok",
        "ready": registry.is_ready,
        "diseases": sorted(registry.diseases),
    }


@app.get("/api/diseases")
def get_diseases():
    return registry.catalog()


@app.get("/api/metrics")
def get_metrics():
    return registry.metrics


@app.post("/api/predict")
def predict(request: PredictRequest):
    disease = request.disease
    if disease not in registry.diseases:
        available = sorted(registry.diseases)
        return JSONResponse(
            status_code=404,
            content={"error": f"Unknown disease '{disease}'. Available: {available}"},
        )

    model_type = request.model_type or DEFAULT_MODEL
    models = registry.diseases[disease]["models"]
    if model_type not in models:
        available = sorted(models)
        return JSONResponse(
            status_code=404,
            content={
                "error": (
                    f"Model '{model_type}' is not available for '{disease}'. "
                    f"Available: {available}"
                )
            },
        )

    features = registry.diseases[disease]["features"]
    values, errors = validate_features(disease, features, request.feature_values())
    if errors:
        return JSONResponse(
            status_code=400,
            content={"error": "One or more inputs are invalid.", "fields": errors},
        )

    return registry.predict(disease, model_type, values)


@app.exception_handler(Exception)
def handle_unexpected_error(request: Request, exc: Exception):
    """Return JSON on a crash instead of leaking a traceback to the client."""
    log.exception("unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"error": "Internal server error."})
