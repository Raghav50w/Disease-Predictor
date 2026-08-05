"""Discovery, loading, and inference for the trained artifacts.

A class rather than module-level state, so the registry can be pointed at a
temporary directory of fixtures in tests instead of the real ``models/``.

Security note: this unpickles every ``*.pkl`` it finds, which is arbitrary
code execution if an attacker can write to the models directory. That is
acceptable here only because the artifacts are generated during the Docker
build from CSVs committed to this repo - they are never fetched from a remote
or user-supplied location. Do not change that without adding integrity checks.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from healthscan.config import (
    DEFAULT_MODEL,
    METRICS_FILENAME,
    MODEL_DISPLAY_NAMES,
    disease_meta,
    feature_meta,
)

log = logging.getLogger(__name__)


class RegistryError(Exception):
    """Raised when a disease or model is not available."""


class InferenceError(Exception):
    """Raised when a loaded model fails to produce a prediction."""


@dataclass
class DiseaseArtifacts:
    """Everything needed to serve one disease."""

    name: str
    scaler: object
    features: list[str]
    models: dict[str, object] = field(default_factory=dict)


class ModelRegistry:
    def __init__(self, models_dir: Path | str):
        self.models_dir = Path(models_dir)
        self.diseases: dict[str, DiseaseArtifacts] = {}
        self.metrics: dict = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Scan the models directory and load every complete artifact set."""
        self.diseases = {}
        self.metrics = {}

        if not self.models_dir.is_dir():
            log.warning(
                "models directory %s not found - run `python -m healthscan.ml.train`",
                self.models_dir,
            )
            return

        for scaler_path in sorted(self.models_dir.glob("*_scaler.pkl")):
            name = scaler_path.name.removesuffix("_scaler.pkl")
            features_path = self.models_dir / f"{name}_features.pkl"
            if not features_path.exists():
                log.warning("skipping %s: no %s", name, features_path.name)
                continue

            # Everything downstream - the catalog, validation bounds, the
            # form the frontend renders - is driven by DISEASE_META. A stale
            # artifact for a disease that is no longer configured is skipped
            # here rather than defended against at each use site.
            if disease_meta(name) is None:
                log.warning("skipping %s: no DISEASE_META entry", name)
                continue

            artifacts = DiseaseArtifacts(
                name=name,
                scaler=self._unpickle(scaler_path),
                features=self._unpickle(features_path),
            )

            for model_path in sorted(self.models_dir.glob(f"{name}_*.pkl")):
                model_type = model_path.name.removeprefix(f"{name}_").removesuffix(".pkl")
                if model_type in ("scaler", "features"):
                    continue
                artifacts.models[model_type] = self._unpickle(model_path)

            if not artifacts.models:
                log.warning("skipping %s: no model artifacts found", name)
                continue

            self.diseases[name] = artifacts
            log.info("loaded %s (%s)", name, ", ".join(sorted(artifacts.models)))

        self._load_metrics()

        if not self.diseases:
            log.warning("no models loaded from %s", self.models_dir)

    def _load_metrics(self) -> None:
        metrics_path = self.models_dir / METRICS_FILENAME
        if not metrics_path.exists():
            log.warning("no %s found; metrics endpoint will be empty", METRICS_FILENAME)
            return
        try:
            self.metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.error("could not read %s: %s", metrics_path, exc)

    @staticmethod
    def _unpickle(path: Path):
        with open(path, "rb") as fh:
            return pickle.load(fh)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        return bool(self.diseases)

    def get(self, disease: str) -> DiseaseArtifacts:
        try:
            return self.diseases[disease]
        except KeyError:
            raise RegistryError(
                f"Unknown disease '{disease}'. Available: {sorted(self.diseases)}"
            ) from None

    def get_model(self, disease: str, model_type: str):
        artifacts = self.get(disease)
        try:
            return artifacts.models[model_type]
        except KeyError:
            raise RegistryError(
                f"Model '{model_type}' is not available for '{disease}'. "
                f"Available: {sorted(artifacts.models)}"
            ) from None

    # ------------------------------------------------------------------
    # API payloads
    # ------------------------------------------------------------------

    def catalog(self) -> dict:
        """The ``/api/diseases`` payload.

        Carries all the metadata the frontend needs to render a form, so the
        UI holds no hardcoded knowledge of any specific disease.
        """
        catalog = {}
        for name, artifacts in self.diseases.items():
            meta = disease_meta(name)
            catalog[name] = {
                "display_name": meta["display_name"],
                "description": meta["description"],
                "features": [
                    {"name": feature, **feature_meta(name, feature)}
                    for feature in artifacts.features
                ],
                "default_model": self.default_model(name),
                "models": [
                    {
                        "name": model_type,
                        "display_name": MODEL_DISPLAY_NAMES.get(model_type, model_type),
                    }
                    for model_type in sorted(artifacts.models)
                ],
            }
        return {"diseases": catalog}

    def default_model(self, disease: str) -> str:
        """Model the UI pre-selects, falling back if it was never trained."""
        models = self.get(disease).models
        return DEFAULT_MODEL if DEFAULT_MODEL in models else sorted(models)[0]

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, disease: str, model_type: str, values: list[float]) -> dict:
        """Run one prediction. ``values`` must already be validated."""
        artifacts = self.get(disease)
        model = self.get_model(disease, model_type)

        try:
            X = np.asarray([values], dtype=float)
            X_scaled = artifacts.scaler.transform(X)
            prediction = int(model.predict(X_scaled)[0])
            probability = self._positive_probability(model, X_scaled)
        except Exception as exc:  # noqa: BLE001 - surfaced as a 500 by the caller
            log.exception("inference failed for %s/%s", disease, model_type)
            raise InferenceError(str(exc)) from exc

        display_name = disease_meta(disease)["display_name"]
        return {
            "disease": disease,
            "model_used": model_type,
            "prediction": prediction,
            "label": "positive" if prediction == 1 else "negative",
            "summary": f"{'High' if prediction == 1 else 'Low'} risk of {display_name}",
            "probability": probability,
        }

    @staticmethod
    def _positive_probability(model, X_scaled) -> float | None:
        """Probability of the positive class, or ``None`` if unsupported.

        Indexes through ``classes_`` instead of assuming the positive class is
        column 1 - that assumption breaks on any model whose classes are
        ordered differently.
        """
        if not hasattr(model, "predict_proba"):
            return None
        classes = list(getattr(model, "classes_", []))
        if 1 not in classes:
            return None
        proba = model.predict_proba(X_scaled)[0]
        return round(float(proba[classes.index(1)]) * 100, 1)
