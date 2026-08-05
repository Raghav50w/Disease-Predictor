"""Regenerate the published metrics from models/metrics.json.

    python scripts/update_metrics_docs.py          # rewrite both files
    python scripts/update_metrics_docs.py --check  # fail if either is stale

Two outputs, from one source:

* the short table between the markers in ``README.md``
* the full breakdown in ``RESULTS.md`` - every metric, both cross-validation
  and held-out, with the split sizes

Generating them means the published numbers are always the ones training
actually produced. CI runs ``--check``, so a doc that has drifted from the
models fails the build.
"""

from __future__ import annotations

import argparse
import json
import sys

from healthscan.config import (
    BASE_DIR,
    CV_FOLDS,
    METRICS_FILENAME,
    MODEL_DISPLAY_NAMES,
    MODELS_DIR,
    TEST_SIZE,
)

README_PATH = BASE_DIR / "README.md"
RESULTS_PATH = BASE_DIR / "RESULTS.md"
BEGIN = "<!-- METRICS:BEGIN -->"
END = "<!-- METRICS:END -->"

# Column order used by both the CV and the held-out tables.
METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")
HEADERS = ("Accuracy", "Precision", "Recall", "F1", "ROC-AUC")


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _cell(metric: str, value: float | None) -> str:
    """ROC-AUC reads naturally as 0-1; everything else as a percentage."""
    return _num(value) if metric == "roc_auc" else _pct(value)


def _spread(metric: str, std: float) -> str:
    """Std in the same unit as the mean it follows - points, not fractions."""
    return f"{std:.3f}" if metric == "roc_auc" else f"{std * 100:.1f}"


def _table(header: str, rows: list[str]) -> list[str]:
    return [header, "| --- | " + " | ".join(["---"] * len(METRICS)) + " |", *rows, ""]


# ---------------------------------------------------------------- README


def render_readme_block(metrics: dict) -> str:
    """The short table that sits between the markers in README.md."""
    lines: list[str] = []

    for data in metrics["diseases"].values():
        balance = data["class_balance"]
        lines.append(
            f"**{data['display_name']}** - {data['n_samples']} rows, "
            f"{data['n_features']} features, "
            f"{balance['0']} negative / {balance['1']} positive"
        )
        lines.append("")
        lines.append("| Model | CV accuracy | Test accuracy | ROC-AUC |")
        lines.append("| --- | --- | --- | --- |")

        for model_key, scores in data["models"].items():
            name = MODEL_DISPLAY_NAMES.get(model_key, model_key)
            cv = scores["cross_validation"]["accuracy"]
            holdout = scores["holdout"]
            lines.append(
                f"| {name} "
                f"| {_pct(cv['mean'])} ± {cv['std'] * 100:.1f} "
                f"| {_pct(holdout['accuracy'])} "
                f"| {_num(holdout['roc_auc'])} |"
            )
        lines.append("")

    # No timestamp here on purpose: it would change on every training run, so
    # `--check` could never pass. It is still recorded in models/metrics.json.
    lines.append(
        f"Trained with scikit-learn {metrics['sklearn_version']}, "
        f"random_state={metrics['random_state']}. "
        f"CV accuracy is {CV_FOLDS}-fold cross-validation on the training data; the test "
        "columns are a 20% split the models never saw. Full per-model precision, "
        "recall and F1 are in [RESULTS.md](RESULTS.md)."
    )

    return "\n".join(lines)


# --------------------------------------------------------------- RESULTS


def render_results_doc(metrics: dict) -> str:
    """The full breakdown written to RESULTS.md."""
    lines = [
        "# Detailed results",
        "",
        "Generated from `models/metrics.json` by `scripts/update_metrics_docs.py`.",
        "Do not edit by hand - retrain and rerun the script instead.",
        "",
        f"scikit-learn {metrics['sklearn_version']}, "
        f"`random_state={metrics['random_state']}`, "
        f"{int(TEST_SIZE * 100)}% held-out test split, "
        f"{CV_FOLDS}-fold stratified cross-validation.",
        "",
        "## How to read this",
        "",
        f"**Cross-validation** is {CV_FOLDS}-fold on the *training* split only, so the",
        "`±` is the spread across folds - with datasets this small that spread",
        "matters more than any single number. **Held-out** is one fixed test split",
        "the models never saw during fitting or tuning.",
        "",
        "Scaling happens inside a `Pipeline`, so the `StandardScaler` is refit",
        "within each fold rather than on the full dataset. Accuracy is the least",
        "useful column here: these are imbalanced screening problems, so recall",
        "(how many positive cases were actually caught) is the number that",
        "matters, and it is consistently the weakest one.",
        "",
        "The `.pkl` artifacts that get served are refit on 100% of the data after",
        "this evaluation, so the scores below describe the same configuration but",
        "not literally the same fitted objects.",
        "",
    ]

    for data in metrics["diseases"].values():
        balance = data["class_balance"]
        lines += [
            f"## {data['display_name']}",
            "",
            f"- {data['n_samples']} rows, {data['n_features']} features",
            f"- {data['n_train']} train / {data['n_test']} test",
            f"- Class balance: {balance['0']} negative / {balance['1']} positive "
            f"({int(balance['1']) / (int(balance['0']) + int(balance['1'])) * 100:.1f}% positive)",
            "",
            f"### Cross-validation ({CV_FOLDS}-fold, training split, mean ± std)",
            "",
        ]

        cv_rows = []
        holdout_rows = []
        for model_key, scores in data["models"].items():
            name = MODEL_DISPLAY_NAMES.get(model_key, model_key)

            cells = []
            for metric in METRICS:
                fold = scores["cross_validation"][metric]
                cells.append(f"{_cell(metric, fold['mean'])} ± {_spread(metric, fold['std'])}")
            cv_rows.append(f"| {name} | " + " | ".join(cells) + " |")

            holdout = scores["holdout"]
            holdout_rows.append(
                f"| {name} | " + " | ".join(_cell(m, holdout[m]) for m in METRICS) + " |"
            )

        header = "| Model | " + " | ".join(HEADERS) + " |"
        lines += _table(header, cv_rows)
        lines += [f"### Held-out test set ({data['n_test']} rows)", ""]
        lines += _table(header, holdout_rows)

    return "\n".join(lines)


def splice(readme: str, block: str) -> str:
    start = readme.index(BEGIN) + len(BEGIN)
    end = readme.index(END)
    return readme[:start] + "\n\n" + block + "\n\n" + readme[end:]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if either file is out of date instead of rewriting it",
    )
    args = parser.parse_args()

    metrics_path = MODELS_DIR / METRICS_FILENAME
    if not metrics_path.exists():
        print(f"error: {metrics_path} not found - run `python -m healthscan.ml.train`")
        return 1

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    readme = README_PATH.read_text(encoding="utf-8")

    if BEGIN not in readme or END not in readme:
        print(f"error: README.md is missing the {BEGIN} / {END} markers")
        return 1

    wanted = {
        README_PATH: splice(readme, render_readme_block(metrics)),
        RESULTS_PATH: render_results_doc(metrics),
    }

    stale = [
        path
        for path, content in wanted.items()
        if not path.exists() or path.read_text(encoding="utf-8") != content
    ]

    if args.check:
        if stale:
            names = ", ".join(p.name for p in stale)
            print(f"error: {names} out of date - run `python scripts/update_metrics_docs.py`")
            return 1
        print("README.md and RESULTS.md are up to date.")
        return 0

    if not stale:
        print("README.md and RESULTS.md already up to date.")
        return 0

    for path in stale:
        path.write_text(wanted[path], encoding="utf-8")
        print(f"updated {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
