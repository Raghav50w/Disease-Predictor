# HealthScanner

**Live demo: <https://healthscanner-s7nv.onrender.com>**
(it's on a free instance, so the first load can take ~50 seconds while it wakes up)

A multi-disease risk predictor I built to learn how a machine learning model
actually gets from a CSV to something people can use in a browser.

It trains four scikit-learn classifiers on three public medical datasets, then
serves them through a small Flask API with a plain HTML/JS frontend. You pick a
condition, pick a model, type in some measurements, and it gives you a
prediction with the model's estimated probability.

**Not a medical tool.** It is trained on small public research datasets and has
never been clinically validated, so please don't use it to make any real health
decision.

## What it predicts

- **Diabetes** - Pima Indians Diabetes Database, 768 rows, 4 features
- **Heart Disease** - UCI Cleveland, 303 rows, 6 features
- **Breast Cancer** - UCI Wisconsin Diagnostic (WDBC), 569 rows, 6 features

Each one is served by a Random Forest, Gaussian Naive Bayes, an MLP neural
network, and a Perceptron. You choose which at request time.

The two bigger datasets were cut down to their six most useful features using
Random Forest feature importances. Run
`python -m healthscan.ml.feature_analysis` to print the ranking that decided
which ones to keep.

## How it fits together

```
datasets/raw/  ->  preprocess.py  ->  datasets/*.csv  ->  train.py  ->  models/*.pkl
                                                                            |
                                            static/index.html  <->  Flask API
```

Everything about a feature - its label, unit, min/max, help text, dropdown
options, and default value - lives in one dict called `DISEASE_META` in
`src/healthscan/config.py`. The API sends that to the frontend, and the
frontend builds the form from it. So the HTML and JavaScript know nothing
about diabetes or heart disease specifically, and adding a fourth condition
means adding a `DISEASE_META` entry and a CSV, with no frontend changes.

## Setup

You need Python 3.11 or newer, and git.

**1. Download the code**

```bash
git clone https://github.com/Raghav50w/Disease-Predictor.git
```

```bash
cd Disease-Predictor
```

**2. Make a virtual environment**

```bash
python -m venv venv
```

Then activate it. On Windows:

```bash
venv\Scripts\activate
```

On macOS or Linux:

```bash
source venv/bin/activate
```

**3. Install everything**

```bash
pip install -e ".[dev]"
```

The `-e` installs the project itself so `python -m healthscan.…` works from
anywhere, and `[dev]` adds pytest and ruff.

## Running it

The trained models are not committed to git (they're big binary files that go
stale), so build them first. This only needs doing once, or again after you
change a dataset or the training code.

**1. Build the clean CSVs from the raw UCI files**

```bash
python -m healthscan.ml.preprocess
```

**2. Train the models**

```bash
python -m healthscan.ml.train
```

This writes `models/*.pkl` and `models/metrics.json`. Takes about a minute.

**3. Start the server**

```bash
python -m healthscan.wsgi
```

Now open <http://127.0.0.1:5000> in a browser.

If you see "No trained models are available", step 2 didn't run or didn't
finish - check that `models/` has `.pkl` files in it.

## Model performance

These numbers come from `python -m healthscan.ml.train` and are copied into
this README by a script, so they're always the ones training actually produced.

Scaling happens inside a scikit-learn `Pipeline`, which means the
`StandardScaler` is refit inside each cross-validation fold and on the training
data only. Fitting the scaler on everything before splitting is a common
mistake that leaks test data into training and makes every score look better
than it is.

<!-- METRICS:BEGIN -->

**Breast Cancer** - 569 rows, 6 features, 357 negative / 212 positive

| Model | CV accuracy | Test accuracy | ROC-AUC |
| --- | --- | --- | --- |
| Random Forest | 94.9% ± 2.6 | 94.7% | 0.993 |
| Naive Bayes | 94.9% ± 2.0 | 94.7% | 0.993 |
| Neural Network (MLP) | 95.8% ± 2.2 | 98.2% | 0.993 |
| Perceptron | 94.7% ± 1.9 | 97.4% | 0.996 |

**Diabetes** - 768 rows, 4 features, 500 negative / 268 positive

| Model | CV accuracy | Test accuracy | ROC-AUC |
| --- | --- | --- | --- |
| Random Forest | 75.1% ± 2.6 | 74.7% | 0.809 |
| Naive Bayes | 77.5% ± 1.9 | 70.8% | 0.754 |
| Neural Network (MLP) | 73.6% ± 2.3 | 75.3% | 0.844 |
| Perceptron | 70.5% ± 4.6 | 68.2% | 0.736 |

**Heart Disease** - 303 rows, 6 features, 164 negative / 139 positive

| Model | CV accuracy | Test accuracy | ROC-AUC |
| --- | --- | --- | --- |
| Random Forest | 80.6% ± 2.1 | 85.2% | 0.951 |
| Naive Bayes | 82.2% ± 4.0 | 83.6% | 0.933 |
| Neural Network (MLP) | 78.5% ± 2.0 | 78.7% | 0.868 |
| Perceptron | 71.4% ± 13.6 | 73.8% | 0.775 |

Trained with scikit-learn 1.8.0, random_state=42. CV accuracy is 5-fold cross-validation on the training data; the test columns are a 20% split the models never saw. Full per-model precision, recall and F1 are in [RESULTS.md](RESULTS.md).

<!-- METRICS:END -->

The `.pkl` files that actually get served are refit on 100% of the data after
this evaluation is done. The scores above come from data those models hadn't
seen at the time they were scored.

To regenerate both the result files after retraining:

```bash
python scripts/update_metrics_docs.py
```

## The API

- `GET /` - the web page
- `GET /healthz` - health check, says whether the models loaded
- `GET /api/diseases` - every condition, its features, and its models
- `GET /api/metrics` - the full contents of `models/metrics.json`
- `POST /api/predict` - make one prediction

Example:

```bash
curl -X POST http://127.0.0.1:5000/api/predict -H 'Content-Type: application/json' -d '{"disease":"diabetes","model_type":"random_forest","Glucose":150,"Insulin":120,"BMI":34.2,"Age":52}'
```

```json
{
  "disease": "diabetes",
  "model_used": "random_forest",
  "prediction": 1,
  "label": "positive",
  "summary": "High risk of Diabetes",
  "probability": 71.5
}
```

If something is missing or out of range you get a `400` with a message per
field, which is what the frontend shows under each input:

```json
{
  "error": "One or more inputs are invalid.",
  "fields": { "Glucose": "Plasma Glucose must be at most 400 mg/dL." }
}
```

An unknown condition or model gives a `404`. Every error is JSON, never an
HTML error page.

### Settings

All optional, set as environment variables:

- `PORT` - port to run on, default `5000`
- `FLASK_DEBUG` - set to `1` for the dev debugger. Never turn this on anywhere
  public, it lets anyone run code on the machine
- `HEALTHSCAN_LOG_LEVEL` - default `INFO`
- `HEALTHSCAN_MODELS_DIR` - where the `.pkl` files are, default `./models`
- `HEALTHSCAN_DATASETS_DIR` - where the CSVs are, default `./datasets`
- `HEALTHSCAN_STATIC_DIR` - where the frontend is, default `./static`

## Docker

The image trains the models during the build, so you don't need to run
anything beforehand:

```bash
docker build -t healthscanner .
```

```bash
docker run -p 5000:5000 healthscanner
```

Then open <http://127.0.0.1:5000>.

It's a two-stage build. The first stage installs the dependencies and runs
preprocess and train; the second copies just the finished models and the
installed packages into a clean image and runs it as a non-root user with
gunicorn.

## Tests and linting

```bash
pytest
```

```bash
pytest --cov=healthscan
```

```bash
ruff check .
```

```bash
ruff format .
```

The tests train their own tiny models into a temp folder, so they pass on a
fresh clone before you've trained anything.

## CI/CD

`.github/workflows/ci.yml` runs on every push and pull request:

1. **Lint** - `ruff check` and `ruff format --check`
2. **Build datasets** - `python -m healthscan.ml.preprocess`
3. **Train** - `python -m healthscan.ml.train`, which also proves the training
   code still runs
4. **Test** - `pytest --cov`
5. **Check the docs** - fails if README.md or RESULTS.md no longer match the models
6. **Build the Docker image** - then start the container and check `/healthz`
   reports `ready:true`
7. **Deploy** - only on `main`, and only if everything above passed

### Setting up deployment

Deployment goes to [Render](https://render.com) using `render.yaml`, and it is
optional - without it CI still runs everything else and just skips the last
step.

1. On Render, choose **New → Blueprint** and point it at this repo. It reads
   `render.yaml` and creates the service.
2. Open the new service's settings and copy its **deploy hook URL**.
3. In this repo on GitHub, go to **Settings → Secrets and variables → Actions**
   and add a secret named `RENDER_DEPLOY_HOOK_URL` with that URL.

`render.yaml` sets `autoDeploy: false`, so Render ignores pushes on its own.
The only thing that triggers a deploy is the CI job, and that only runs after
the tests and the image check pass - so a broken commit can't reach the live
site.

## Project layout

```
src/healthscan/
  config.py              paths, constants, DISEASE_META
  wsgi.py                entrypoint for gunicorn and the dev server
  api/
    __init__.py          create_app() factory
    routes.py            the endpoints and the JSON error handlers
    registry.py          loads the .pkl files, runs predictions
    validation.py        checks incoming requests
  ml/
    preprocess.py        raw .data files -> clean .csv
    train.py             train, evaluate, save
    evaluate.py          cross-validation and held-out scoring
    feature_analysis.py  feature importance ranking
tests/                   tests for the API, registry, validation, training
scripts/                 regenerates the metrics in README.md and RESULTS.md
static/                  index.html, css/, js/ - no build step, no npm
datasets/raw/            the original UCI files
```

## Things I know are wrong with it

Worth being upfront about, because the accuracy numbers look better than they
deserve to be trusted:

- **The datasets are tiny.** 303 heart disease records is nowhere near enough
  to generalise. The breast cancer scores look great mostly because WDBC is an
  easy, well-separated dataset.
- **One test split.** With 61 test samples for heart disease, a single extra
  mistake moves accuracy by more than a point. The `±` on the CV column is the
  more honest number.
- **The probabilities aren't calibrated.** They're raw model outputs. Random
  Forest probabilities especially should not be read as a real risk percentage.
- **The populations don't match yours.** The Pima data is women aged 21+ of
  Pima heritage, and the Cleveland data is from 1988.
- **Zero-imputation.** Impossible zeros in the diabetes data (374 of 768
  insulin readings) get replaced with the column median, which squashes that
  feature's variance. Those medians are also worked out over the whole dataset
  before the train/test split, so unlike the scaler they do leak a little test
  information into training.
