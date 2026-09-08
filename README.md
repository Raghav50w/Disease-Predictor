# HealthScanner

**Live demo: <https://healthscanner-s7nv.onrender.com>**
(free instance, so the first load can take ~50 seconds while it wakes up)

A multi-disease risk predictor I built to learn how a machine learning model
gets from a CSV to something usable in a browser.

It trains four scikit-learn classifiers on three public medical datasets and
serves them through a FastAPI backend with a plain HTML/JS frontend. Pick a
condition, pick a model, enter some measurements, get a prediction and the
model's estimated probability.

**Not a medical tool.** Trained on small public research datasets, never
clinically validated. Don't use it for any real health decision.

## What it predicts

| Condition | Dataset | Rows | Features |
|---|---|---|---|
| Diabetes | Pima Indians Diabetes Database | 768 | 4 |
| Heart Disease | UCI Cleveland | 303 | 6 |
| Breast Cancer | UCI Wisconsin Diagnostic (WDBC) | 569 | 6 |

Each is served by a Random Forest, Gaussian Naive Bayes, an MLP neural network,
and a Perceptron — you choose which at request time.

The two larger datasets were cut to their six most useful columns using Random
Forest feature importances (30 → 6 for WDBC, 13 → 6 for Cleveland).

## How it fits together

```
datasets/raw/ -> preprocess.py -> datasets/*.csv -> train.py -> models/
                                                                  |
                                            main.py (FastAPI) <---+
                                                  |
                                            static/ (browser)
```

## Running it

```bash
pip install -r requirements.txt
python -m healthscan.ml.preprocess
python -m healthscan.ml.train
uvicorn healthscan.main:app --reload
```

Then open <http://127.0.0.1:8000>.

## Model performance

Scored on a held-out 20% test split. CV Accuracy is 5-fold stratified
cross-validation on the training split only.

Recall is the number that matters most here — for a screening tool, missing a
positive case is worse than a false alarm.

### Diabetes (768 rows, 4 features)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV Accuracy |
|---|---|---|---|---|---|---|
| Random Forest | 0.747 | 0.660 | 0.574 | 0.614 | 0.809 | 0.751 ± 0.026 |
| Naive Bayes | 0.708 | 0.600 | 0.500 | 0.545 | 0.754 | 0.775 ± 0.019 |
| Neural Network (MLP) | 0.753 | 0.660 | 0.611 | 0.635 | 0.844 | 0.736 ± 0.023 |
| Perceptron | 0.682 | 0.545 | 0.556 | 0.550 | - | 0.705 ± 0.046 |

### Heart Disease (303 rows, 6 features)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV Accuracy |
|---|---|---|---|---|---|---|
| Random Forest | 0.852 | 0.806 | 0.893 | 0.847 | 0.951 | 0.806 ± 0.021 |
| Naive Bayes | 0.836 | 0.765 | 0.929 | 0.839 | 0.933 | 0.822 ± 0.040 |
| Neural Network (MLP) | 0.787 | 0.692 | 0.964 | 0.806 | 0.868 | 0.785 ± 0.020 |
| Perceptron | 0.738 | 0.667 | 0.857 | 0.750 | - | 0.714 ± 0.136 |

### Breast Cancer (569 rows, 6 features)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV Accuracy |
|---|---|---|---|---|---|---|
| Random Forest | 0.947 | 0.974 | 0.881 | 0.925 | 0.993 | 0.949 ± 0.026 |
| Naive Bayes | 0.947 | 0.950 | 0.905 | 0.927 | 0.993 | 0.949 ± 0.020 |
| Neural Network (MLP) | 0.982 | 1.000 | 0.952 | 0.976 | 0.993 | 0.958 ± 0.022 |
| Perceptron | 0.974 | 0.976 | 0.952 | 0.964 | - | 0.947 ± 0.019 |

Perceptron has no `predict_proba`, so it reports no ROC-AUC and no probability.

## The API

```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"disease":"diabetes","model_type":"random_forest","Glucose":150,"Insulin":120,"BMI":34.2,"Age":52}'
```

`GET /api/diseases` lists the conditions, their input fields and bounds.
`GET /api/metrics` returns the table above as JSON. `GET /docs` is the
auto-generated API reference.

## Things I know are wrong with it

The accuracy numbers look better than they deserve to be trusted:

- **Feature selection saw the test data.** The six features were chosen by
  fitting a Random Forest on the *full* dataset, so the held-out scores are
  mildly optimistic — the choice of columns was informed by rows the models are
  then scored on.
- **Zero-imputation leaks too.** Impossible zeros in the diabetes data (374 of
  768 insulin readings) are replaced with the column median, which squashes
  that feature's variance. Those medians are computed over the whole dataset
  before the train/test split.
- **The datasets are tiny.** 303 heart disease records is nowhere near enough
  to generalise. The breast cancer scores look great largely because WDBC is an
  easy, well-separated dataset.
- **One test split.** With 61 test samples for heart disease, one extra mistake
  moves accuracy by more than a point. The `±` on the CV column is the more
  honest number.
- **The probabilities aren't calibrated.** They're raw model outputs, not real
  risk percentages.
- **The populations don't match yours.** The Pima data is women aged 21+ of
  Pima heritage; the Cleveland data is from 1988.
- **The shipped models are refit on 100% of the data.** The scores above come
  from identically-configured models trained on 80%.

## Layout

```
healthscan/       config, FastAPI app, model loading
healthscan/ml/    preprocessing, training, evaluation
static/           the frontend
tests/            14 tests
datasets/raw/     the original UCI files
```

## Credits

Datasets from the [UCI Machine Learning
Repository](https://archive.ics.uci.edu/). MIT licensed — see `LICENSE`.
