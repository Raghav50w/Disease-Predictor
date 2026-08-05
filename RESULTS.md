# Detailed results

Generated from `models/metrics.json` by `scripts/update_metrics_docs.py`.
Do not edit by hand - retrain and rerun the script instead.

scikit-learn 1.8.0, `random_state=42`, 20% held-out test split, 5-fold stratified cross-validation.

## How to read this

**Cross-validation** is 5-fold on the *training* split only, so the
`±` is the spread across folds - with datasets this small that spread
matters more than any single number. **Held-out** is one fixed test split
the models never saw during fitting or tuning.

Scaling happens inside a `Pipeline`, so the `StandardScaler` is refit
within each fold rather than on the full dataset. Accuracy is the least
useful column here: these are imbalanced screening problems, so recall
(how many positive cases were actually caught) is the number that
matters, and it is consistently the weakest one.

The `.pkl` artifacts that get served are refit on 100% of the data after
this evaluation, so the scores below describe the same configuration but
not literally the same fitted objects.

## Breast Cancer

- 569 rows, 6 features
- 455 train / 114 test
- Class balance: 357 negative / 212 positive (37.3% positive)

### Cross-validation (5-fold, training split, mean ± std)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Random Forest | 94.9% ± 2.6 | 94.2% ± 4.5 | 92.4% ± 4.0 | 93.2% ± 3.4 | 0.980 ± 0.011 |
| Naive Bayes | 94.9% ± 2.0 | 94.1% ± 3.5 | 92.4% ± 3.5 | 93.2% ± 2.7 | 0.984 ± 0.010 |
| Neural Network (MLP) | 95.8% ± 2.2 | 96.4% ± 3.5 | 92.4% ± 3.0 | 94.3% ± 3.0 | 0.974 ± 0.029 |
| Perceptron | 94.7% ± 1.9 | 93.2% ± 3.8 | 92.9% ± 4.8 | 92.9% ± 2.6 | 0.987 ± 0.011 |

### Held-out test set (114 rows)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Random Forest | 94.7% | 97.4% | 88.1% | 92.5% | 0.993 |
| Naive Bayes | 94.7% | 95.0% | 90.5% | 92.7% | 0.993 |
| Neural Network (MLP) | 98.2% | 100.0% | 95.2% | 97.6% | 0.993 |
| Perceptron | 97.4% | 97.6% | 95.2% | 96.4% | 0.996 |

## Diabetes

- 768 rows, 4 features
- 614 train / 154 test
- Class balance: 500 negative / 268 positive (34.9% positive)

### Cross-validation (5-fold, training split, mean ± std)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Random Forest | 75.1% ± 2.6 | 66.2% ± 5.2 | 58.9% ± 4.2 | 62.2% ± 3.7 | 0.816 ± 0.021 |
| Naive Bayes | 77.5% ± 1.9 | 72.9% ± 3.7 | 56.6% ± 2.9 | 63.7% ± 3.0 | 0.829 ± 0.009 |
| Neural Network (MLP) | 73.6% ± 2.3 | 63.0% ± 3.5 | 58.9% ± 4.9 | 60.8% ± 4.1 | 0.775 ± 0.051 |
| Perceptron | 70.5% ± 4.6 | 58.6% ± 9.0 | 58.9% ± 8.3 | 58.2% ± 5.9 | 0.782 ± 0.036 |

### Held-out test set (154 rows)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Random Forest | 74.7% | 66.0% | 57.4% | 61.4% | 0.809 |
| Naive Bayes | 70.8% | 60.0% | 50.0% | 54.5% | 0.754 |
| Neural Network (MLP) | 75.3% | 66.0% | 61.1% | 63.5% | 0.844 |
| Perceptron | 68.2% | 54.5% | 55.6% | 55.0% | 0.736 |

## Heart Disease

- 303 rows, 6 features
- 242 train / 61 test
- Class balance: 164 negative / 139 positive (45.9% positive)

### Cross-validation (5-fold, training split, mean ± std)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Random Forest | 80.6% ± 2.1 | 79.6% ± 4.7 | 78.3% ± 6.2 | 78.7% ± 2.5 | 0.884 ± 0.020 |
| Naive Bayes | 82.2% ± 4.0 | 83.5% ± 9.0 | 78.3% ± 6.8 | 80.3% ± 3.9 | 0.877 ± 0.023 |
| Neural Network (MLP) | 78.5% ± 2.0 | 77.5% ± 2.0 | 74.7% ± 3.8 | 76.1% ± 2.8 | 0.839 ± 0.022 |
| Perceptron | 71.4% ± 13.6 | 68.6% ± 15.9 | 71.0% ± 16.8 | 69.4% ± 14.9 | 0.768 ± 0.160 |

### Held-out test set (61 rows)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Random Forest | 85.2% | 80.6% | 89.3% | 84.7% | 0.951 |
| Naive Bayes | 83.6% | 76.5% | 92.9% | 83.9% | 0.933 |
| Neural Network (MLP) | 78.7% | 69.2% | 96.4% | 80.6% | 0.868 |
| Perceptron | 73.8% | 66.7% | 85.7% | 75.0% | 0.775 |
