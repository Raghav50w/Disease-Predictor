import os
import sys
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

DATA_PATH = "diabetes.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2
KFOLD = 5

def find_column(df, names):
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n and n.lower() in lower:
            return lower[n.lower()]
    return None

def replace_invalid_zeros(df, cols):
    for c in cols:
        if c in df.columns:
            pos = df[c] > 0
            med = df.loc[pos, c].median() if pos.any() else df[c].median()
            df.loc[df[c] == 0, c] = med
    return df

def load_data(path):
    if not os.path.exists(path):
        print("Data file missing:", path)
        sys.exit(1)
    return pd.read_csv(path)

def main():
    df = load_data(DATA_PATH)

    target_col = find_column(df, ["Outcome","diabetes","target","label","class"])
    if target_col is None:
        target_col = df.columns[-1]

    age_col      = find_column(df, ["age"])
    glucose_col  = find_column(df, ["glucose","blood sugar"])
    insulin_col  = find_column(df, ["insulin"])
    bmi_col      = find_column(df, ["bmi"])

    if None in [age_col, glucose_col, insulin_col, bmi_col]:
        print("Required columns not found.")
        sys.exit(1)

    df = replace_invalid_zeros(df, [glucose_col, insulin_col, bmi_col])

    X = df[[age_col, glucose_col, insulin_col, bmi_col]].values

    try:
        y = df[target_col].astype(int).values
    except:
        vals = pd.unique(df[target_col])
        if len(vals) == 2:
            mapping = {vals[0]:0, vals[1]:1}
            y = np.array([mapping[v] for v in df[target_col]])
        else:
            raise

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    nb = GaussianNB()
    nb.fit(X_train, y_train)
    print("\nGaussianNB Results:")
    print(classification_report(y_test, nb.predict(X_test)))

    mlp = MLPClassifier(hidden_layer_sizes=(50,), activation='relu',
                        solver='adam', max_iter=1000, random_state=RANDOM_STATE)
    mlp.fit(X_train_scaled, y_train)
    print("\nMLPClassifier Results:")
    print(classification_report(y_test, mlp.predict(X_test_scaled)))

    scoring = ['accuracy','precision','recall','f1']
    kf = KFold(n_splits=KFOLD, shuffle=True, random_state=RANDOM_STATE)

    print(f"\n{KFOLD}-Fold CV (GaussianNB):")
    cv_nb = cross_validate(nb, X, y, scoring=scoring, cv=kf)
    for m in scoring:
        v = cv_nb[f'test_{m}']
        print(f"{m}: {v.mean():.4f} +/- {v.std():.4f}")

    print(f"\n{KFOLD}-Fold CV (MLP):")
    X_scaled_full = scaler.fit_transform(X)
    cv_mlp = cross_validate(mlp, X_scaled_full, y, scoring=scoring, cv=kf)
    for m in scoring:
        v = cv_mlp[f'test_{m}']
        print(f"{m}: {v.mean():.4f} +/- {v.std():.4f}")

    with open("naive_bayes_model.pkl","wb") as f:
        pickle.dump(nb, f)
    with open("mlp_model.pkl","wb") as f:
        pickle.dump(mlp, f)
    with open("scaler.pkl","wb") as f:
        pickle.dump(scaler, f)

    print("\nModels saved.")

if __name__ == "__main__":
    main()
