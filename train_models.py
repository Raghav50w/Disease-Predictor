"""
Dynamic multi-disease and multi-model training script.

Loops through every .csv file in datasets/, trains Random Forest, Naive Bayes,
MLP, and Perceptron for each, and saves the models, scaler, and feature list
into the models/ folder.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RANDOM_STATE = 42
TEST_SIZE = 0.2
KFOLD = 5


def replace_invalid_zeros(df, feature_cols):
    """Replace physiological zeros with column median."""
    zero_invalid_keywords = [
        "glucose", "blood_pressure", "resting_bp", "skin_thickness",
        "insulin", "bmi", "cholesterol", "max_heart_rate"
    ]
    for col in feature_cols:
        col_lower = col.lower()
        if any(kw in col_lower for kw in zero_invalid_keywords):
            pos = df[col] > 0
            if pos.any():
                med = df.loc[pos, col].median()
                df.loc[df[col] == 0, col] = med
    return df


def get_models():
    """Return dictionary of un-initialized model instances to train."""
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=None, min_samples_split=5, 
            min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "naive_bayes": GaussianNB(),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(100, 50), max_iter=500,
            random_state=RANDOM_STATE
        ),
        "perceptron": Perceptron(
            max_iter=1000, random_state=RANDOM_STATE
        )
    }


def train_disease(csv_path, disease_name):
    """Train multiple models for a single disease dataset."""
    print(f"\n{'='*60}")
    print(f"  Training models for: {disease_name}")
    print(f"{'='*60}")

    df = pd.read_csv(csv_path)
    feature_cols = list(df.columns[:-1])
    target_col = df.columns[-1]

    df = replace_invalid_zeros(df, feature_cols)

    X = df[feature_cols].values
    y = df[target_col].astype(int).values

    # Scale full data for saving models
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Save scaler and features
    scaler_path = os.path.join(MODELS_DIR, f"{disease_name}_scaler.pkl")
    features_path = os.path.join(MODELS_DIR, f"{disease_name}_features.pkl")
    
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    with open(features_path, "wb") as f:
        pickle.dump(feature_cols, f)
        
    print(f"  Saved: {os.path.basename(scaler_path)}")
    print(f"  Saved: {os.path.basename(features_path)}")

    models = get_models()
    
    # Train each model type
    for model_name, model in models.items():
        print(f"\n  >> Training {model_name}...")
        model.fit(X_scaled, y)
        
        model_path = os.path.join(MODELS_DIR, f"{disease_name}_{model_name}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
            
        print(f"     Saved: {os.path.basename(model_path)}")

    return True


def main():
    if not os.path.isdir(DATASETS_DIR):
        print(f"Error: datasets/ folder not found at {DATASETS_DIR}")
        sys.exit(1)

    csv_files = sorted([f for f in os.listdir(DATASETS_DIR) if f.endswith(".csv")])
    if not csv_files:
        print("No .csv files found in datasets/ folder.")
        sys.exit(1)

    # Clean old models
    for f in os.listdir(MODELS_DIR):
        if f.endswith(".pkl"):
            os.remove(os.path.join(MODELS_DIR, f))

    print(f"Found {len(csv_files)} dataset(s): {csv_files}")

    for csv_file in csv_files:
        disease_name = os.path.splitext(csv_file)[0]
        csv_path = os.path.join(DATASETS_DIR, csv_file)
        train_disease(csv_path, disease_name)

    print(f"\n{'='*60}")
    print(f"  All models trained and saved successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
