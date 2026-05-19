"""
Feature Importance Analysis Script

This script demonstrates how the top features were selected for the 
Heart Disease and Breast Cancer datasets. It trains a Random Forest model
on the full raw datasets and extracts the 'feature_importances_' metric to
prove mathematically which physiological inputs are the most predictive.
"""

import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "datasets", "raw")

def analyze_heart_disease():
    print("\n" + "="*50)
    print(" HEART DISEASE - FEATURE IMPORTANCE ANALYSIS")
    print("="*50)
    
    raw_path = os.path.join(RAW_DIR, "processed.cleveland.data")
    if not os.path.exists(raw_path):
        print("Raw heart disease data not found.")
        return

    cols = [
        "age", "sex", "chest_pain_type", "resting_bp", "cholesterol",
        "fasting_blood_sugar", "resting_ecg", "max_heart_rate",
        "exercise_angina", "st_depression", "st_slope",
        "num_vessels", "thalassemia", "target"
    ]
    
    df = pd.read_csv(raw_path, header=None, names=cols, na_values="?")
    
    # Quick clean for analysis
    for col in df.columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
            
    df["target"] = (df["target"] > 0).astype(int)
    
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X, y)
    
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)
    
    print(f"Total Original Features: {len(X.columns)}")
    print("\nRanked Features by Predictive Importance:")
    for i, (feat, imp) in enumerate(importances.items(), 1):
        marker = "<-- SELECTED" if i <= 6 else ""
        print(f"  {i}. {feat.ljust(20)}: {imp:.4f}  {marker}")

def analyze_breast_cancer():
    print("\n" + "="*50)
    print(" BREAST CANCER - FEATURE IMPORTANCE ANALYSIS")
    print("="*50)
    
    raw_path = os.path.join(RAW_DIR, "wdbc.data")
    if not os.path.exists(raw_path):
        print("Raw breast cancer data not found.")
        return

    feature_names = [
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
        "smoothness_mean", "compactness_mean", "concavity_mean",
        "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
        "radius_se", "texture_se", "perimeter_se", "area_se",
        "smoothness_se", "compactness_se", "concavity_se",
        "concave_points_se", "symmetry_se", "fractal_dimension_se",
        "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
        "smoothness_worst", "compactness_worst", "concavity_worst",
        "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
    ]
    
    cols = ["id", "diagnosis"] + feature_names
    df = pd.read_csv(raw_path, header=None, names=cols)
    
    df.drop(columns=["id"], inplace=True)
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
    
    # Move target to end
    target = df.pop("diagnosis")
    df["target"] = target
    
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X, y)
    
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)
    
    print(f"Total Original Features: {len(X.columns)}")
    print("\nTop 15 Ranked Features by Predictive Importance:")
    for i, (feat, imp) in enumerate(importances.head(15).items(), 1):
        marker = "<-- SELECTED" if i <= 6 else ""
        print(f"  {i}. {feat.ljust(20)}: {imp:.4f}  {marker}")

if __name__ == "__main__":
    analyze_heart_disease()
    analyze_breast_cancer()
