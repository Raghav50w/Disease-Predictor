"""
One-time script to convert raw .data files into clean .csv files
and organize them into the datasets/ folder.

This script:
1. Converts processed.cleveland.data -> datasets/heart_disease.csv (top 6 features)
2. Converts wdbc.data -> datasets/breast_cancer.csv (top 6 features)
3. Copies diabetes.csv -> datasets/diabetes.csv
"""

import os
import shutil
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")

os.makedirs(DATASETS_DIR, exist_ok=True)


# -----------------------------------------------
# 1. Heart Disease (Cleveland) - Top 6 features
# -----------------------------------------------
print("Processing: processed.cleveland.data -> heart_disease.csv")

all_heart_cols = [
    "age", "sex", "chest_pain_type", "resting_bp", "cholesterol",
    "fasting_blood_sugar", "resting_ecg", "max_heart_rate",
    "exercise_angina", "st_depression", "st_slope",
    "num_vessels", "thalassemia", "target"
]

heart_df = pd.read_csv(
    os.path.join(BASE_DIR, "processed.cleveland.data"),
    header=None,
    names=all_heart_cols,
    na_values="?"
)

# Replace missing values with column median
for col in heart_df.columns:
    if heart_df[col].isna().any():
        median_val = heart_df[col].median()
        heart_df[col] = heart_df[col].fillna(median_val)
        print(f"  Filled {col} missing values with median={median_val:.2f}")

# Binarize target: 0 = healthy, >0 = disease present (1)
heart_df["target"] = (heart_df["target"] > 0).astype(int)

# Keep only the top 6 most important features + target
heart_keep = ["chest_pain_type", "thalassemia", "num_vessels",
              "max_heart_rate", "st_depression", "age", "target"]
heart_df = heart_df[heart_keep]

heart_df.to_csv(os.path.join(DATASETS_DIR, "heart_disease.csv"), index=False)
print(f"  Saved: datasets/heart_disease.csv  ({len(heart_df)} rows, {len(heart_df.columns)} cols)")
print(f"  Features kept: {heart_keep[:-1]}")
print(f"  Target distribution: {dict(heart_df['target'].value_counts())}")


# -----------------------------------------------
# 2. Breast Cancer (WDBC) - Top 6 features
# -----------------------------------------------
print("\nProcessing: wdbc.data -> breast_cancer.csv")

wdbc_feature_names = [
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

wdbc_cols = ["id", "diagnosis"] + wdbc_feature_names

wdbc_df = pd.read_csv(
    os.path.join(BASE_DIR, "wdbc.data"),
    header=None,
    names=wdbc_cols
)

# Drop patient ID column
wdbc_df.drop(columns=["id"], inplace=True)

# Convert diagnosis: M=1 (Malignant), B=0 (Benign)
wdbc_df["diagnosis"] = wdbc_df["diagnosis"].map({"M": 1, "B": 0})

# Rename diagnosis to target and move to last column
target = wdbc_df.pop("diagnosis")

# Keep only the top 6 most important features
bc_keep = ["perimeter_worst", "area_worst", "concave_points_worst",
           "concave_points_mean", "radius_worst", "area_mean"]
wdbc_df = wdbc_df[bc_keep]
wdbc_df["target"] = target

wdbc_df.to_csv(os.path.join(DATASETS_DIR, "breast_cancer.csv"), index=False)
print(f"  Saved: datasets/breast_cancer.csv  ({len(wdbc_df)} rows, {len(wdbc_df.columns)} cols)")
print(f"  Features kept: {bc_keep}")
print(f"  Target distribution: {dict(wdbc_df['target'].value_counts())}")


# -----------------------------------------------
# 3. Diabetes (just copy to datasets/)
# -----------------------------------------------
print("\nCopying: diabetes.csv -> datasets/diabetes.csv")

src = os.path.join(BASE_DIR, "diabetes.csv")
dst = os.path.join(DATASETS_DIR, "diabetes.csv")
shutil.copy2(src, dst)

diabetes_df = pd.read_csv(dst)
print(f"  Saved: datasets/diabetes.csv  ({len(diabetes_df)} rows, {len(diabetes_df.columns)} cols)")

print("\nAll datasets prepared successfully!")
