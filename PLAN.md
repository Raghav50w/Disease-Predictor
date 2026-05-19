# Multi-Disease Predictor — Final Improvement Plan

## What This Plan Covers
A focused, practical upgrade to the existing Diabetes Predictor to support three diseases (Diabetes, Heart Disease, Breast Cancer), a better ML model, a cleaner codebase, and a smarter, dynamic webpage. The goal is to improve meaningfully without over-complicating the project.

---

## What I Will Do (My Jobs)

### 1. Convert the Raw `.data` Files to Clean `.csv` Files
You dropped two files in the project root that need to be preprocessed before they can be trained on:

#### `processed.cleveland.data` → `heart_disease.csv`
- **Format**: Comma-separated, 14 columns, no headers.
- **Key Issues**:
  - Some cells contain `?` (missing values) — I will replace them with the column median.
  - The target column (last column) has values `0, 1, 2, 3, 4`. I will **binarize** it: `0` stays as healthy, and anything `> 0` becomes `1` (disease present). This makes it a binary classifier, consistent with the other datasets.
- **Columns I will assign**:
  `age, sex, chest_pain_type, resting_bp, cholesterol, fasting_blood_sugar, resting_ecg, max_heart_rate, exercise_angina, st_depression, st_slope, num_vessels, thalassemia, target`

#### `wdbc.data` → `breast_cancer.csv`
- **Format**: Comma-separated, 32 columns, no headers.
- **Key Issues**:
  - Column 1 is a patient ID number — I will **drop** it (irrelevant noise for ML).
  - Column 2 is the diagnosis label as a letter (`M` = Malignant, `B` = Benign) — I will convert it to `1` and `0` respectively.
  - The remaining 30 columns are all numeric cell measurement features (radius, texture, smoothness, etc.).

---

### 2. Reorganize the Project Structure
I will create a clean folder layout:
```
ML_Diabetes_App-main/
│
├── datasets/
│   ├── diabetes.csv               ← (moved here, unchanged)
│   ├── heart_disease.csv          ← (newly created from .data file)
│   └── breast_cancer.csv          ← (newly created from .data file)
│
├── models/                        ← (all .pkl files will go here)
│
├── static/
│   └── index.html                 ← (upgraded UI)
│
├── app.py                         ← (upgraded)
├── train_models.py                ← (upgraded)
└── requirements.txt               ← (updated if needed)
```

---

### 3. Upgrade `train_models.py`
I will rewrite the training script to be **fully dynamic and automatic**. The new logic will:

1. Loop through every `.csv` file inside the `datasets/` folder.
2. For each file:
   - Treat everything to the **left of the last column** as input features (X).
   - Treat the **last column** as the target/label (y).
   - Replace invalid zeros in physiological columns with the column median (same as before).
3. Train **one model per disease** using a **Random Forest Classifier** (replacing Naive Bayes + Perceptron, which performed poorly). Random Forest is significantly more accurate on tabular medical data.
4. Save three files per disease into the `models/` folder:
   - `{disease_name}_model.pkl` — the trained classifier
   - `{disease_name}_scaler.pkl` — the StandardScaler for that disease
   - `{disease_name}_features.pkl` — **the list of input column names** (this is the key to a dynamic webpage)

---

### 4. Upgrade `app.py`
I will update the Flask backend to:
- Load all models dynamically from the `models/` folder into a dictionary at startup.
- Add a `/diseases` endpoint that tells the frontend what diseases and what input fields are available.
- Update the `/predict` endpoint to accept a `disease` parameter and route to the correct model.
- **Optionally**, add a `/predict_all` endpoint that takes all possible inputs and runs every applicable disease model, returning a results summary (for the "comprehensive health check" mode).

---

### 5. Upgrade `static/index.html` (The Webpage)
I will redesign the webpage UI with two modes:

**Mode 1 — "Single Disease" Picker**
- A dropdown at the top lets the user select the disease (Diabetes, Heart Disease, Breast Cancer).
- The JavaScript fetches the input fields required from `/diseases` and **dynamically builds the form** based on the selected disease.
- On submit, it calls `/predict` with the selected disease and input values.
- The result displays as a color-coded badge with a **probability percentage** (e.g., `Heart Disease: 78% Risk`).

**Mode 2 — "Full Health Scan"**
- A toggle button switches to a master form showing all possible fields at once.
- The user fills in as many as they know.
- On submit, it calls `/predict_all`, and the page displays a card for each disease that had enough data, with a risk level.

---

## What I Need From You (Your Jobs)

> [!IMPORTANT]
> Before I can start, I need you to do **two small things**:

### ✅ Job 1: Confirm the two `.data` files are in the right place
Make sure `processed.cleveland.data` and `wdbc.data` are sitting in the **project root** (`e:\Code\Projects\ML_Diabetes_App-main\`). (They already appear to be there based on our directory listing.)

### ✅ Job 2: Stop the currently running Flask server
You currently have `python app.py` running in the terminal. **Press `Ctrl+C`** in that terminal to stop it before I start modifying the files.

Once you confirm both of the above, I will start implementing everything in order.

---

## Estimated Scope of Changes

| File | Change Type | Description |
| :--- | :--- | :--- |
| `processed.cleveland.data` | Read-only (input) | Converted to `datasets/heart_disease.csv` |
| `wdbc.data` | Read-only (input) | Converted to `datasets/breast_cancer.csv` |
| `diabetes.csv` | Move | Moved to `datasets/` folder |
| `train_models.py` | Full Rewrite | Dynamic, multi-disease, Random Forest |
| `app.py` | Full Rewrite | Dynamic multi-model loading, new endpoints |
| `static/index.html` | Full Rewrite | Two-mode dynamic UI |
| `requirements.txt` | Minor Update | Verify all packages are listed |
| `models/` | New Folder | All `.pkl` files stored here |
| `datasets/` | New Folder | All `.csv` files stored here |
