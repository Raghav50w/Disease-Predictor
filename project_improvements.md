# Disease Predictor - Enhancement Roadmap

## 1. Expanding to Multi-Disease Prediction
To make the application capable of predicting multiple diseases (e.g., Heart Disease, Breast Cancer, Kidney Disease), we need a structured approach that scales easily.

### Where to get datasets
* **Kaggle (Highly Recommended)**: The best source for beginner-friendly, pre-cleaned medical datasets. Search for:
  * *Heart Disease*: "Heart Disease UCI" dataset.
  * *Breast Cancer*: "Breast Cancer Wisconsin (Diagnostic)" dataset.
  * *Chronic Kidney Disease*: "Chronic Kidney Disease" dataset.
* **UCI Machine Learning Repository**: The original academic source for many classic medical datasets.

### How to format new datasets
To ensure compatibility with our existing system, format new datasets to mimic `diabetes.csv`:
1. **File Type**: Must be a `.csv` file.
2. **Target Column**: Ensure the prediction label is in a single column (e.g., `Outcome`, `target`, or `class`). Binary values are required (`0` for Negative/Healthy, `1` for Positive/Disease).
3. **Numeric Features**: All input columns must be numeric. If a dataset has categorical data (like "Male"/"Female"), convert them to numbers (e.g., `0`/`1`) before saving the CSV.
4. **Handling Missing Data**: Clean the dataset so there are no empty cells or `NaN` values. You can leave them as `0` if you configure `train_models.py` to replace zeros with the column median (as it currently does for glucose and insulin).

---

## 2. Webpage / UI Enhancements
We want to keep the UI simple, clean, and not overly complicated, while making it much more powerful.

### Two Input Modes
* **Mode 1: Specific Disease Assessment**
  Add a dropdown at the top of the page to select the target disease (Diabetes, Heart Disease, etc.). Using simple JavaScript, the form dynamically hides/shows the input fields required for that specific disease.
  
* **Mode 2: Comprehensive Health Check (All-in-One)**
  The user enters all their available medical metrics (Age, BMI, Glucose, Blood Pressure, Cholesterol, etc.) into one master form. The backend API (`app.py`) checks which inputs are provided and automatically runs inference for **all applicable diseases**. The UI then displays a summary dashboard (e.g., "Diabetes: Low Risk", "Heart Disease: Moderate Risk").

### Visual & UX Improvements
* **Probability Scores**: Instead of just saying "Diabetic" or "Non-Diabetic", have the backend return the probability (e.g., "75% Risk"). Display this on the frontend using a colored progress bar or gauge chart.
* **Input Grid**: As you add more input fields, update the CSS to display the inputs in a 2-column grid so the page doesn't require endless scrolling.

---

## 3. Modeling & Backend Upgrades
We can improve the backend without introducing complex frameworks.

### Better Machine Learning Models
* **Random Forest Classifier**: Add this to `train_models.py`. Tree-based models generally perform better on tabular medical data than Naive Bayes or simple Perceptrons, and they require almost no tuning.
* **Logistic Regression**: A simple, interpretable model that is excellent for providing accurate percentage-based risk probabilities.

### Codebase Organization
* **Folder Structure**: Move all `.csv` files into a `datasets/` folder, and all `.pkl` files into a `models/` folder to keep the root directory clean.
* **Dynamic Model Loading**: Update `app.py` to load models into a dictionary, making it easy to route predictions based on the user's request:
  ```python
  models = {
      "diabetes": pickle.load(open("models/diabetes_rf.pkl", "rb")),
      "heart": pickle.load(open("models/heart_rf.pkl", "rb"))
  }
  ```
* **Iterative Training**: Refactor `train_models.py` so it automatically loops through every CSV in the `datasets/` folder and trains a model for each one, meaning you don't have to write new code for every new disease you add.
