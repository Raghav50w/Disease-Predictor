# HealthScan AI - Comprehensive Multi-Disease Risk Predictor

**HealthScan AI** is an advanced, machine learning-powered web application designed to predict the risk of three critical diseases: **Diabetes**, **Heart Disease**, and **Breast Cancer**. 

Unlike standard diagnostic tools that rely on a single algorithm, HealthScan AI provides a robust **Multi-Model architecture**. It dynamically trains and evaluates four distinct machine learning algorithms for each disease, allowing users and medical professionals to compare predictions across different computational approaches.

This project was built with a focus on clinical usability, utilizing feature importance techniques to reduce input fatigue while maintaining high predictive accuracy.

---

## 🌟 Key Features & Architecture

### 1. Multi-Disease Support
The platform seamlessly handles three distinct medical datasets:
- **Breast Cancer (WDBC)**: Originally 30 features, reduced to the top 6 most critical cellular metrics.
- **Heart Disease (Cleveland)**: Originally 13 features, reduced to the top 6 most critical cardiovascular metrics.
- **Diabetes (Pima Indians format)**: Utilizes 4 core metabolic features.

### 2. Multi-Model Support
For every disease, the system automatically trains, serializes, and serves four different machine learning models:
1. **Random Forest Classifier**: An ensemble learning method using 200 decision trees. Highly robust against overfitting and excellent for tabular clinical data. (Default model).
2. **Gaussian Naive Bayes**: A probabilistic classifier based on applying Bayes' theorem with strong independence assumptions between the features.
3. **Multi-Layer Perceptron (MLP)**: A feedforward artificial neural network (100x50 hidden layers) capable of learning non-linear clinical relationships.
4. **Custom Perceptron**: A linear binary classifier suitable for simple linear separability.

### 3. Dynamic REST API Backend
Built on **Flask**, the backend is entirely dynamic. It scans the `models/` directory at startup, reads the serialized `.pkl` files, and automatically maps available diseases, their required input features, and the available ML models.

### 4. Adaptive Frontend UI
The frontend uses pure HTML/CSS/JS to create a clean, clinical interface. When a user selects a disease from the dropdown, the UI makes a `GET /diseases` request to the backend and dynamically generates the exact input fields required for that disease, complete with realistic clinical placeholder values.

---

## 📂 Project Directory Structure

```text
ML_Diabetes_App-main/
│
├── datasets/                   # Cleaned, ready-to-use CSV datasets
│   ├── raw/                    # Original raw data files (.data files)
│   │   ├── processed.cleveland.data
│   │   └── wdbc.data
│   ├── breast_cancer.csv       # Trimmed to top 6 features
│   ├── diabetes.csv            # Original 4 features
│   └── heart_disease.csv       # Trimmed to top 6 features
│
├── models/                     # Auto-generated serialized files by train_models.py
│   ├── {disease}_features.pkl  # Lists the specific input features required
│   ├── {disease}_scaler.pkl    # StandardScaler instances for input normalization
│   └── {disease}_{model}.pkl   # The actual trained scikit-learn models
│
├── static/
│   └── index.html              # Clean, dynamic frontend User Interface
│
├── app.py                      # Flask API backend and routing
├── feature_analysis.py         # Script to mathematically prove feature selection
├── preprocess_data.py          # Data cleaning & feature reduction script
├── train_models.py             # Script to train 4 algorithms per disease
└── requirements.txt            # Python environment dependencies
```

---

## 🚀 Installation & Setup Guide

### 1. Prerequisites
Ensure you have Python 3.8 or higher installed on your system. You will also need `git` to clone the repository.

### 2. Clone the Repository
```bash
git clone https://github.com/yourusername/ML_Diabetes_App.git
cd ML_Diabetes_App
```

### 3. Create a Virtual Environment (Highly Recommended)
Creating a virtual environment isolates the project dependencies from your global Python installation.

**On Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
Install all required packages (Flask, scikit-learn, pandas, numpy) using `pip`:
```bash
pip install -r requirements.txt
```

---

## 🛠️ Usage & Execution

### Step 1: Feature Selection Analysis (Optional)
To see exactly *why* we reduced the datasets from 30+ features down to just 6, you can run the Feature Analysis script:
```bash
python feature_analysis.py
```
**What this does:** It trains a Random Forest model on the raw data and extracts the mathematical `feature_importances_` metric, ranking every single physiological input by its predictive power. We use the top 6 from this list for our actual application.

### Step 2: Data Preprocessing
*(Note: This step is already completed in the repository, but you can run it again if you modify the raw data.)*

The raw datasets contain missing values and excessive features. Run the preprocessing script to clean the data and extract only the most statistically significant features identified in Step 1:
```bash
python preprocess_data.py
```
**What this does:**
- Imputes missing physiological values with column medians.
- Binarizes target variables (e.g., converting classes 1-4 into simply "Risk Present").
- Uses Random Forest Feature Importance to drop the dataset down to just 6 inputs per disease.

### Step 2: Model Training
*(Note: Pre-trained models are already included in the `models/` directory, but you should run this to verify your local environment.)*

Train all 12 machine learning models (3 diseases × 4 algorithms):
```bash
python train_models.py
```
**What this does:**
- Loads the cleaned CSVs.
- Normalizes the data using `StandardScaler`.
- Performs 5-Fold Cross Validation.
- Saves the trained models, scalers, and feature lists to the `models/` folder as `.pkl` files.

### Step 3: Launch the Web Server
Start the Flask application backend:
```bash
python app.py
```
The terminal will display the models it successfully loaded. The server will start running on `http://127.0.0.1:5000`.

### Step 4: Access the Application
1. Open your web browser and navigate to `http://127.0.0.1:5000`.
2. Select your target disease from the first dropdown menu.
3. Select the Machine Learning Model you wish to use from the second dropdown menu.
4. Input the patient's physiological metrics into the dynamically generated form.
5. Click **Analyze Risk**. The system will return a risk assessment (e.g., "Risk Detected") along with the model's statistical confidence percentage.

---

## 📊 Model Performance Metrics

The models were evaluated using 5-fold cross-validation on the trimmed datasets. Below are the approximate accuracy scores:

| Disease | Random Forest | Naive Bayes | MLP (Neural Net) | Perceptron |
|---------|---------------|-------------|------------------|------------|
| **Breast Cancer** | **~94.7%** | ~93.9% | ~95.0% | ~92.5% |
| **Heart Disease** | **~78.8%** | ~80.1% | ~81.0% | ~68.4% |
| **Diabetes** | **~75.1%** | ~76.2% | ~77.5% | ~65.1% |

**Clinical Note**: While Neural Networks (MLP) sometimes edge out Random Forest in raw accuracy, Random Forest is set as the default model due to its superior robustness against overfitting on small clinical datasets and its reliable probability calibration.

---

## 🧬 Understanding the Features Used

To reduce the burden on healthcare workers entering data, this platform only requires the most critical features for prediction:

### Breast Cancer (WDBC)
- **Cell Perimeter (worst)**: The largest perimeter among the observed cells.
- **Cell Area (worst)**: The largest area among the observed cells.
- **Concave Points (worst)**: The maximum number of concave portions of the contour.
- **Concave Points (mean)**: The average number of concave portions.
- **Cell Radius (worst)**: The largest radius among the observed cells.
- **Cell Area (mean)**: The average area of the cells.

### Heart Disease (Cleveland)
- **Chest Pain Type**: Ranges from 1 (typical angina) to 4 (asymptomatic).
- **Thalassemia**: Blood disorder metric (3 = normal; 6 = fixed defect; 7 = reversible defect).
- **Major Vessels Colored**: Number of major vessels (0-3) colored by fluoroscopy.
- **Max Heart Rate Achieved**: The patient's maximum heart rate during stress testing.
- **ST Depression**: ST depression induced by exercise relative to rest.
- **Patient Age**: The patient's age in years.

### Diabetes
- **Glucose Level**: Plasma glucose concentration.
- **Insulin Level**: 2-Hour serum insulin (mu U/ml).
- **BMI**: Body mass index (weight in kg/(height in m)^2).
- **Patient Age**: The patient's age in years.
