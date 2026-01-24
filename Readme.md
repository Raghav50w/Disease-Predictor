# ML_Diabetes_App

Simple diabetes classifier app.

## Files
- `diabetes.csv` : dataset (place in project root)
- `train_models.py` : trains models and saves naive_bayes_model.pkl, mlp_model.pkl, scaler.pkl
- `app.py` : Flask app that serves UI and /predict endpoint
- `static/index.html` : simple frontend

## Usage
1. Activate venv:
   - PowerShell: `.\venv\Scripts\Activate.ps1`
   - cmd: `venv\Scripts\activate`
2. Install requirements: `python -m pip install -r requirements.txt`
3. Train: `python train_models.py` 
4. Run server: `python app.py` 
5. Open browser: http://127.0.0.1:5000

## Deliverables
1. app.py: Flask application to demonstrate the use of both models in real-time predictions
2. train_modules.py: Python code for generating both the models, upon running comparision of their performance
3. index.html: Frontend code for web app