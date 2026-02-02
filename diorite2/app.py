from flask import Flask, jsonify
from flask_cors import CORS  # Install with pip install flask-cors
from configuration import *
from data_prep import load_data, parse_human_numbers, split_by_year
from preprocessor import build_column_specifiers, train_preprocessor
from model import load_model
from forecast import run_forecast
import pandas as pd

app = Flask(__name__)
CORS(app)  # Allow frontend requests

# Load model and preprocessor on startup (expensive, do once)
data_path = "data/Multivariate DBS data 2015-2025.xlsx"  # Fixed path from run_forecast.py snippet
data = load_data(data_path)
column_specifiers = build_column_specifiers(data)
cols_to_clean = column_specifiers['control_columns']
data = parse_human_numbers(data, cols_to_clean)
train, _, test = split_by_year(data, TARGET_YEAR)
tsp = train_preprocessor(train, column_specifiers, CONTEXT_LENGTH, FORECAST_LENGTH)
model = load_model(tsp, TTM_MODEL_PATH, REVISION, CONTEXT_LENGTH, FORECAST_LENGTH)

@app.route('/forecast', methods=['GET'])
def get_forecast():
    forecast_df = run_forecast(model, tsp, test)
    row = forecast_df.iloc[0]
    history = row['Price'].tolist()  # Past prices
    prediction = row['Price_prediction'].tolist()  # Forecast
    return jsonify({'history': history, 'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
