# run_forecast.py
from configuration import *
from data_prep import load_data, parse_human_numbers, split_by_year
from preprocessor import build_column_specifiers, train_preprocessor
from model import load_model
from forecast import run_forecast, plot_forecast

DATA_PATH = "data/Multivariate DBS data 2015-2025.xlsx"

data = load_data(DATA_PATH)

column_specifiers = build_column_specifiers(data)
cols_to_clean = column_specifiers["control_columns"] + column_specifiers["target_columns"]
data = parse_human_numbers(data, cols_to_clean)

train, valid, test = split_by_year(data, TARGET_YEAR)

tsp = train_preprocessor(train, column_specifiers, CONTEXT_LENGTH, FORECAST_LENGTH)

model = load_model(tsp, TTM_MODEL_PATH, REVISION, CONTEXT_LENGTH, FORECAST_LENGTH)

forecast_df = run_forecast(model, tsp, test)

row = forecast_df.iloc[0]
plot_forecast(
    history=row["Price"],
    prediction=row["Price_prediction"],
    title="DBS Stock Forecast"
)
