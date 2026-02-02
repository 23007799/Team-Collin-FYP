# preprocess.py
from tsfm_public import TimeSeriesPreprocessor

def build_column_specifiers(data, target_col="Price"):
    control_columns = [c for c in data.columns if c not in ["Date", target_col]]

    return {
        "timestamp_column": "Date",
        "id_columns": [],
        "target_columns": [target_col],
        "control_columns": control_columns,
        "static_categorical_columns": [],
        "categorical_columns": [],
    }


def train_preprocessor(train_data, column_specifiers, context, forecast):
    tsp = TimeSeriesPreprocessor(
        **column_specifiers,
        context_length=context,
        prediction_length=forecast,
        scaling=True,
        encode_categorical=True,
        scaler_type="standard",
    )
    tsp.train(train_data)
    return tsp
