# config.py
import torch

FORECAST_LENGTH = 28
CONTEXT_LENGTH = 90

TTM_MODEL_PATH = "ibm-granite/granite-timeseries-ttm-r2"
REVISION = "90-30-ft-l1-r2.1"

DATA_PATH = "data/Events_encoded_DBS.xlsx"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_YEAR = 2025
