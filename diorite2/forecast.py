# forecast.py
import numpy as np
import matplotlib.pyplot as plt
from tsfm_public import TimeSeriesForecastingPipeline

def run_forecast(model, tsp, data):
    pipeline = TimeSeriesForecastingPipeline(
        model,
        feature_extractor=tsp,
        batch_size=1,
    )
    return pipeline(data)


def plot_forecast(history, prediction, title):
    plt.figure(figsize=(10,4))
    plt.plot(history, label="History")
    plt.plot(range(len(history), len(history)+len(prediction)), prediction, label="Forecast")
    plt.legend()
    plt.title(title)
    plt.grid(True)
    plt.show()
