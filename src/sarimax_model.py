import numpy as np
import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX

def fit_sarimax(series, exog):
    model = SARIMAX(
        series,
        exog=exog,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False)
    
    return model.fit(disp=False, maxiter=200)

def forecast_august(result, exog_august):
    forecast = result.forecast(steps=31, exog=exog_august)
    return np.clip(np.array(forecast), 0, None)
