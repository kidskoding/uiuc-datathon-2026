import numpy as np
import pandas as pd

from src.features import august_2025_dates, interval_labels
from src.data import PORTFOLIOS

def blend_weights(nn_weights, empirical_weights, alpha=0.6):
    blended = alpha * nn_weights + (1 - alpha) * empirical_weights
    return blended / blended.sum()

def apply_bias(forecast, bias=1.07):
    return forecast * bias

def clip_forecast(cv, cct, abd):
    cv = np.clip(cv, 0, None)
    cct = np.clip(cct, 0, None)
    abd = np.clip(abd, 0, 1)
    return cv, cct, abd

def assemble_csv(forecasts, template_path="template_forecast_v00.csv"):
    slots = interval_labels()
    aug_dates = august_2025_dates()
    
    rows = []
    for day_idx, date in enumerate(aug_dates):
        day_num = day_idx + 1
        for slot_idx in range(48):
            row = {
                "Month": "August",
                "Day": date.day,
                "Interval": slots[slot_idx],
            }
            
            for p in PORTFOLIOS:
                cv = forecasts[p]["cv"][day_idx, slot_idx]
                abd = forecasts[p]["abd"][day_idx, slot_idx]
                cct = forecasts[p]["cct"][day_idx, slot_idx]
                
                row[f"Calls_Offered_{p}"] = cv
                row[f"Abandoned_Rate_{p}"] = abd
                row[f"CCT_{p}"] = cct
                row[f"Abandoned_Calls_{p}"] = cv * abd
            
            rows.append(row)
            
    df = pd.DataFrame(rows)
    template = pd.read_csv(template_path, nrows=0)
    df = df[template.columns.tolist()]
                 
    return df
