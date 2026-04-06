import pandas as pd
import numpy as np
from datetime import timedelta
import holidays

US_HOLIDAYS = holidays.US(years=[2024, 2025])

def add_calendar_features(df, date_col='Date'):
    df = df.copy()
    
    dates = pd.to_datetime(df[date_col])
    df["DayOfWeek"] = dates.dt.dayofweek
    df["Month"] = dates.dt.month
    df["DayOfMonth"] = dates.dt.day
    
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
    df["IsHoliday"] = dates.dt.date.map(lambda d: int(d in US_HOLIDAYS))
    
    day_before = (dates - timedelta(days=1)).dt.date
    day_after = (dates + timedelta(days=1)).dt.date
    
    df["NearHoliday"] = day_before.map(lambda d: int(d in US_HOLIDAYS)) \
        | day_after.map(lambda d: int(d in US_HOLIDAYS))
    
    return df

def encode_dow(dow):
    vec = np.zeros(7, dtype=np.float32)
    vec[dow] = 1.0
    
    return vec

def august_2025_dates():
    return pd.date_range(start="2025-08-01", end="2025-08-31", freq="D")

def interval_labels():
    labels = []
    for hour in range(24):
        labels.append(f"{hour}:00")
        labels.append(f"{hour}:30")
        
    return labels