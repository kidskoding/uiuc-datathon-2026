import pandas as pd
import numpy as np

EXCEL_PATH = "data/raw/datathon-data.xlsx"
PORTFOLIOS = ["A", "B", "C", "D"]

def load_daily(portfolio):
    df = pd.read_excel(EXCEL_PATH, f"{portfolio} - Daily")
    df.columns = df.columns.str.strip()
    
    df = df.rename(columns={
        "Date": "Date",
        "Call Volume": "CV",
        "CCT": "CCT",
        "Service Level": "SL",
        "Abandon Rate": "ABD"
    })
    
    df['Date'] = pd.to_datetime(df['Date'].str.split(' ').str[0], format='%m/%d/%y', errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)
    
    df['Portfolio'] = portfolio
    return df

def load_interval(portfolio):
    df = pd.read_excel(EXCEL_PATH, f"{portfolio} - Interval")
    df.columns = df.columns.str.strip()
    
    df.rename(columns={
        "Service Level": "SL",
        "Call Volume": "CV",
        "Abandoned Calls": "AbandonedCalls",
        "Abandoned Rate": "ABD",
    }, inplace=True)
    
    df['Month'] = df['Month'].astype(str).str.strip()
    df['Day'] = df['Day'].astype(int)
    df['Interval'] = df['Interval'].astype(str)
    
    df['Date'] = pd.to_datetime(
        df['Month'] + " " + df['Day'].astype(str) + " 2025",
        format="%B %d %Y",
        errors='coerce'
    )
    
    df['Portfolio'] = portfolio
    return df

def load_staffing():
    df = pd.read_excel(EXCEL_PATH, "Daily Staffing")
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"Unnamed: 0": "Date"})
    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.sort_values('Date').reset_index(drop=True)
    
    return df

def clean(df):
    df = df.copy()
    
    df['ABD'] = pd.to_numeric(df['ABD'], errors='coerce')
    df = df[df['ABD'] < 1.0]
    
    df['CCT'] = pd.to_numeric(df['CCT'], errors='coerce')
    df['CCT'] = df['CCT'].clip(upper=1000)
    
    df['CCT'] = df['CCT'].replace(0, np.nan)
    df['CCT'] = df['CCT'].interpolate(method='linear').bfill()
    
    df['CV'] = df['CV'].clip(0)
    df['CCT'] = df['CCT'].clip(0)
    df['ABD'] = df['ABD'].clip(lower=0, upper=1)
    
    return df
