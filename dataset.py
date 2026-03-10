import pandas as pd

df = pd.read_csv("bitcoin_hourly.csv")  # your filename here

print("✅ Columns in your dataset:")
print(df.columns.tolist())

print("\n🔍 First 3 rows:")
print(df.head(3))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pickle

# ============================================================
# STEP 1: LOAD & FIX THE DATASET (yfinance format)
# ============================================================

df_raw = pd.read_csv("bitcoin_hourly.csv")

# Drop first 2 junk rows (Ticker & Datetime label rows)
df_raw = df_raw.iloc[2:].reset_index(drop=True)

# Rename 'Price' column to 'Date' (that's actually the date column)
df_raw.rename(columns={'Price': 'Date'}, inplace=True)

# Keep only needed columns
df = df_raw[['Date', 'Close', 'High', 'Low', 'Open', 'Volume']].copy()

# Fix data types
df['Date']   = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)
df['Close']  = pd.to_numeric(df['Close'],  errors='coerce')
df['High']   = pd.to_numeric(df['High'],   errors='coerce')
df['Low']    = pd.to_numeric(df['Low'],    errors='coerce')
df['Open']   = pd.to_numeric(df['Open'],   errors='coerce')
df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')

# Sort & clean
df = df.sort_values('Date').reset_index(drop=True)
df.dropna(inplace=True)

print("✅ Dataset Cleaned Successfully!")
print(f"📅 Date Range : {df['Date'].min()} → {df['Date'].max()}")
print(f"📊 Total Rows : {len(df)}")
print(f"\n🔍 Sample:\n{df.head(3)}")