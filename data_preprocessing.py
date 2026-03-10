# ============================================================
# PHASE 1: Data Loading & Preprocessing (FIXED for yfinance)
# Bitcoin Price Prediction using LSTM
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
os.makedirs('data', exist_ok=True)
print("✅ data/ folder created!")

# ✅ Add this before plt.savefig()
os.makedirs('data', exist_ok=True)

plt.savefig('btc_closing_price.png')

# ============================================================
# STEP 1: LOAD & FIX THE DATASET (yfinance format)
# ============================================================

df_raw = pd.read_csv("bitcoin_hourly.csv")

# Drop first 2 junk rows (Ticker & Datetime label rows)
df_raw = df_raw.iloc[2:].reset_index(drop=True)

# Rename 'Price' column to 'Date'
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

# Sort by date & drop nulls
df = df.sort_values('Date').reset_index(drop=True)
df.dropna(inplace=True)

print("=" * 55)
print("✅ Dataset Loaded & Cleaned Successfully!")
print("=" * 55)
print(f"📅 Date Range  : {df['Date'].min()} → {df['Date'].max()}")
print(f"📊 Total Rows  : {len(df)}")
print(f"📋 Columns     : {list(df.columns)}")

# ============================================================
# STEP 2: EXPLORE THE DATASET
# ============================================================

print(f"\n🔍 First 5 Rows:\n{df.head()}")
print(f"\n📉 Missing Values:\n{df.isnull().sum()}")
print(f"\n📊 Basic Statistics:\n{df.describe()}")

# ============================================================
# STEP 3: CLOSING PRICE STATS
# ============================================================

print(f"\n💰 Close Price Stats:")
print(f"   Min  : ${df['Close'].min():,.2f}")
print(f"   Max  : ${df['Close'].max():,.2f}")
print(f"   Mean : ${df['Close'].mean():,.2f}")

# ============================================================
# STEP 4: VISUALIZE THE CLOSING PRICE
# ============================================================

plt.figure(figsize=(14, 5))
plt.plot(df['Date'], df['Close'], color='orange', linewidth=1.5)
plt.title('Bitcoin Closing Price - Last 730 Days', fontsize=16)
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('data/btc_closing_price.png')
plt.show()
print("\n✅ Chart saved → data/btc_closing_price.png")

# ============================================================
# STEP 5: NORMALIZE THE DATA (MinMaxScaler)
# ============================================================

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['Close']])

print(f"\n✅ Data Normalized (MinMaxScaler)!")
print(f"   Original Sample : {df['Close'].values[:3]}")
print(f"   Scaled Sample   : {scaled_data[:3].flatten()}")

# ============================================================
# STEP 6: CREATE SEQUENCES (Sliding Window = 60 days)
# ============================================================

TIME_STEP = 60  # Look back 60 days to predict next day

def create_sequences(data, time_step):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i - time_step:i, 0])  # past 60 days
        y.append(data[i, 0])                # next day price
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, TIME_STEP)

print(f"\n✅ Sequences Created (Time Step = {TIME_STEP} days)!")
print(f"   X shape (input) : {X.shape}")
print(f"   y shape (target): {y.shape}")

# ============================================================
# STEP 7: TRAIN / TEST SPLIT (80% / 20%)
# ============================================================

split = int(len(X) * 0.80)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"\n✅ Train/Test Split Done!")
print(f"   Training Samples : {len(X_train)}")
print(f"   Testing Samples  : {len(X_test)}")

# ============================================================
# STEP 8: RESHAPE FOR LSTM (3D Format)
# ============================================================
# LSTM requires: [samples, time_steps, features]

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test  = X_test.reshape(X_test.shape[0],  X_test.shape[1],  1)

print(f"\n✅ Reshaped for LSTM Input!")
print(f"   X_train : {X_train.shape}  →  [samples, 60 days, 1 feature]")
print(f"   X_test  : {X_test.shape}   →  [samples, 60 days, 1 feature]")

# ============================================================
# STEP 9: SAVE ALL PREPROCESSED FILES
# ============================================================

os.makedirs('data', exist_ok=True)

np.save('data/X_train.npy', X_train)
np.save('data/X_test.npy',  X_test)
np.save('data/y_train.npy', y_train)
np.save('data/y_test.npy',  y_test)

# Save scaler (needed in Flask app to inverse transform)
with open('data/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save cleaned dataframe (needed in Flask app for chart)
df.to_csv('data/btc_cleaned.csv', index=False)

print("\n" + "=" * 55)
print("🎉 PHASE 1 COMPLETE!")
print("=" * 55)
print("📁 Saved Files:")
print("   ✅ data/X_train.npy           → Training input")
print("   ✅ data/X_test.npy            → Testing input")
print("   ✅ data/y_train.npy           → Training labels")
print("   ✅ data/y_test.npy            → Testing labels")
print("   ✅ data/scaler.pkl            → MinMaxScaler object")
print("   ✅ data/btc_cleaned.csv       → Cleaned dataset")
print("   ✅ data/btc_closing_price.png → Price chart")
print("\n▶️  Next Step → Run: train_model.py (Phase 2)")