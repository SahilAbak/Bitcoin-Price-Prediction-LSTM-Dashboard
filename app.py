# ============================================================
# PHASE 3: Flask Backend — Bitcoin Prediction Dashboard
# ============================================================

from flask import Flask, render_template, jsonify
import numpy as np
import pandas as pd
import pickle
import json
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ============================================================
# LOAD ALL SAVED FILES AT STARTUP
# ============================================================

print("🔄 Loading model and data...")

model = load_model('model/lstm_model.h5')
scaler = pickle.load(open('data/scaler.pkl', 'rb'))
df     = pd.read_csv('data/btc_cleaned.csv')

y_pred   = np.load('data/y_pred.npy').flatten()
y_actual = np.load('data/y_actual.npy').flatten()
metrics  = json.load(open('data/metrics.json'))

df['Date'] = pd.to_datetime(df['Date'])

print("✅ All files loaded successfully!")

# ============================================================
# HELPER: PREDICT NEXT N DAYS
# ============================================================

def predict_future(n_days=30):
    scaled_data = scaler.transform(df[['Close']])
    last_60     = scaled_data[-60:]
    predictions = []

    input_seq = last_60.copy()

    for _ in range(n_days):
        X = input_seq[-60:].reshape(1, 60, 1)
        pred = model.predict(X, verbose=0)[0][0]
        predictions.append(pred)
        input_seq = np.append(input_seq, [[pred]], axis=0)

    future_prices = scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1)
    ).flatten()

    last_date    = df['Date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_days)

    return future_dates.strftime('%Y-%m-%d').tolist(), future_prices.tolist()

# ============================================================
# ROUTES
# ============================================================

@app.route('/')
def index():
    current_price = round(df['Close'].iloc[-1], 2)
    prev_price    = round(df['Close'].iloc[-2], 2)
    change        = round(current_price - prev_price, 2)
    change_pct    = round((change / prev_price) * 100, 2)

    return render_template('index.html',
        current_price = f"{current_price:,.2f}",
        change        = f"{change:+,.2f}",
        change_pct    = f"{change_pct:+.2f}",
        rmse          = f"{metrics['rmse']:,.2f}",
        mae           = f"{metrics['mae']:,.2f}",
        mape          = f"{metrics['mape']:.2f}",
        accuracy      = f"{metrics['accuracy']:.2f}"
    )

# ── API: Actual vs Predicted Chart Data ──────────────────────
@app.route('/api/actual_vs_predicted')
def actual_vs_predicted():
    total     = len(y_actual)
    dates     = df['Date'].iloc[-total:].dt.strftime('%Y-%m-%d').tolist()

    return jsonify({
        'dates'    : dates,
        'actual'   : [round(float(v), 2) for v in y_actual],
        'predicted': [round(float(v), 2) for v in y_pred]
    })

# ── API: Full Historical Price ────────────────────────────────
@app.route('/api/historical')
def historical():
    return jsonify({
        'dates' : df['Date'].dt.strftime('%Y-%m-%d').tolist(),
        'close' : [round(float(v), 2) for v in df['Close'].tolist()],
        'high'  : [round(float(v), 2) for v in df['High'].tolist()],
        'low'   : [round(float(v), 2) for v in df['Low'].tolist()],
        'volume': [round(float(v), 2) for v in df['Volume'].tolist()]
    })

# ── API: Future Forecast ──────────────────────────────────────
@app.route('/api/forecast')
def forecast():
    dates, prices = predict_future(n_days=30)
    return jsonify({
        'dates' : dates,
        'prices': [round(float(p), 2) for p in prices]
    })

# ── API: Metrics ──────────────────────────────────────────────
@app.route('/api/metrics')
def get_metrics():
    return jsonify(metrics)

# ============================================================
# RUN APP
# ============================================================

if __name__ == '__main__':
    app.run(debug=True, port=5000)