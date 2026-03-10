# ============================================================
# PHASE 2: Build & Train LSTM Model
# Bitcoin Price Prediction using LSTM
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import pickle

# TensorFlow / Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

os.makedirs('model', exist_ok=True)

# ============================================================
# STEP 1: LOAD PREPROCESSED DATA
# ============================================================

X_train = np.load('data/X_train.npy')
X_test  = np.load('data/X_test.npy')
y_train = np.load('data/y_train.npy')
y_test  = np.load('data/y_test.npy')

with open('data/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

print("=" * 55)
print("✅ Data Loaded Successfully!")
print("=" * 55)
print(f"   X_train : {X_train.shape}")
print(f"   X_test  : {X_test.shape}")
print(f"   y_train : {y_train.shape}")
print(f"   y_test  : {y_test.shape}")

# ============================================================
# STEP 2: BUILD THE LSTM MODEL
# ============================================================

model = Sequential([

    # First LSTM Layer
    LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),

    # Second LSTM Layer
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),

    # Third LSTM Layer
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),

    # Output Layer
    Dense(units=1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

print("\n✅ Model Built Successfully!")
print("\n📋 Model Summary:")
model.summary()

# ============================================================
# STEP 3: CALLBACKS (EarlyStopping + ModelCheckpoint)
# ============================================================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    filepath='model/best_model.keras',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# ============================================================
# STEP 4: TRAIN THE MODEL
# ============================================================

print("\n🚀 Training Started...\n")

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

print("\n✅ Training Complete!")

# ============================================================
# STEP 5: PLOT TRAINING LOSS
# ============================================================

plt.figure(figsize=(12, 5))
plt.plot(history.history['loss'],     label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Val Loss',   color='red')
plt.title('Model Training Loss', fontsize=16)
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('data/training_loss.png')
plt.show()
print("✅ Loss chart saved → data/training_loss.png")

# ============================================================
# STEP 6: MAKE PREDICTIONS
# ============================================================

y_pred_scaled = model.predict(X_test)

# Inverse transform → back to original price scale
y_pred = scaler.inverse_transform(y_pred_scaled)
y_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

print(f"\n✅ Predictions Generated!")
print(f"   Sample Actual    : {y_actual[:3].flatten()}")
print(f"   Sample Predicted : {y_pred[:3].flatten()}")

# ============================================================
# STEP 7: EVALUATE THE MODEL
# ============================================================

rmse = math.sqrt(mean_squared_error(y_actual, y_pred))
mae  = mean_absolute_error(y_actual, y_pred)
mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100

print("\n" + "=" * 55)
print("📊 MODEL EVALUATION METRICS")
print("=" * 55)
print(f"   RMSE : ${rmse:,.2f}   (Root Mean Square Error)")
print(f"   MAE  : ${mae:,.2f}   (Mean Absolute Error)")
print(f"   MAPE : {mape:.2f}%     (Mean Absolute % Error)")
print(f"   Accuracy : ~{100 - mape:.2f}%")

# ============================================================
# STEP 8: PLOT ACTUAL vs PREDICTED
# ============================================================

plt.figure(figsize=(14, 6))
plt.plot(y_actual, label='Actual Price',    color='blue',   linewidth=1.5)
plt.plot(y_pred,   label='Predicted Price', color='orange', linewidth=1.5, linestyle='--')
plt.title('Bitcoin Price: Actual vs Predicted', fontsize=16)
plt.xlabel('Days')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('data/actual_vs_predicted.png')
plt.show()
print("✅ Chart saved → data/actual_vs_predicted.png")

# ============================================================
# STEP 9: SAVE PREDICTIONS & METRICS
# ============================================================

np.save('data/y_pred.npy',   y_pred)
np.save('data/y_actual.npy', y_actual)

# Save metrics for Flask dashboard
import json
metrics = {
    "rmse"     : round(rmse, 2),
    "mae"      : round(mae, 2),
    "mape"     : round(mape, 2),
    "accuracy" : round(100 - mape, 2)
}
with open('data/metrics.json', 'w') as f:
    json.dump(metrics, f)

# Save final model
model.save('model/lstm_model.keras')

print("\n" + "=" * 55)
print("🎉 PHASE 2 COMPLETE!")
print("=" * 55)
print("📁 Saved Files:")
print("   ✅ model/lstm_model.keras       → Final trained model")
print("   ✅ model/best_model.keras       → Best checkpoint model")
print("   ✅ data/y_pred.npy              → Predictions array")
print("   ✅ data/y_actual.npy            → Actual values array")
print("   ✅ data/metrics.json            → Evaluation metrics")
print("   ✅ data/training_loss.png       → Loss curve chart")
print("   ✅ data/actual_vs_predicted.png → Comparison chart")
print("\n▶️  Next Step → Run: app.py (Phase 3 - Flask Dashboard)")