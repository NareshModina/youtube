import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Data preparation
def fetch_stock_data(ticker="AAPL", start="2019-01-01", end="2024-01-01"):
    df = yf.download(ticker, start=start, end=end)
    return df

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Get data
df = fetch_stock_data()
data = df['Close'].values.reshape(-1, 1)

# Scale data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Create sequences
seq_length = 60
X, y = create_sequences(data_scaled, seq_length)

# Split data
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build LSTM model
model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(seq_length, 1)),
    Dropout(0.2),
    LSTM(50, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
model.summary()

# Train model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform predictions
train_predict = scaler.inverse_transform(train_predict)
y_train_inv = scaler.inverse_transform(y_train)
test_predict = scaler.inverse_transform(test_predict)
y_test_inv = scaler.inverse_transform(y_test)

# Visualize results
plt.style.use('dark_background')
plt.figure(figsize=(15, 8))

# Plot training data
train_dates = df.index[seq_length:train_size+seq_length]
plt.plot(train_dates, y_train_inv, label='Actual Price (Train)', color='cyan', alpha=0.5)
plt.plot(train_dates, train_predict, label='Predicted Price (Train)', color='blue', linestyle='--')

# Plot testing data
test_dates = df.index[train_size+seq_length:]
plt.plot(test_dates, y_test_inv, label='Actual Price (Test)', color='yellow', alpha=0.5)
plt.plot(test_dates, test_predict, label='Predicted Price (Test)', color='red', linestyle='--')

# Calculate metrics
train_rmse = np.sqrt(np.mean((y_train_inv - train_predict) ** 2))
test_rmse = np.sqrt(np.mean((y_test_inv - test_predict) ** 2))

plt.title(f'LSTM Stock Price Prediction\nTrain RMSE: ${train_rmse:.2f} | Test RMSE: ${test_rmse:.2f}')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.grid(True, alpha=0.2)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.2)
plt.show()