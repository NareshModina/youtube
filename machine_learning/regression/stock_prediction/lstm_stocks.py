import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Fetch stock data
ticker = "AAPL"
stock_data = yf.download(ticker, start="2019-01-01", end="2024-01-01")

# Create features from technical indicators
def add_technical_indicators(df):
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    return df

# Prepare dataset
df = add_technical_indicators(stock_data)
df = df.dropna()

# Features and target
features = ['Open', 'High', 'Low', 'Volume', 'MA20', 'MA50', 'RSI', 'Volume_MA']
target = 'Close'

X = df[features]
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape data for LSTM [samples, timesteps, features]
timesteps = 10
def create_sequences(data, target, timesteps):
    X_seq, y_seq = [], []
    for i in range(len(data) - timesteps):
        X_seq.append(data[i:i + timesteps])
        y_seq.append(target.iloc[i + timesteps])
    return np.array(X_seq), np.array(y_seq)

X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, timesteps)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, timesteps)

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(timesteps, X_train_seq.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=25))
model.add(Dense(units=1))

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
history = model.fit(X_train_seq, y_train_seq, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

# Make predictions
train_pred = model.predict(X_train_seq).flatten()  # Flatten directly here
test_pred = model.predict(X_test_seq).flatten()    # Flatten directly here

# Adjust predictions to match original y_train and y_test lengths for plotting
train_pred_full = np.full(len(y_train), np.nan)  # 1D array matching y_train length
test_pred_full = np.full(len(y_test), np.nan)    # 1D array matching y_test length
train_pred_full[timesteps:] = train_pred         # Assign flattened predictions
test_pred_full[timesteps:] = test_pred           # Assign flattened predictions

# Calculate metrics
train_rmse = np.sqrt(mean_squared_error(y_train_seq, train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test_seq, test_pred))
train_r2 = r2_score(y_train_seq, train_pred)
test_r2 = r2_score(y_test_seq, test_pred)

# Visualize results
plt.style.use('dark_background')

# First Plot: Actual vs Predicted Prices
plt.figure(figsize=(15, 8))
plt.plot(y_train.index, y_train, label='Actual Price (Train)', color='cyan', alpha=0.5)
plt.plot(y_test.index, y_test, label='Actual Price (Test)', color='yellow', alpha=0.5)
plt.plot(y_train.index, train_pred_full, label='Predicted Price (Train)', color='blue', linestyle='--')
plt.plot(y_test.index, test_pred_full, label='Predicted Price (Test)', color='red', linestyle='--')

plt.title(f'{ticker} Stock Price Prediction with LSTM\nTrain R²: {train_r2:.4f} | Test R²: {test_r2:.4f}\nTrain RMSE: ${train_rmse:.2f} | Test RMSE: ${test_rmse:.2f}')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.grid(True, alpha=0.2)
plt.legend(loc='upper left')
plt.xticks(rotation=45)

# Add train/test split line
split_date = y_test.index[0]
plt.axvline(x=split_date, color='white', linestyle=':', alpha=0.5)

plt.tight_layout()

# Second Plot: Residuals (Actual - Predicted)
plt.figure(figsize=(15, 6))
train_residuals = y_train_seq - train_pred
test_residuals = y_test_seq - test_pred

plt.plot(y_train.index[timesteps:], train_residuals, label='Train Residuals', color='blue', alpha=0.7)
plt.plot(y_test.index[timesteps:], test_residuals, label='Test Residuals', color='red', alpha=0.7)
plt.axhline(y=0, color='white', linestyle='--', alpha=0.5)

plt.title(f'{ticker} Stock Price Prediction Residuals with LSTM\n(Actual - Predicted)')
plt.xlabel('Date')
plt.ylabel('Residual ($)')
plt.grid(True, alpha=0.2)
plt.legend(loc='upper left')
plt.xticks(rotation=45)

# Add train/test split line
plt.axvline(x=split_date, color='white', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.show()

# Print model summary
model.summary()