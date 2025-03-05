import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Fetch stock data
ticker = "AAPL"
stock_data = yf.download(ticker, start="2019-01-01", end="2024-01-01")

# Create features from technical indicators
def add_technical_indicators(df):
    # Moving averages
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Volume
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

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
train_pred = model.predict(X_train_scaled)
test_pred = model.predict(X_test_scaled)

# Calculate metrics
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
train_r2 = r2_score(y_train, train_pred)
test_r2 = r2_score(y_test, test_pred)

# Visualize results
plt.style.use('dark_background')
plt.figure(figsize=(15, 8))

# Plot actual vs predicted
plt.plot(y_train.index, y_train, label='Actual Price (Train)', color='cyan', alpha=0.5)
plt.plot(y_test.index, y_test, label='Actual Price (Test)', color='yellow', alpha=0.5)
plt.plot(y_train.index, train_pred, label='Predicted Price (Train)', color='blue', linestyle='--')
plt.plot(y_test.index, test_pred, label='Predicted Price (Test)', color='red', linestyle='--')

plt.title(f'{ticker} Stock Price Prediction\nTrain R²: {train_r2:.4f} | Test R²: {test_r2:.4f}\nTrain RMSE: ${train_rmse:.2f} | Test RMSE: ${test_rmse:.2f}')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.grid(True, alpha=0.2)
plt.legend(loc='upper left')
plt.xticks(rotation=45)

# Add train/test split line
split_date = y_test.index[0]
plt.axvline(x=split_date, color='white', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.show()

# Print feature importance
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': np.abs(model.coef_)
})
print("\nFeature Importance:")
print(feature_importance.sort_values('Importance', ascending=False))