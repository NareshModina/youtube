import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Download stock data
print("Downloading stock data...")
ticker = "AAPL"  # Apple stock
start_date = "2020-01-01"
end_date = "2024-02-20"

stock_data = yf.download(ticker, start=start_date, end=end_date)
print("Data downloaded!")

# Feature engineering
def create_features(df):
    df = df.copy()
    
    # Technical indicators
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['STD_20'] = df['Close'].rolling(window=20).std()
    
    # Price changes
    df['Returns'] = df['Close'].pct_change()
    df['Returns_5'] = df['Close'].pct_change(periods=5)
    df['Returns_20'] = df['Close'].pct_change(periods=20)
    
    # Volume features
    df['Volume_SMA_5'] = df['Volume'].rolling(window=5).mean()
    df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
    
    # Price ranges
    df['Daily_Range'] = df['High'] - df['Low']
    df['Daily_Range_SMA_5'] = df['Daily_Range'].rolling(window=5).mean()
    
    return df

# Create features
df = create_features(stock_data)
df = df.dropna()  # Remove rows with NaN values

# Prepare features and target
feature_columns = ['Open', 'High', 'Low', 'Volume', 
                  'SMA_5', 'SMA_20', 'STD_20',
                  'Returns', 'Returns_5', 'Returns_20',
                  'Volume_SMA_5', 'Volume_SMA_20',
                  'Daily_Range', 'Daily_Range_SMA_5']

X = df[feature_columns].values
y = df['Close'].values

# Split the data - use time series split
split_idx = int(len(df) * 0.8)
X_train = X[:split_idx]
X_test = X[split_idx:]
y_train = y[:split_idx]
y_test = y[split_idx:]

# Scale features
X_scaler = StandardScaler()
X_train_scaled = X_scaler.fit_transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

# Scale target
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()  # Flatten to 1D
y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).flatten()  # Flatten to 1D

# Initialize SGD Regressor
sgd_reg = SGDRegressor(
    loss='squared_error',
    penalty='l2',
    alpha=0.0001,
    learning_rate='adaptive',
    eta0=0.01,
    max_iter=1000,
    tol=1e-5,
    random_state=42
)

# Training parameters
n_epochs = 100
batch_size = 32
n_batches = len(X_train_scaled) // batch_size
train_errors = []
val_errors = []

print("\nTraining SGD regressor...")
best_val_error = float('inf')
epochs_without_improvement = 0
patience = 5

for epoch in range(n_epochs):
    # Shuffle the training data
    shuffle_idx = np.random.permutation(len(X_train_scaled))
    X_train_shuffled = X_train_scaled[shuffle_idx]
    y_train_shuffled = y_train_scaled[shuffle_idx]
    
    for batch in range(n_batches):
        start_idx = batch * batch_size
        end_idx = start_idx + batch_size
        X_batch = X_train_shuffled[start_idx:end_idx]
        y_batch = y_train_shuffled[start_idx:end_idx]
        
        sgd_reg.partial_fit(X_batch, y_batch)
    
    # Calculate errors
    train_pred_scaled = sgd_reg.predict(X_train_scaled)
    val_pred_scaled = sgd_reg.predict(X_test_scaled)
    
    # Reshape predictions for inverse transform
    train_pred = y_scaler.inverse_transform(train_pred_scaled.reshape(-1, 1)).flatten()
    val_pred = y_scaler.inverse_transform(val_pred_scaled.reshape(-1, 1)).flatten()
    
    train_error = mean_squared_error(y_train, train_pred)
    val_error = mean_squared_error(y_test, val_pred)
    train_errors.append(train_error)
    val_errors.append(val_error)
    
    if val_error < best_val_error:
        best_val_error = val_error
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{n_epochs}")
        print(f"Training MSE: {train_error:.4f}")
        print(f"Validation MSE: {val_error:.4f}")
    
    if epochs_without_improvement >= patience:
        print(f"\nEarly stopping triggered at epoch {epoch + 1}")
        break

# Final predictions
y_pred_scaled = sgd_reg.predict(X_test_scaled)
y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)

print("\nFinal Model Performance:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# Plot predictions vs actual
plt.figure(figsize=(15, 6))
plt.plot(df.index[split_idx:], y_test, label='Actual', alpha=0.8)
plt.plot(df.index[split_idx:], y_pred, label='Predicted', alpha=0.8)
plt.title(f'{ticker} Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# Plot feature importance
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Coefficient': np.abs(sgd_reg.coef_)
})
feature_importance = feature_importance.sort_values('Coefficient', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(data=feature_importance, x='Coefficient', y='Feature')
plt.title('Feature Importance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot training progress
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_errors) + 1), train_errors, label='Training Error')
plt.plot(range(1, len(val_errors) + 1), val_errors, label='Validation Error')
plt.title('Training and Validation Error Over Time')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True)
plt.show()

# Print prediction statistics
print("\nPrediction Statistics:")
print("\nActual Stock Prices:")
print(pd.Series(y_test).describe())
print("\nPredicted Stock Prices:")
print(pd.Series(y_pred).describe())