# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load the dataset (update the file path to where you saved Advertising.csv)
data = pd.read_csv('advertising.xls')

# Drop the index column if it exists
data = data.drop(columns=['Unnamed: 0'], errors='ignore')

# Extract features
X = data[['TV','Radio','Newspaper']]
y = data['Sales']  # Sales revenue

# Create and fit the model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Calculate regression metrics
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)

# Get the slope and intercept
slope = model.coef_[0]
intercept = model.intercept_
print(f"Slope: {slope:.4f}, Intercept: {intercept:.4f}")
print(f"R² Score: {r2:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")