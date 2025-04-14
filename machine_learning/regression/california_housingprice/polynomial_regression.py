# This script compares polynomial regression with linear regression using the California housing dataset.
# It demonstrates how to create polynomial features, fit a linear regression model, and evaluate the performance of both models.

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Load the California housing dataset
california = fetch_california_housing()

# Create a pandas DataFrame with features
df = pd.DataFrame(california.data, columns=california.feature_names)

# Print all features
print("Original features in the dataset:")
print(df.columns.tolist())
print("First few rows of the original dataset:")
print(df.head())

# Remove longitude and latitude columns
features_to_keep = [col for col in california.feature_names if col not in ['Latitude', 'Longitude']]
X = df[features_to_keep]
y = california.target

# Print remaining features after removal
print("\nFeatures after removing Latitude and Longitude:")
print(X.columns.tolist())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Linear Regression ---
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Make predictions
y_pred_linear = linear_model.predict(X_test)

# Calculate performance metrics for linear regression
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

print("\nLinear Regression Performance:")
print(f"Mean Squared Error: {mse_linear:.2f}")
print(f"R-squared Score: {r2_linear:.2f}")

# Print feature coefficients for linear regression
coefficients_linear = pd.DataFrame({
    'Feature': features_to_keep,
    'Coefficient': linear_model.coef_
})
print("\nLinear Regression Feature Coefficients:")
print(coefficients_linear)

# --- Polynomial Regression ---
degree = 2
polyreg = make_pipeline(PolynomialFeatures(degree), LinearRegression())

# Train the model
polyreg.fit(X_train, y_train)

# Make predictions
y_pred_poly = polyreg.predict(X_test)

# Calculate performance metrics for polynomial regression
mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print("\nPolynomial Regression Performance (Degree=3):")
print(f"Mean Squared Error: {mse_poly:.2f}")
print(f"R-squared Score: {r2_poly:.2f}")

# Extract coefficients from polynomial regression
linear_model = polyreg.named_steps['linearregression']
poly_features = polyreg.named_steps['polynomialfeatures']
feature_names = poly_features.get_feature_names_out(features_to_keep)

# Print feature coefficients for polynomial regression
coefficients_poly = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': linear_model.coef_
})
print("\nPolynomial Regression Feature Coefficients:")
print(coefficients_poly)

# Compare models
print("\nModel Comparison:")
print(f"Linear Regression - MSE: {mse_linear:.2f}, R²: {r2_linear:.2f}")
print(f"Polynomial Regression - MSE: {mse_poly:.2f}, R²: {r2_poly:.2f}")
if mse_poly < mse_linear:
    print("Polynomial regression shows improvement over linear regression (lower MSE).")
else:
    print("Polynomial regression does not improve over linear regression (higher or equal MSE).")