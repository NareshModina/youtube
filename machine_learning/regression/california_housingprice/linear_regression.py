# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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
# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate and print performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")

# Print feature coefficients
coefficients = pd.DataFrame({
    'Feature': features_to_keep,
    'Coefficient': model.coef_
})
print("\nFeature Coefficients:")
print(coefficients)