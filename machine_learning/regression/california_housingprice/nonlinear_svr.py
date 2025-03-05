# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load the California housing dataset
california = fetch_california_housing()

# Create a pandas DataFrame with features
df = pd.DataFrame(california.data, columns=california.feature_names)

# Print all features
print("Original features in the dataset:")
print(df.columns.tolist())
print("\nFirst few rows of the original dataset:")
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

# Scale the features (important for SVR)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the SVR model with linear kernel
# model = SVR(kernel="poly", C=100, gamma="auto", degree=2, epsilon=0.1, coef0=1) # Polynomial kernel
model = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1) # RBF kernel
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Calculate and print performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"Mean Squared Error: {mse:.2f}") 
print(f"R-squared Score: {r2:.2f}")