import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Create a DataFrame for better visualization
feature_names = iris.feature_names
iris_df = pd.DataFrame(X, columns=feature_names)
iris_df['target'] = y
iris_df['target_names'] = iris_df['target'].map({
    0: iris.target_names[0],
    1: iris.target_names[1],
    2: iris.target_names[2]
})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the logistic regression model
model = LogisticRegression(multi_class='auto', max_iter=1000) # multi_class='auto' selects the best option
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create confusion matrix visualization
plt.figure(figsize=(10, 8))
ax = plt.axes()
ax.set_facecolor('none')
sns.heatmap(cm, annot=True, fmt='d', cmap='Grays',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Print feature importance
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': np.abs(model.coef_).mean(axis=0)
})
feature_importance = feature_importance.sort_values('Coefficient', ascending=False)
print("\nFeature Importance:")
print(feature_importance)