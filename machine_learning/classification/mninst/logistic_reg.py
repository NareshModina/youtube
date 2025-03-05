import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load MNIST dataset (this may take a few moments)
print("Loading MNIST dataset...")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
print("Dataset loaded!")

# Convert data to float32 to save memory
X = X.astype('float32')

# Take a subset of the data to speed up training (optional)
# Remove or modify these lines if you want to use the full dataset
n_samples = 10000
X = X[:n_samples]
y = y[:n_samples]

# Normalize the data
X = X / 255.0

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training logistic regression model...")
# Create and train the logistic regression model
# Increased max_iter because MNIST is more complex than Iris
model = LogisticRegression(multi_class='multinomial', max_iter=1000, n_jobs=-1)
model.fit(X_train, y_train)
print("Model training completed!")

# Make predictions
y_pred = model.predict(X_test)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")

# Print detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create confusion matrix visualization
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Visualize some example predictions
def plot_example_predictions(X_test, y_test, y_pred, num_examples=5):
    plt.figure(figsize=(15, 3))
    for i in range(num_examples):
        plt.subplot(1, num_examples, i + 1)
        plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
        plt.title(f'Pred: {y_pred[i]}\nTrue: {y_test[i]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Show some example predictions
print("\nExample predictions:")
plot_example_predictions(X_test, y_test, y_pred)

# Calculate and display performance metrics for each digit
print("\nPer-digit accuracy:")
for digit in sorted(set(y_test)):
    mask = y_test == digit
    digit_accuracy = accuracy_score(y_test[mask], y_pred[mask])
    print(f"Digit {digit}: {digit_accuracy:.4f}")