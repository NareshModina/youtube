import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

# Load MNIST dataset
print("Loading MNIST dataset...")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
print("Dataset loaded!")

# Convert data to float32 to save memory
X = X.astype('float32')

# Normalize the data
X = X / 255.0

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize SGD Classifier
sgd_clf = SGDClassifier(
    loss='log_loss',  # Using log loss for probability estimates
    learning_rate='optimal',
    max_iter=100,
    tol=1e-3,
    random_state=42,
    n_jobs=-1
)

# Training with partial_fit to monitor progress
n_epochs = 10
batch_size = 1000
n_batches = len(X_train) // batch_size
train_scores = []

print("\nTraining SGD classifier...")
for epoch in range(n_epochs):
    # Shuffle the training data
    shuffle_idx = np.random.permutation(len(X_train))
    X_train_shuffled = X_train[shuffle_idx]
    y_train_shuffled = y_train[shuffle_idx]
    
    for batch in range(n_batches):
        start_idx = batch * batch_size
        end_idx = start_idx + batch_size
        X_batch = X_train_shuffled[start_idx:end_idx]
        y_batch = y_train_shuffled[start_idx:end_idx]
        
        # Partial fit on the batch
        if batch == 0 and epoch == 0:
            sgd_clf.partial_fit(X_batch, y_batch, classes=np.unique(y))
        else:
            sgd_clf.partial_fit(X_batch, y_batch)
        
    # Calculate training score after each epoch
    train_score = sgd_clf.score(X_train, y_train)
    train_scores.append(train_score)
    print(f"Epoch {epoch + 1}/{n_epochs}, Training accuracy: {train_score:.4f}")

# Make predictions
y_pred = sgd_clf.predict(X_test)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nFinal Test Accuracy: {accuracy:.4f}")

# Print detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot training progress
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_epochs + 1), train_scores, marker='o')
plt.title('Training Progress')
plt.xlabel('Epoch')
plt.ylabel('Training Accuracy')
plt.grid(True)
plt.show()

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create confusion matrix visualization
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Function to visualize misclassified examples
def plot_misclassified_examples(X_test, y_test, y_pred, num_examples=10):
    misclassified_idx = np.where(y_pred != y_test)[0]
    num_examples = min(num_examples, len(misclassified_idx))
    
    plt.figure(figsize=(20, 4))
    for i, idx in enumerate(misclassified_idx[:num_examples]):
        plt.subplot(1, num_examples, i + 1)
        plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
        plt.title(f'Pred: {y_pred[idx]}\nTrue: {y_test[idx]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Show misclassified examples
print("\nMisclassified Examples:")
plot_misclassified_examples(X_test, y_test, y_pred)

# Calculate per-class metrics
print("\nPer-digit accuracy:")
for digit in sorted(set(y_test)):
    mask = y_test == digit
    digit_accuracy = accuracy_score(y_test[mask], y_pred[mask])
    print(f"Digit {digit}: {digit_accuracy:.4f}")