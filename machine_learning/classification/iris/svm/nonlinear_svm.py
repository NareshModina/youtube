# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns

# Step 1: Load the Iris dataset
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]  # Use only petal length and petal width for 2D visualization
y = iris.target  # Target labels (0: Setosa, 1: Versicolor, 2: Virginica)

print("Dataset loaded! Features:", iris.feature_names[2:4])
print("Classes:", iris.target_names)

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# Step 3: Preprocess the data (standardize features)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Data standardized!")

# Step 4: Train the SVM Classifier
svm_clf = SVC(kernel='rbf', C=1.0, gamma='auto', random_state=42)  # RBF kernel for non-linear separation
svm_clf.fit(X_train_scaled, y_train)
print("SVM model trained!")

# Step 5: Make predictions and evaluate
y_pred = svm_clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

# Step 6: Visualize the decision boundary
def plot_decision_boundary(X, y, model, title):
    h = 0.02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.xlabel('Petal Length (standardized)')
    plt.ylabel('Petal Width (standardized)')
    plt.title(title)
    plt.show()

# Plot decision boundary for training data
plot_decision_boundary(X_train_scaled, y_train, svm_clf, "SVM Decision Boundary (Training Data)")

# Plot decision boundary for test data
plot_decision_boundary(X_test_scaled, y_test, svm_clf, "SVM Decision Boundary (Test Data)")

# Step 7: Bonus - Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()