import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2]  # First two features for 2D plotting
y = iris.target

# Define the three SVM models
kernels = [('linear', 'Linear'), ('rbf', 'RBF'), ('poly', 'Polynomial (degree 3)')]
models = [
    SVC(kernel='linear'),
    SVC(kernel='rbf'),
    SVC(kernel='poly', degree=3)
]

# Create mesh grid for decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

# Train and plot for each kernel
for i, (kernel, title) in enumerate(kernels):
    model = models[i]
    model.fit(X, y)
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Create new figure with black background
    plt.figure(facecolor='black')
    ax = plt.gca()  # Get current axes
    
    # Set axis spines to white
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    
    # Set tick and tick label colors to white
    ax.tick_params(axis='both', colors='white')
    
    # Plot decision boundary and data
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k')
    plt.title(f'SVM Decision Boundary - {title}', color='white')
    plt.xlabel('Sepal Length', color='white')
    plt.ylabel('Sepal Width', color='white')
    
    # Save with high quality (300 DPI)
    plt.savefig(f'svm_classification_{kernel}.png', dpi=300)
    plt.close()

print("High-quality plots saved as svm_classification_linear.png, svm_classification_rbf.png, and svm_classification_poly.png with black backgrounds and white axes")