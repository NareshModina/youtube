import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

# Generate a simple 2D dataset
X, y = make_blobs(n_samples=60, centers=2, random_state=42, cluster_std=1.5)

# Define kernels and configurations
configs = [
    ('linear_hard', 'Linear (Hard Margin, C=1e5)', SVC(kernel='linear', C=1e5)),  # High C for hard margin
    ('linear_soft', 'Linear (Soft Margin, C=1)', SVC(kernel='linear', C=1)),      # Lower C for soft margin
    ('rbf', 'RBF Kernel', SVC(kernel='rbf', C=1)),
    ('poly', 'Polynomial (degree 3)', SVC(kernel='poly', degree=3, C=1))
]

# Create mesh grid for decision boundary and margin
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

for kernel_name, title, model in configs:
    # Train the model
    model.fit(X, y)
    
    # Compute decision boundary and margins
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Create figure with black background
    plt.figure(facecolor='black')
    ax = plt.gca()
    
    # Set axis spines, ticks, and labels to white
    for spine in ax.spines.values():
        spine.set_color('red')
    ax.tick_params(axis='both', colors='white')
    
    # Plot decision boundary (0-level) and margins (+1 and -1 levels) with enhanced visibility
    plt.contour(xx, yy, Z, levels=[-1], colors=['blue'], linestyles=['--'], linewidths=2, alpha=0.9)  # Margin -1
    plt.contour(xx, yy, Z, levels=[0], colors=['orange'], linestyles=['-'], linewidths=2.5, alpha=1.0)  # Decision boundary
    plt.contour(xx, yy, Z, levels=[1], colors=['blue'], linestyles=['--'], linewidths=2, alpha=0.9)   # Margin +1
    
    # Plot data points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k')
    
    # Highlight support vectors
    plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], 
                s=100, facecolors='none', edgecolors='red', linewidths=1.5)
    
    # Labels and title
    plt.title(f'SVM: {title}', color='white')
    plt.xlabel('Feature 1', color='white')
    plt.ylabel('Feature 2', color='white')
    
    # Save as high-quality PNG
    plt.savefig(f'svm_explain_{kernel_name}.png', dpi=300)
    plt.close()

print("Explanatory plots saved as svm_explain_linear_hard.png, svm_explain_linear_soft.png, svm_explain_rbf.png, and svm_explain_poly.png")