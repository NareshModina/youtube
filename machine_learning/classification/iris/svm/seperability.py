import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles
from sklearn.svm import SVC

# 1. Linearly Separable Dataset
X_sep, y_sep = make_blobs(n_samples=60, centers=2, random_state=42, cluster_std=1.0)

# 2. Non-Linearly Separable Dataset
X_nonsep, y_nonsep = make_circles(n_samples=60, noise=0.1, factor=0.5, random_state=42)

# Configurations: (dataset, labels, filename, title, model)
configs = [
    (X_sep, y_sep, 'linear_separable', 'Linearly Separable (Hard Margin, C=1e5)', 
     SVC(kernel='linear', C=1e5)),
    (X_nonsep, y_nonsep, 'linear_nonseparable', 'Non-Linearly Separable (Soft Margin, C=1)', 
     SVC(kernel='linear', C=1))
]

for X, y, filename, title, model in configs:
    # Train the model
    model.fit(X, y)
    
    # Create mesh grid for decision boundary and margin
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    
    # Compute decision boundary and margins
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Create figure with black background
    plt.figure(facecolor='black')
    ax = plt.gca()
    
    # Set axes background to black
    ax.set_facecolor('black')
    
    # Set axis spines, ticks, and labels to white
    for spine in ax.spines.values():
        spine.set_color('white')
    ax.tick_params(axis='both', colors='white')
    
    # Plot decision boundary and margins with enhanced visibility
    plt.contour(xx, yy, Z, levels=[-1], colors=['cyan'], linestyles=['--'], linewidths=2, alpha=0.9)  # Margin -1
    plt.contour(xx, yy, Z, levels=[0], colors=['yellow'], linestyles=['-'], linewidths=2.5, alpha=1.0)  # Decision boundary
    plt.contour(xx, yy, Z, levels=[1], colors=['cyan'], linestyles=['--'], linewidths=2, alpha=0.9)   # Margin +1
    
    # Plot data points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='Wistia', edgecolors='k')
    
    # Highlight support vectors
    plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], 
                s=100, facecolors='none', edgecolors='white', linewidths=1.5)
    
    # Labels and title
    plt.title(f'SVM: {title}', color='white')
    plt.xlabel('Feature 1', color='white')
    plt.ylabel('Feature 2', color='white')
    
    # Save as high-quality PNG
    plt.savefig(f'svm_explain_{filename}.png', dpi=300)
    plt.close()

print("Explanatory plots saved as svm_explain_linear_separable.png and svm_explain_linear_nonseparable.png with fully black backgrounds")