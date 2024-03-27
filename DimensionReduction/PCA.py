import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from sklearn import datasets


"""
Concept:
https://www.youtube.com/watch?v=FgakZw6K1QQ

Implementation:
https://www.youtube.com/watch?v=Rjr62b_h7S4
"""

class PCA:
    def __init__(self, num_components):
        self.num_components = num_components

    def fit(self, X):
        # Center data
        self.mean = np.mean(X, axis=0)
        X -= self.mean

        # Calculate covariance
        covariance = np.cov(X.T)

        # Calculate eigenvectors and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(covariance)
        eigenvectors = eigenvectors.T

        # Sort eigenvectors
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvalues = eigenvectors[idxs]

        self.components = eigenvectors[: self.num_components]

    def transform(self, X):
        X -= self.mean
        return np.dot(X, self.components.T)


if __name__ == "__main__":
    data = datasets.load_iris()
    X = data.data
    y = data.target

    pca = PCA(2)
    pca.fit(X)
    X_projected = pca.transform(X)

    print("Shape of X:", X.shape)
    print("Shape of transformed X:", X_projected.shape)

    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]

    plt.scatter(
        x1, x2, c=y, edgecolor="none", alpha=0.8, cmap=colormaps.get_cmap("viridis")
    )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar()
    plt.show()
