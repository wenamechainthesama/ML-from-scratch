import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from sklearn import datasets

"""
Concept:
https://www.youtube.com/watch?v=azXCzI57Yfc

Implementation:
https://youtu.be/9IDXYHhAfGA?feature=shared
"""


class LDA:
    def __init__(self, num_components):
        self.num_components = num_components

    def fit(self, X: np.ndarray, y):
        num_features = X.shape[1]
        class_labels = np.unique(y)

        mean_overall = np.mean(X, axis=0)
        SW = np.zeros((num_features, num_features))
        SB = np.zeros((num_features, num_features))
        for label in class_labels:
            X_c = X[y == label]
            class_mean = np.mean(X_c, axis=0)
            SW += (X_c - class_mean).T.dot((X_c - class_mean))

            class_samples = X_c.shape[0]
            mean_diff = (class_mean - mean_overall).reshape(num_features, 1)
            SB += class_samples * mean_diff.dot(mean_diff.T)

        A = np.linalg.inv(SW).dot(SB)
        eigenvalues, eigenvectors = np.linalg.eig(A)
        eigenvectors = eigenvectors.T
        idxs = np.argsort(abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        self.linear_discriminants = eigenvectors[: self.num_components]

    def transform(self, X):
        return np.dot(X, self.linear_discriminants.T)


if __name__ == "__main__":
    data = datasets.load_iris()
    X, y = data.data, data.target

    lda = LDA(2)
    lda.fit(X, y)
    X_projected = lda.transform(X)

    print("Shape of X:", X.shape)
    print("Shape of transformed X:", X_projected.shape)

    x1, x2 = X_projected[:, 0], X_projected[:, 1]

    plt.scatter(
        x1, x2, c=y, edgecolor="none", alpha=0.8, cmap=colormaps.get_cmap("viridis")
    )

    plt.xlabel("Linear Discriminant 1")
    plt.ylabel("Linear Discriminant 2")
    plt.colorbar()
    plt.show()
