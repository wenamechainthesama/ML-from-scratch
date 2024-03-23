from sklearn import datasets, decomposition
import matplotlib.pyplot as plt
import numpy as np


class PCA:
    def __init__(self, num_components=3):
        self.num_components = num_components

    def transform(self, X: np.ndarray):
        num_samples, num_features = X.shape

        # Find center of the data
        center = np.mean(X, axis=0)

        # Find PCs
        num_pcs = min(num_samples, num_features)
        pcs = []
        for _ in range(num_pcs):
            pc = self.fit_line(X, pcs)
            pcs.append(pc)

        # Find eigenvalues for each PC

        # Compare variance of each PC and total variance

        # Get rid of the most useless PCs

        # Plot final 2D or 3D graph

    def fit_line(self, X, perpendicular_lines):
        pass
        # Find projections of data points onto the line


iris = datasets.load_iris()
X = iris.data
y = iris.target

fig = plt.figure(1, figsize=(4, 3))

ax = fig.add_subplot(111, projection="3d", elev=48, azim=134)
ax.set_position([0, 0, 0.95, 1])


pca = decomposition.PCA(n_components=3)
pca.fit(X)
X = pca.transform(X)

for name, label in [("Setosa", 0), ("Versicolour", 1), ("Virginica", 2)]:
    ax.text3D(
        X[y == label, 0].mean(),
        X[y == label, 1].mean() + 1.5,
        X[y == label, 2].mean(),
        name,
        horizontalalignment="center",
        bbox=dict(alpha=0.5, edgecolor="w", facecolor="w"),
    )
# Reorder the labels to have colors matching the cluster results
y = np.choose(y, [1, 2, 0]).astype(float)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.nipy_spectral, edgecolor="k")

ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])

plt.show()
