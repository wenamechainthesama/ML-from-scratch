"""
Statistics:
    Distributions:
        https://www.youtube.com/watch?v=8idr1WZ1A7Q
        https://www.youtube.com/watch?v=ZA4JkHKZM50
        https://www.youtube.com/watch?v=zeJD6dqJ5lo
        https://www.youtube.com/watch?v=cy8r7WSuT1I
    p-values:
        https://youtu.be/vemZtEM63GY?feature=shared
        https://youtu.be/JQc3yx0-Q9E?feature=shared
    Covariance:
        https://youtu.be/qtaqvPAeEJY?feature=shared
        https://www.youtube.com/watch?v=152tSYtiQbw
    Correlation:
        https://youtu.be/xZ_z8KWkhXE?feature=shared
    R-squared:
        https://youtu.be/2AQKmw14mHM?feature=shared
"""

"""
Concept:
https://www.youtube.com/watch?v=EWd1xRkyEog

Implementation:
https://github.com/ScienceKot/mysklearn/blob/master/Gaussian%20Mixture%20Models/GMM.py
"""

import numpy as np


class GMM:
    def __init__(self, n_components, max_iterations=100):
        self.n_componets = n_components
        self.max_iterations = max_iterations
        self.components_names = [
            f"component{index}" for index in range(self.n_componets)
        ]
        self.pi = [1 / self.n_componets for _ in range(self.n_componets)]

    def multivariate_normal(self, X, mean_vector, covariance_matrix):
        return (
            (2 * np.pi) ** (-len(X) / 2)
            * np.linalg.det(covariance_matrix) ** (-1 / 2)
            * np.exp(
                -np.dot(
                    np.dot((X - mean_vector).T, np.linalg.inv(covariance_matrix)),
                    (X - mean_vector),
                )
                / 2
            )
        )

    def fit(self, X):
        new_X = np.array_split(X, self.n_componets)
        self.mean_vector = [np.mean(x, axis=0) for x in new_X]
        self.covariance_matrixes = [np.cov(x.T) for x in new_X]
        for _ in range(self.max_iterations):
            """--------------------------   E - STEP   --------------------------"""
            # Initiating the r matrix, every row contains the probabilities
            # for every cluster for this row
            self.r = np.zeros((len(X), self.n_componets))
            # Calculating the r matrix
            for n in range(len(X)):
                for k in range(self.n_componets):
                    self.r[n][k] = self.pi[k] * self.multivariate_normal(
                        X[n], self.mean_vector[k], self.covariance_matrixes[k]
                    )
                    self.r[n][k] /= sum(
                        [
                            self.pi[j]
                            * self.multivariate_normal(
                                X[n], self.mean_vector[j], self.covariance_matrixes[j]
                            )
                            for j in range(self.n_componets)
                        ]
                    )
            N = np.sum(self.r, axis=0)
            """ --------------------------   M - STEP   -------------------------- """
            # Initializing the mean vector as a zero vector
            self.mean_vector = np.zeros((self.n_componets, len(X[0])))
            # Updating the mean vector
            for k in range(self.n_componets):
                for n in range(len(X)):
                    self.mean_vector[k] += self.r[n][k] * X[n]
            self.mean_vector = [
                1 / N[k] * self.mean_vector[k] for k in range(self.n_componets)
            ]
            # Initiating the list of the covariance matrixes
            self.covariance_matrixes = [
                np.zeros((len(X[0]), len(X[0]))) for k in range(self.n_componets)
            ]
            # Updating the covariance matrices
            for k in range(self.n_componets):
                self.covariance_matrixes[k] = np.cov(
                    X.T, aweights=(self.r[:, k]), ddof=0
                )
            self.covariance_matrixes = [
                1 / N[k] * self.covariance_matrixes[k] for k in range(self.n_componets)
            ]
            # Updating the pi list
            self.pi = [N[k] / len(X) for k in range(self.n_componets)]

    def predict(self, X):
        probas = []
        for n in range(len(X)):
            probas.append(
                [
                    self.multivariate_normal(
                        X[n], self.mean_vector[k], self.covariance_matrixes[k]
                    )
                    for k in range(self.n_componets)
                ]
            )
        cluster = []
        for proba in probas:
            cluster.append(self.components_names[proba.index(max(proba))])
        return cluster
