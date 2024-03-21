import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

"""
Concept:
https://www.youtube.com/watch?v=HZGCoVF3YvM
https://www.youtube.com/watch?v=lFJbZ6LVxN8

Implementation:
https://www.youtube.com/watch?v=TLInuAorxqE
"""


class NaiveBayes:
    def fit(self, X, y):
        num_samples, num_features = X.shape
        self._classes = np.unique(y)
        num_classes = len(self._classes)

        # Calculate mean, var, and prior for each class
        self._mean = np.zeros((num_classes, num_features), dtype=np.float64)
        self._var = np.zeros((num_classes, num_features), dtype=np.float64)
        self._priors = np.zeros(num_classes, dtype=np.float64)

        for idx, label in enumerate(self._classes):
            X_label = X[y == label]
            self._mean[idx, :] = X_label.mean(axis=0)
            self._var[idx, :] = X_label.var(axis=0)
            self._priors[idx] = X_label.shape[0] / float(num_samples)

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        posteriors = []

        # Calculate posterior probability for each class
        for idx in range(len(self._classes)):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = posterior + prior
            posteriors.append(posterior)

        # Return class with the highest posterior
        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        # pdf - probability density function
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator


if __name__ == "__main__":
    X, y = datasets.make_classification(
        n_samples=1000, n_features=10, n_classes=2, random_state=54132342
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    predictions = nb.predict(X_test)

    def accuracy(y_true, y_pred):
        return np.sum(y_true == y_pred) / len(y_true)

    print("Naive Bayes classification accuracy", accuracy(y_test, predictions))
