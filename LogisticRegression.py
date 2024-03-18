import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


"""
Math:
https://www.youtube.com/watch?v=YMJtsYIp4kg
https://habr.com/ru/companies/io/articles/265007/
https://www.youtube.com/watch?v=49ck7kCyxr4

Implementation:
https://www.youtube.com/watch?v=YYEJ_GUguHw
"""


class LogisticRegression:
    def __init__(self, learning_rate=0.001, threshold=0.5) -> None:
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.weights = None
        self.bias = None

    def fit(self, X, y, epochs=1000):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(epochs):
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_pred)
            dw = 1 / num_samples * np.dot(X.T, predictions - y)
            db = 1 / num_samples * np.sum(predictions - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        return sigmoid(np.dot(self.weights, X.T) + self.bias)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return [0 if y < self.threshold else 1 for y in probabilities]


bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


def accuracy(y_pred, y_test):
    return np.sum(y_pred == y_test) / len(y_test)


acc = accuracy(y_pred, y_test)
print(acc)
