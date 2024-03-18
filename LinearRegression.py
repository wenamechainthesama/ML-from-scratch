import numpy as np
import matplotlib.pyplot as plt


# https://ajaykrish-krishnanrb.medium.com/multiple-linear-regression-from-scratch-using-python-ae150ac505c
class MultipleLinearRegression:
    def __init__(self, num_weights, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.weights = np.random.rand(num_weights)
        self.bias = 0.001 * np.random.rand()

    def loss(self, y_true, predictions):
        return np.mean(np.square(y_true - predictions))

    def gradient_descent(self, features, y_true, predictions):
        error = y_true - predictions
        dW = []
        for feature in features:
            dW.append(-2 * np.mean(error * feature))
        db = -2 * np.mean(error)
        return dW, db

    def optimize_model_parametes(self, features, y_true, predictions):
        dW, db = self.gradient_descent(features, y_true, predictions)
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dW[i]
        self.bias -= self.learning_rate * db

    def fit(self, x_train, y_true, epochs=100, verbose=False):
        history = []
        for epoch in range(epochs):
            predicts = self.predict(x_train)
            loss = self.loss(y_true, predicts)
            self.optimize_model_parametes(x_train, y_true, predicts)
            if verbose:
                print("epoch:", epoch, "loss:", loss)
            history.append(loss)
        return history

    def predict(self, features):
        return (np.transpose(features) @ self.weights) + self.bias

    def get_coefs(self):
        return self.weights, self.bias

    def evaluate(self, test_features, y_test):
        y_hat = self.predict(test_features)
        loss = self.loss(y_test, y_hat)
        return loss


x1 = 4 * np.random.rand(100, 1) - 2
x2 = 2 * np.random.rand(100, 1) + 3
x3 = 5 * np.random.rand(100, 1) - 1
x4 = 7 * np.random.rand(100, 1) + 4
y = 4 - 9 * x1 + 13 * x2 + 3 * x3 - 7 * x4 + 8 * np.random.rand(100, 1)
model = MultipleLinearRegression(num_weights=4)
history = model.fit(
    np.array([x1.flatten(), x2.flatten(), x3.flatten(), x4.flatten()]), y
)

# Plot learning curve
model_coef, bias = model.get_coefs()
plt.figure(figsize=(8, 5))
plt.plot(history)
plt.title(f"Learning curve # Learned Weights:{model_coef} and bias:{bias :.2f}")
plt.xlabel("Epochs")
plt.ylabel("Mean squared error")
plt.show()


# Read https://muthu.co/maths-behind-polynomial-regression/ for understanding math behind it
# Source: https://medium.com/codex/code-a-polynomial-regression-model-from-scratch-using-python-6f02d708177
class PolynomialRegression:
    def __init__(self, degree=5):
        self.degree = degree

    # See: https://muthu.co/wp-content/uploads/2018/06/Snip20191206_1.png
    def fit(self, X, y):
        A = []
        C = []
        for i in range(self.degree):
            A_row = []
            for j in range(self.degree):
                A_row.append(np.sum([x ** (i + j) for x in X]))
            A.append(A_row)
            C.append(np.sum(np.multiply([x**i for x in X], y)))
        self.W = np.linalg.inv(A) * np.transpose(C)

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            predictions.append(
                np.sum([weight * x**j for j, weight in enumerate(self.W)])
            )
        return predictions


x = 4 * np.random.rand(100, 1) - 2
y = 4 - x + 20 * x**2 + 10 * x**3 - 2 * x**4 + 8 * np.random.rand(100, 1)

model = PolynomialRegression(degree=4)
model.fit(x, y)
plt.scatter(x, y)
X_vals = np.linspace(-2, 2, 100).reshape(-1, 1)
y_vals = model.predict(X_vals)
plt.plot(X_vals, y_vals, c="red")
plt.show()


# Source: https://youtu.be/zMqhnpWgSS8?feature=shared
class LinearClassifier:
    def __init__(self, learning_rate=0.001) -> None:
        self.learning_rate = learning_rate
        self.weights = [0, -1]

    def fit(self, features, y_train):
        features = list(features)
        for i in range(len(features)):
            features[i] = list(features[i])
            features[i].append(1)
            features[i] = np.array(features[i])
        features = np.array(features)
        pt = np.sum([x * y for x, y in zip(features, y_train)], axis=0)
        xxt = np.sum([np.outer(x, x) for x in features], axis=0)
        self.weights = np.dot(pt, np.linalg.inv(xxt))

    def get_coefs(self):
        return self.weights


X = np.array([[9, 1], [8, 1], [7, 2], [10, 2], [5, 5], [5, 4], [6, 3], [6, 6]])
y = np.array([-1, -1, -1, -1, 1, 1, 1, 1])

model = LinearClassifier()
model.fit(X, y)
coefs = model.get_coefs()
line_x = list(range(max(X[:, 0])))
line_y = [-x * coefs[0] / coefs[1] - coefs[2] / coefs[1] for x in line_x]

x_0 = X[y == 1]
x_1 = X[y == -1]

plt.scatter(x_0[:, 0], x_0[:, 1], color="red")
plt.scatter(x_1[:, 0], x_1[:, 1], color="blue")
plt.plot(line_x, line_y, color="green")

plt.xlim([3, 11])
plt.ylim([0, 7])
plt.grid(True)
plt.show()
