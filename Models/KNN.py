import numpy as np
import matplotlib.pyplot as plt
from itertools import islice


def euclidean_distance(pointA, pointB):
    return np.sum(np.power(np.subtract(pointA, pointB), 2)) ** 0.5


class KNNClassifier:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        y_pred = []
        for test_point in X:
            distances = {}
            for i, train_point in enumerate(self.X):
                distances.setdefault(i, euclidean_distance(test_point, train_point))

            distances = dict(sorted(distances.items(), key=lambda item: item[1]))

            votes = {}
            for key in dict(islice(distances.items(), self.k)).keys():
                point_class = self.y[key]
                if point_class in votes:
                    votes[point_class] += 1
                else:
                    votes.setdefault(point_class, 1)

            class_pred = max(votes.items(), key=lambda item: item[1])[0]
            y_pred.append(class_pred)

        return y_pred


if __name__ == "__main__":
    # Illustrative example
    dots = [
        [1, 1],
        [1, 1.25],
        [1.75, 1],
        [1.25, 0.8],
        [2, 1.5],
        [2, 4.25],
        [2.7, 4.25],
        [2.5, 4],
        [3, 5],
        [3.5, 3],
        [3.9, 2.7],
        [4.25, 3],
        [4.75, 4],
    ]
    y = [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    for i in range(len(dots)):
        color = None
        if y[i] == 1:
            color = "red"
        elif y[i] == 2:
            color = "blue"
        elif y[i] == 3:
            color = "green"
        plt.scatter(dots[i][0], dots[i][1], c=color)

    dots_to_predict = [[2.75, 3], [2.5, 2.7], [2.25, 2.35]]

    model = KNNClassifier(k=5)
    model.fit(dots, y)
    prediction = model.predict(dots_to_predict)
    for i, dot in enumerate(dots_to_predict):
        color = None
        if prediction[i] == 1:
            color = "red"
        elif prediction[i] == 2:
            color = "blue"
        elif prediction[i] == 3:
            color = "green"
        plt.scatter(dot[0], dot[1], color=color)
        plt.annotate("test", (dot[0], dot[1]))

    plt.xlim(0, 6)
    plt.ylim(0, 6)
    plt.grid(True)
    plt.show()
