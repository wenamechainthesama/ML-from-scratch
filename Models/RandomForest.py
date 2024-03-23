import numpy as np
from collections import Counter
from DecisionTree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

"""
Concept:
https://www.youtube.com/watch?v=CmRqP-HtxY4&list=PLA0M1Bcd0w8zxDIDOTQHsX68MCDOAJDtj&index=41
"""


class RandomForestClassifier:
    def __init__(self, num_trees, max_features, max_depth, num_samples_per_tree):
        self.num_samples_per_tree = num_samples_per_tree
        self.max_features = max_features
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X: np.ndarray, y):
        num_samples = X.shape[0]
        for _ in range(self.num_trees):
            tree = DecisionTreeClassifier(
                num_split_features=self.max_features, max_depth=self.max_depth
            )
            # Choosing samples for current tree from original data
            tree_data_idxs = np.random.choice(num_samples, self.num_samples_per_tree)
            tree_X = X[tree_data_idxs, :]
            tree_y = y[tree_data_idxs]

            tree.fit(tree_X, tree_y)

            self.trees.append(tree)

    def predict(self, X):
        multiple_decisions = [[] for _ in range(X.shape[0])]
        for tree in self.trees:
            tree_predictions = tree.predict(X)
            for prediction_idx, tree_prediction in enumerate(tree_predictions):
                multiple_decisions[prediction_idx].append(tree_prediction)

        forest_predictions = []
        for decisions in multiple_decisions:
            counter = Counter(decisions)
            forest_predictions.append(counter.most_common(1)[0][0])

        return forest_predictions


if __name__ == "__main__":
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=345
    )

    model = RandomForestClassifier(
        max_features=10, max_depth=3, num_trees=20, num_samples_per_tree=30
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    def accuracy(y_test, y_pred):
        return np.sum(y_test == y_pred) / len(y_test)

    print("Random forest classification accuracy", accuracy(y_test, predictions))
