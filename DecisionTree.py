from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np

"""
Concept behind:
https://www.youtube.com/watch?v=ZVR2Way4nwQ
https://youtu.be/JfSaNKDowww?feature=shared
https://youtu.be/i139J4SuoLM?feature=shared
https://youtu.be/TOj-A3C-tJ8?feature=shared

Implementation:
https://www.youtube.com/watch?v=NxEHSAfFlK8
"""


class Node:
    def __init__(
        self,
        feature=None,
        threshold=None,
        left_child=None,
        right_child=None,
        value=None,
    ):
        self.split_feature = feature
        self.threshold = threshold
        self.left = left_child
        self.right = right_child
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTreeClassifier:
    def __init__(self, num_split_features, max_depth):
        self.num_split_features = num_split_features
        self.max_depth = max_depth
        self.root_node = None

    def fit(self, X, y):
        self.root_node = self._grow_tree(X, y)

    def _grow_tree(self, X: np.ndarray, y, depth=0):
        num_classes = len(np.unique(y))

        if num_classes == 1 or depth >= self.max_depth:
            counter = Counter(y)
            value = counter.most_common(1)[0][0]
            return Node(value=value)

        # Find best split
        best_split_feature_idx, best_threshold = self._best_split(X, y)

        # Create child nodes
        left_idxs, right_idxs = self._split(
            X[:, best_split_feature_idx], best_threshold
        )
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        return Node(best_split_feature_idx, best_threshold, left, right)

    def _best_split(self, X, y):
        best_gain = -1
        best_split_feature_idx, best_threshold = None, None

        num_features = X.shape[1]
        if self.num_split_features <= 0:
            self.num_split_features = num_features
        self.num_split_features = min(num_features, self.num_split_features)
        feature_idxs = np.random.choice(
            num_features, self.num_split_features, replace=False
        )
        for feature_idx in feature_idxs:
            feature_column = X[:, feature_idx]
            thresholds = np.unique(feature_column)
            for threshold in thresholds:
                gain = self._information_gain(feature_column, y, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_split_feature_idx = feature_idx
                    best_threshold = threshold
        return best_split_feature_idx, best_threshold

    def _information_gain(self, feature_column, y, threshold):
        parent_entropy = self._entropy(y)

        # Split using threshold
        left_idxs, right_idxs = self._split(feature_column, threshold)

        # Calculate children entropy
        left_entropy = self._entropy(y[left_idxs])
        right_entropy = self._entropy(y[right_idxs])

        # Calculate information gain
        left_weight = len(left_idxs) / len(y)
        left_weighted_entropy = left_entropy * left_weight

        right_weight = len(right_idxs) / len(y)
        right_weighted_entropy = right_entropy * right_weight

        children_weighted_entropy = left_weighted_entropy + right_weighted_entropy
        gain = parent_entropy - children_weighted_entropy

        return gain

    def _entropy(self, y):
        labels = np.unique(y)
        entropy = 0
        for label in labels:
            counter = Counter(y)
            label_occurrence = counter[label]
            label_probability = label_occurrence / len(y)
            if label_probability > 0:
                entropy -= label_probability * np.log2(label_probability)
        return entropy

    def _split(self, feature_column, threshold):
        left_idxs = np.argwhere(feature_column <= threshold).flatten()
        right_idxs = np.argwhere(feature_column > threshold).flatten()
        return left_idxs, right_idxs

    def predict(self, X):
        pred = [self._traverse_tree(x, self.root_node) for x in X]
        return pred

    def _traverse_tree(self, x, node: Node):
        if node.is_leaf_node():
            return node.value

        if x[node.split_feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


if __name__ == "__main__":
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=3221348
    )

    model = DecisionTreeClassifier(num_split_features=5, max_depth=3)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    def accuracy(y_true, y_pred):
        return np.sum(y_true == y_pred) / len(y_true)

    print("Decision Tree classification accuracy", accuracy(y_test, predictions))
