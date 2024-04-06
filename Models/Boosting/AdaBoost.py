import numpy as np
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

"""
Concept:
https://www.youtube.com/watch?v=LsK-xG1cLYA
"""


class Stump:
    def __init__(self, split_feature, threshold, left_value, right_value, weight):
        self.split_feature = split_feature
        self.threshold = threshold
        self.left_value = left_value
        self.right_value = right_value
        self.weight = weight


class AdaBoostClassifier:
    def __init__(self, num_stumps):
        self.num_stumps = num_stumps
        self.stumps = []

    def fit(self, X: np.ndarray, y):
        num_samples = X.shape[0]
        self.num_labels = len(np.unique(y))

        # Define initial sample weights
        sample_weights = np.full(num_samples, 1 / num_samples)
        for _ in range(self.num_stumps):
            # Data preparation: choose (almost) random samples from
            # initial data and use it to form new stump
            random_samples_idxs = np.random.choice(
                range(num_samples), size=num_samples, p=sample_weights
            )
            X = X[random_samples_idxs, :]
            y = y[random_samples_idxs]

            # Set up equal weights for all samples
            sample_weights = np.full(num_samples, 1 / num_samples)

            # Find best split (threshold and feature) using gini index
            best_split_feature_idx, best_threshold = self._best_split(X, y)

            # Determine stump values (left and right)
            left_idxs, right_idxs = self._split(
                X[:, best_split_feature_idx], best_threshold
            )

            counter = Counter(y[left_idxs])
            left_value = counter.most_common(1)[0][0]

            counter = Counter(y[right_idxs])
            right_value = counter.most_common(1)[0][0]

            # Calculate stump error
            left_error = 0

            # Find incorrectly classified samples
            left_error_idxs = np.argwhere(y[left_idxs] != left_value).flatten()

            # Add error sample weights to node error
            for left_error_idx in left_error_idxs:
                left_error += sample_weights[left_error_idx]

            # Same for right node
            right_error = 0
            right_error_idxs = np.argwhere(y[right_idxs] != right_value).flatten()
            for right_error_idx in right_error_idxs:
                right_error += sample_weights[right_error_idx]

            total_error = left_error + right_error

            # To prevent division by zero if total_error is zero or log error if total_error is 1
            if total_error == 0.0:
                total_error += 10 ** (-15)
            elif total_error == 1.0:
                total_error -= 10 ** (-15)

            # Calculate stump weight
            stump_weight = np.log((1 - total_error) / total_error) / 2

            # Find all incorrectly classified samples
            all_error_idxs = list(left_error_idxs) + list(right_error_idxs)

            # Increase weight for sample misclassification
            for idx in all_error_idxs:
                sample_weights[idx] *= np.exp(stump_weight)

            # Decrease weight for other samples (which were correctly classified)
            for idx in np.delete(range(num_samples), all_error_idxs):
                sample_weights[idx] *= np.exp(-stump_weight)

            # Normalize sample weights (divide by its sum each weight) and replace old onces with them
            sample_weights_sum = sum(sample_weights)
            sample_weights = [weight / sample_weights_sum for weight in sample_weights]

            # Create stump
            stump = Stump(
                best_split_feature_idx,
                best_threshold,
                left_value,
                right_value,
                stump_weight,
            )

            # Add stump and its weight to list
            self.stumps.append(stump)

    def _best_split(self, X: np.ndarray, y):
        best_gini = float("inf")
        best_split_feature_idx, best_threshold = None, None

        for feature_idx in range(X.shape[1]):
            feature_column = X[:, feature_idx]
            thresholds = np.unique(feature_column)
            for threshold in thresholds:
                gini = self._gini_impurity(feature_column, y, threshold)
                if gini < best_gini:
                    best_gini = gini
                    best_split_feature_idx = feature_idx
                    best_threshold = threshold

        return best_split_feature_idx, best_threshold

    def _gini_impurity(self, feature_column, y, threshold):
        # Split using threshold
        left_idxs, right_idxs = self._split(feature_column, threshold)

        # Calculate gini for each child
        left_gini = self._gini(y[left_idxs])
        right_gini = self._gini(y[right_idxs])

        # Calculate overall weighted gini impurity
        left_weight = len(left_idxs) / len(y)
        left_weighted_gini = left_gini * left_weight

        right_weight = len(right_idxs) / len(y)
        right_weighted_gini = right_gini * right_weight

        children_weighted_gini = left_weighted_gini + right_weighted_gini

        return children_weighted_gini

    def _split(self, feature_column, threshold):
        left_idxs = np.argwhere(feature_column <= threshold).flatten()
        right_idxs = np.argwhere(feature_column > threshold).flatten()
        return left_idxs, right_idxs

    def _gini(self, y):
        labels = np.unique(y)
        gini = 1
        for label in labels:
            counter = Counter(y)
            label_occurrence = counter[label]
            label_probability = label_occurrence / len(y)
            gini -= label_probability**2

        return gini

    def predict(self, X):
        return [self._predict(x) for x in X]

    def _predict(self, x):
        label_scores = [0 for _ in range(self.num_labels)]
        for stump in self.stumps:
            if x[stump.split_feature] <= stump.threshold:
                label_scores[stump.left_value] += stump.weight
            else:
                label_scores[stump.right_value] += stump.weight

        return np.argmax(label_scores)


if __name__ == "__main__":
    data = load_iris()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1082347
    )

    # X_train = np.array([
    #     [1, 1, 205],
    #     [0, 1, 180],
    #     [1, 0, 210],
    #     [1, 1, 167],
    #     [0, 1, 156],
    #     [0, 1, 125],
    #     [1, 0, 168],
    #     [1, 1, 172],
    # ])
    # # print(X_train)
    # y_train = np.array([1, 1, 1, 1, 0, 0, 0, 0])

    model = AdaBoostClassifier(num_stumps=7)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    def accuracy(y_true, y_pred):
        return np.sum(y_true == y_pred) / len(y_true)

    print("AdaBoost classification accuracy", accuracy(y_test, predictions))
