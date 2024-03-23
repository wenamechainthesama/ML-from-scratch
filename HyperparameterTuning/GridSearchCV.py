import numpy as np
from itertools import product
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


class GridSearchCV:
    def __init__(self, estimator, params_grid: dict, cv: int, evaluation_function):
        """
        Evaluation function is just function which takes y_pred and y_true as parameters
        and returns some score which shows how good or bad the model is
        """
        self.estimator = estimator
        self.params_grid = params_grid
        self.evaluation_function = evaluation_function
        self.cv = cv

    def fit(self, X, y):
        # Get all possible combinations of params
        params_combos = list(product(*self.params_grid.values()))
        params_names = self.params_grid.keys()

        # Find best params
        params_cv_scores = {}

        # For each combination of params
        for params in params_combos:
            # Create model
            model = self.estimator(**dict(list(zip(params_names, params))))

            # Perform cross validation
            cv_scores = []
            for _ in range(self.cv):
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=1 / self.cv
                )

                # Train model
                model.fit(X_train, y_train)

                # Calculate cv score
                y_pred = model.predict(X_test)
                score = self.evaluation_function(y_pred, y_test)
                cv_scores.append(score)

            # Get average cv score
            avg_cv_score = sum(cv_scores) / len(cv_scores)

            params_cv_scores.setdefault(params, avg_cv_score)

        # Determine the best params based on average cv scores
        self.best_params = max(params_cv_scores, key=params_cv_scores.get)

        # Fit model with best params (optional)
        self.best_model = self.estimator(
            **dict(list(zip(params_names, self.best_params)))
        )
        self.best_model.fit(X, y)

    def predict(self, X):
        return self.best_model.predict(X)


if __name__ == "__main__":
    dataset = datasets.load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.2
    )
    clf = GridSearchCV(
        RandomForestClassifier,
        {
            "n_estimators": [1, 5, 10],
            "criterion": ["gini", "entropy", "log_loss"],
            "max_depth": [5, 10, 15, 20],
        },
        cv=5,
        evaluation_function=lambda y_true, y_pred: np.sum(y_true == y_pred)
        / len(y_true),
    )
    clf.fit(dataset.data, dataset.target)
    print(clf.best_params)
