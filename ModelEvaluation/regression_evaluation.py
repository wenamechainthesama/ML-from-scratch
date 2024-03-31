import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn import metrics


def mean_squared_error(y_true, y_pred):
    return np.square(np.subtract(y_true, y_pred)).mean()


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_error(y_true, y_pred):
    return np.abs(np.subtract(y_true, y_pred)).mean()


def r_squared(y_true, y_pred):
    # SS - sum of squares
    y_avg = np.mean(y_true)
    SS_total = np.sum(np.square(np.array(y_true) - y_avg))
    SS_residuals = np.sum(np.square(np.array(np.subtract(y_true, y_pred))))
    return 1 - SS_residuals / SS_total


def adjusted_r_squared(y_true, y_pred, x_shape):
    num_samples, num_features = x_shape
    numerator = (1 - r_squared(y_true, y_pred)) * (num_samples - 1)
    denomenator = num_samples - num_features - 1
    return 1 - numerator / denomenator


if __name__ == "__main__":
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=345
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("---------------- Sklearn implementation -----------------")
    print("MAE", metrics.mean_absolute_error(y_test, y_pred))
    print("MSE", metrics.mean_squared_error(y_test, y_pred))
    print("R squared", metrics.r2_score(y_test, y_pred))
    print("\n--------------------- Handmade ------------------------")
    print("MAE", mean_absolute_error(y_test, y_pred))
    print("MSE", mean_squared_error(y_test, y_pred))
    print("R squared", r_squared(y_test, y_pred))
