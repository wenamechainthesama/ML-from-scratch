import numpy as np


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
