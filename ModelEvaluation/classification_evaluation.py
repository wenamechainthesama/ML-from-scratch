import numpy as np


def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


def confusion_matrix(y_true, y_pred):
    num_labels = len(np.unique(y_true))
    confusion_matrix = np.zeros((num_labels, num_labels))

    for i in range(len(y_true)):
        confusion_matrix[y_true[i]][y_pred[i]] += 1

    return confusion_matrix


def recall(y_true, y_pred):
    pass


def precision(y_true, y_pred):
    pass


def f1_score(y_true, y_pred):
    pass


def specificity(y_true, y_pred):
    pass
