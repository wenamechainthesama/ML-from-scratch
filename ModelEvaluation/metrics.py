import numpy as np


def confusion_matrix(y_true, y_pred):
    final_matrix = []
    labels = np.unique(y_true)
    num_labels = len(labels)
    for idx, label in enumerate(labels):
        final_matrix.append([[] for _ in range(num_labels)])
        # np.where(y_pred == label)
        pass
