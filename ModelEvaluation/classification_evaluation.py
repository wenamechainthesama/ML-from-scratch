import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, log_loss
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets


def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


def confusion_matrix(y_true, y_pred):
    num_labels = len(np.unique(y_true))
    confusion_matrix = np.zeros((num_labels, num_labels))

    for i in range(len(y_true)):
        confusion_matrix[y_true[i]][y_pred[i]] += 1

    return confusion_matrix


def recall(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    labels = np.unique(y_true)
    recall = []
    for label in labels:
        tp, fn = 0, 0
        for row in range(len(cm)):
            for column in range(len(cm)):
                if row == column and row == label:
                    tp += cm[row][column]
                elif row == label and column != label:
                    fn += cm[row][column]
        label_recall = tp / (tp + fn)
        recall.append(label_recall)

    return np.array(recall)


def precision(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    labels = np.unique(y_true)
    precision = []
    for label in labels:
        tp, fp = 0, 0
        for row in range(len(cm)):
            for column in range(len(cm)):
                if row == column and row == label:
                    tp += cm[row][column]
                elif row != label and column == label:
                    fp += cm[row][column]
        label_precision = tp / (tp + fp)
        precision.append(label_precision)

    return np.array(precision)


def f1_score(y_true, y_pred):
    numerator = 2 * precision(y_true, y_pred) * recall(y_true, y_pred)
    denomenator = precision(y_true, y_pred) + recall(y_true, y_pred)
    return numerator / denomenator


def specificity(y_true, y_pred):
    return recall(y_true, y_pred)[::-1]


def log_loss(y_true, proba):
    loss = 0
    for i in range(len(y_true)):
        loss -= (y_true[i] * np.log(proba[i])) + (1 - y_true[i]) * np.log(
            1 - proba[i]
        ).mean()
    return loss


if __name__ == "__main__":
    X, y = datasets.make_classification(
        n_samples=1000, n_features=10, n_classes=2, random_state=212441
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=12314
    )

    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    print("---------------- Sklearn implementation -----------------")
    print(classification_report(y_test, y_pred))
    print("Sklearn log loss", log_loss(y_test, y_proba))
    print("\n--------------------- Handmade ------------------------")
    print("Precision", precision(y_test, y_pred))
    print("Recall", recall(y_test, y_pred))
    print("Specificity", specificity(y_test, y_pred))
    print("F1 score", f1_score(y_test, y_pred))
    print("Accuracy", accuracy(y_test, y_pred))
    print("Log loss", log_loss(y_test, y_proba))
