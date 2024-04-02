import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets, metrics
import matplotlib.pyplot as plt


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
    losses = []
    for i in range(len(y_true)):
        losses.append(
            (y_true[i] * np.log(proba[i])) + (1 - y_true[i]) * np.log(1 - proba[i])
        )
    return -np.array(losses).mean()


def roc_curve(y_true, y_pred):
    fpr = []
    tpr = []

    num_thresholds = len(y_true)
    thresholds = np.linspace(0, 1, num_thresholds)

    label1 = sum(y_true)
    label2 = len(y_true) - label1

    for threshold in thresholds:
        fp, tp = 0, 0
        threshold = round(threshold, 2)
        for i in range(len(y_pred)):
            if y_pred[i] >= threshold:
                if y_true[i] == 1:
                    tp += 1
                if y_true[i] == 0:
                    fp += 1
        fpr.append(fp / label2)
        tpr.append(tp / label1)

    return fpr, tpr, thresholds


def auc(fpr, tpr):
    return -np.trapz(tpr, fpr)


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
    print(metrics.classification_report(y_test, y_pred))
    print("Sklearn log loss", metrics.log_loss(y_test, y_proba[:, 1]))

    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_proba[:, 1])
    auc_score = metrics.auc(fpr, tpr)
    print("Sklearn AUC", auc_score)

    print("\n--------------------- Handmade ------------------------")
    print("Precision", precision(y_test, y_pred))
    print("Recall", recall(y_test, y_pred))
    print("Specificity", specificity(y_test, y_pred))
    print("F1 score", f1_score(y_test, y_pred))
    print("Accuracy", accuracy(y_test, y_pred))
    print("Log loss", log_loss(y_test, y_proba[:, 1]))

    fpr, tpr, thresholds = roc_curve(y_test, y_proba[:, 1])
    auc_score = auc(fpr, tpr)
    print("AUC", auc_score)

    # ROC AUC plot
    plt.plot(
        fpr,
        tpr,
        marker="o",
        color="darkorange",
        lw=0,
    )
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC curve, AUC = {round(auc_score, 2)}")
    plt.show()
