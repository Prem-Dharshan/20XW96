import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def predict(X, w, b):
    return sigmoid(X @ w + b)  # probabilities, not labels


def scores(y_true, y_prob):
    y_lab = (y_prob >= 0.5).astype(int)
    return {
        "BCE": log_loss(y_true, y_prob),
        "Acc": accuracy_score(y_true, y_lab),
        "F1": f1_score(y_true, y_lab),
        "AUC": roc_auc_score(y_true, y_prob),
    }


def train(X, y, w, b, lr=0.1, epochs=200):
    N = len(X)
    history = {k: [] for k in scores(y, predict(X, w, b))}

    for i in range(epochs):
        y_prob = predict(X, w, b)
        error = y_prob - y

        for k, v in scores(y, y_prob).items():
            history[k].append(v)

        w -= lr / N * (X.T @ error)
        b -= lr / N * error.sum()

    return w, b, history


def plot(history, X, y, w, b):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    # 1. metrics
    for name, vals in history.items():
        ax[0].plot(vals, label=name)
    ax[0].legend()
    ax[0].set_title("training")

    # 2. data + boundary   (c=y colours the two classes automatically)
    ax[1].scatter(X[:, 0], X[:, 1], c=y)
    xs = np.array([X[:, 0].min(), X[:, 0].max()])  # a line needs only 2 points
    ax[1].plot(xs, -(w[0] * xs + b) / w[1], "k--")
    ax[1].set_title("decision boundary")

    # 3. ROC
    fpr, tpr, _ = roc_curve(y, predict(X, w, b))
    ax[2].plot(fpr, tpr)
    ax[2].plot([0, 1], [0, 1], "r--")
    ax[2].set_title("ROC")

    plt.tight_layout()
    plt.show()


X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=42)

w, b, history = train(X, y, np.zeros(2), 0.0)
y_prob = predict(X, w, b)
y_lab = (y_prob >= 0.5).astype(int)

print("w", w, " b", round(b, 4))
print({k: round(v, 4) for k, v in scores(y, y_prob).items()})
print(
    "precision",
    round(precision_score(y, y_lab), 4),
    " recall",
    round(recall_score(y, y_lab), 4),
)
print("confusion matrix\n", confusion_matrix(y, y_lab))
plot(history, X, y, w, b)
