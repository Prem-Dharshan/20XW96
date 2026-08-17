import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import (log_loss, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score,
                             confusion_matrix, roc_curve)


def predict(X, w, b):
    return 1 / (1 + np.exp(-np.clip(X @ w + b, -500, 500)))    # sigmoid


def scores(y, p):
    lab = (p >= 0.5).astype(int)
    return {"BCE": log_loss(y, p), "Acc": accuracy_score(y, lab),
            "Prec": precision_score(y, lab), "Rec": recall_score(y, lab),
            "F1": f1_score(y, lab), "AUC": roc_auc_score(y, p)}


def train(X, y, w, b, lr=0.1, epochs=200):
    hist = []
    for _ in range(epochs):
        p = predict(X, w, b)
        hist.append(scores(y, p))          # list of dicts, no pre-declaring keys
        e = p - y
        w -= lr / len(X) * (X.T @ e)
        b -= lr / len(X) * e.sum()
    return w, b, hist


def plot(hist, X, y, w, b):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    for k in hist[0]:                                    # keys come from scores()
        ax[0].plot([h[k] for h in hist], label=k)
    ax[0].legend()

    ax[1].scatter(X[:, 0], X[:, 1], c=y)
    xs = np.array([X[:, 0].min(), X[:, 0].max()])
    ax[1].plot(xs, -(w[0] * xs + b) / w[1], "k--")       # w.x + b = 0

    fpr, tpr, _ = roc_curve(y, predict(X, w, b))
    ax[2].plot(fpr, tpr)
    ax[2].plot([0, 1], [0, 1], "r--")

    for a, t in zip(ax, ["training", "boundary", "ROC"]):
        a.set_title(t)
    plt.tight_layout()
    plt.show()


X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=42)
w, b, hist = train(X, y, np.zeros(2), 0.0)

print({k: round(v, 4) for k, v in scores(y, predict(X, w, b)).items()})
print(confusion_matrix(y, (predict(X, w, b) >= 0.5).astype(int)))
plot(hist, X, y, w, b)