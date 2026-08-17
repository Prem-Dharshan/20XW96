import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)  # stops exp() overflow. cancels out.
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)  # axis=1 -> each ROW sums to 1


def predict(X, W, b):
    return softmax(X @ W + b)  # (N,f)@(f,C) + (C,) -> (N,C)


def one_hot(y, C):
    return np.eye(C)[y]  # (N,) -> (N,C)


def scores(y, P):
    lab = P.argmax(axis=1)  # was: P >= 0.5
    return {
        "CCE": log_loss(y, P),
        "Acc": accuracy_score(y, lab),
        "Prec": precision_score(y, lab, average="macro", zero_division=0),
        "Rec": recall_score(y, lab, average="macro", zero_division=0),
        "F1": f1_score(y, lab, average="macro", zero_division=0),
        "AUC": roc_auc_score(y, P, multi_class="ovr"),
    }


def train(X, y, W, b, lr=0.5, epochs=300):
    Y = one_hot(y, W.shape[1])
    hist = []
    for _ in range(epochs):
        P = predict(X, W, b)
        hist.append(scores(y, P))
        E = P - Y  # (N,C)
        W -= lr / len(X) * (X.T @ E)  # (f,N)@(N,C) -> (f,C)
        b -= lr / len(X) * E.sum(axis=0)  # collapse samples, keep classes
    return W, b, hist


def plot(hist, y, lab):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    for k in hist[0]:  # one line per metric
        ax[0].plot([h[k] for h in hist], label=k)
    ax[0].legend()
    ax[0].set_title("training")

    ConfusionMatrixDisplay.from_predictions(y, lab, ax=ax[1])
    plt.show()


X, y = make_blobs(n_samples=150, centers=3, n_features=2, random_state=42)
C = len(np.unique(y))

W, b, hist = train(X, y, np.zeros((X.shape[1], C)), np.zeros(C))

P = predict(X, W, b)
print("W shape", W.shape, " b shape", b.shape)
print("row sums (must be 1):", P.sum(axis=1)[:3])
print({k: round(v, 4) for k, v in scores(y, P).items()})
print(confusion_matrix(y, P.argmax(axis=1)))
plot(hist, y, P.argmax(axis=1))
