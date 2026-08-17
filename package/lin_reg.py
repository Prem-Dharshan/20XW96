import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


def predict(X, w, b):
    return X @ w + b


def scores(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {"MSE": mse,
            "SSE": mse * len(y_true),    # SSE is just MSE without the /N
            "R2": r2_score(y_true, y_pred)}


def train(X, y_true, w, b, lr=0.01, epochs=100):
    N = len(X)
    history = {"MSE": [], "SSE": [], "R2": []}

    for i in range(epochs):
        y_pred = predict(X, w, b)
        error = y_pred - y_true

        for k, v in scores(y_true, y_pred).items():
            history[k].append(v)

        w -= lr * (2 / N) * (X.T @ error)
        b -= lr * (2 / N) * error.sum()

        if i % 20 == 0:
            print(f"epoch {i:3d}  MSE {history['MSE'][-1]:.6f}  R2 {history['R2'][-1]:.4f}")

    return w, b, history


def plot(history):
    fig, ax = plt.subplots(1, len(history), figsize=(5 * len(history), 4))
    for a, (name, vals) in zip(ax, history.items()):
        a.plot(vals)
        a.set_title(name)
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


X = np.array([[1., 2, 4], [2, 3, 5], [3, 4, 6]])
y = np.array([2.5, 3.5, 4.5])

w, b, history = train(X, y, np.zeros(X.shape[1]), 0.0)

print("\nw", w, " b", round(b, 4))
print("pred", predict(X, w, b))
print(scores(y, predict(X, w, b)))
plot(history)
