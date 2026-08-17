from matplotlib import pyplot as plt
import numpy as np


def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)


def sse(y_pred, y_true):
    return np.sum((y_pred - y_true) ** 2)


def R2(y_pred, y_true):
    ss_res = sse(y_pred, y_true)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def activation(x):
    return x


def predict(X, w, b):
    return activation(X @ w + b)


def training(X, y_true, w, b, lr=0.01, epochs=100, convergence_tol=1e-6):

    history = {"MSE": [], "SSE": [], "R2": []}

    N = len(X)

    for i in range(epochs):

        y_pred = predict(X, w, b)
        error = y_pred - y_true

        history["MSE"].append(mse(y_pred, y_true))
        history["SSE"].append(sse(y_pred, y_true))
        history["R2"].append(R2(y_pred, y_true))

        grad_w = (2 / N) * (X.T @ error)
        grad_b = (2 / N) * np.sum(error)

        w -= lr * grad_w
        b -= lr * grad_b

        if i % 10 == 0:
            print(f"Epoch {i}: MSE = {history['MSE'][-1]:.6f}  R2 = {history['R2'][-1]:.4f}")

        # CHANGED: index into the MSE list, not the old flat history
        if len(history["MSE"]) > 1 and abs(history["MSE"][-1] - history["MSE"][-2]) < convergence_tol:
            print(f"Converged at epoch {i}.")
            break

    history["MSE"].append(mse(y_pred, y_true))
    history["SSE"].append(sse(y_pred, y_true))
    history["R2"].append(R2(y_pred, y_true))

    return w, b, history


def plot_metrics(history, log=False):
    """history = dict of {name: list of values}"""
    n = len(history)
    fig, ax = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        ax = [ax]

    for a, (name, vals) in zip(ax, history.items()):
        a.plot(vals)
        a.set_title(name)
        a.set_xlabel("Epoch")
        a.grid(alpha=0.3)
        if log and min(vals) > 0:            # CHANGED: log arg actually used
            a.set_yscale("log")              # guard: log of <=0 is invalid (R2 goes negative)

    plt.tight_layout()
    plt.show()


def main():
    X = np.array([[1, 2, 4],
                  [2, 3, 5],
                  [3, 4, 6]])

    y_true = np.array([2.5, 3.5, 4.5])
    w = np.array([1.0, 1.0, 1.0])
    b = 0.5

    w, b, history = training(X, y_true, w.copy(), b, lr=0.01, epochs=100)

    y_pred = predict(X, w, b)

    print(f"\nFinal w: {w}, b: {b:.4f}")
    print(f"Predictions: {y_pred}")
    print(f"True:        {y_true}")
    print(f"MSE: {history['MSE'][-1]:.6f}   R2: {history['R2'][-1]:.4f}")
    print(f"epochs recorded: {len(history['MSE'])}")

    plot_metrics(history, log=True)


main()