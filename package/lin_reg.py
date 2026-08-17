import numpy as np


# -------------------------
# 1. Dataset
# -------------------------

X = np.array(
    [
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0],
        [6.0],
    ]
)

y = np.array(
    [
        [3.0],
        [5.0],
        [7.0],
        [9.0],
        [11.0],
        [13.0],
    ]
)


# -------------------------
# 2. Loss Functions
# -------------------------

def sse(y, y_pred):
    return np.sum((y - y_pred) ** 2)


def mse(y, y_pred):
    return np.mean((y - y_pred) ** 2)


# -------------------------
# 3. Parameters
# -------------------------

N = X.shape[0]

W = np.random.randn(X.shape[1], y.shape[1])
b = np.zeros((1, y.shape[1]))

lr = 0.01


# -------------------------
# 4. Training
# -------------------------

for epoch in range(1000):

    # Forward
    y_pred = X @ W + b

    # Loss
    loss = mse(y, y_pred)

    # Backward
    dZ = 2 * (y_pred - y) / N

    dW = X.T @ dZ
    db = np.sum(dZ, axis=0, keepdims=True)

    # Update
    W -= lr * dW
    b -= lr * db


# -------------------------
# 5. Prediction
# -------------------------

predictions = X @ W + b


# -------------------------
# 6. Results
# -------------------------

print("Weights:")
print(W)

print("\nBias:")
print(b)

print("\nPredictions:")
print(predictions)

print("\nActual:")
print(y)

print("\nSSE:")
print(sse(y, predictions))

print("\nMSE:")
print(mse(y, predictions))
