import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.utils import gen_batches

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
# 2. Train-Test Split
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)


# -------------------------
# 3. Activation Function
# -------------------------


def linear(z):
    return z


# -------------------------
# 4. Loss Functions
# -------------------------


def sse(y, y_pred):
    return np.sum((y - y_pred) ** 2)


def mse(y, y_pred):
    return np.mean((y - y_pred) ** 2)


# -------------------------
# 5. Parameters
# -------------------------

N = X_train.shape[0]

n_features = X_train.shape[1]
n_outputs = y_train.shape[1]

W = np.random.randn(n_features, n_outputs)

b = np.zeros((1, n_outputs))

lr = 0.01
epochs = 1000

batch_size = N
# batch_size = 1  → SGD
# batch_size = N  → Batch Gradient Descent


# -------------------------
# 6. Training
# -------------------------

losses = []

for epoch in range(epochs):

    # -------------------------
    # Shuffle
    # -------------------------

    indices = np.random.permutation(N)

    X_train = X_train[indices]
    y_train = y_train[indices]

    # -------------------------
    # Batches
    # -------------------------

    for batch in gen_batches(N, batch_size):

        X_batch = X_train[batch]
        y_batch = y_train[batch]

        n_batch = len(X_batch)

        # -------------------------
        # Forward Pass
        # -------------------------

        Z = X_batch @ W + b

        y_pred = linear(Z)

        # -------------------------
        # Loss
        # -------------------------

        loss = mse(y_batch, y_pred)

        # -------------------------
        # Backward Pass
        # -------------------------

        dZ = 2 * (y_pred - y_batch) / n_batch

        dW = X_batch.T @ dZ

        db = np.sum(dZ, axis=0, keepdims=True)

        # -------------------------
        # Weight Update
        # -------------------------

        W -= lr * dW

        b -= lr * db

    # -------------------------
    # Epoch Loss
    # -------------------------

    y_epoch_pred = linear(X_train @ W + b)

    losses.append(mse(y_train, y_epoch_pred))


# -------------------------
# 7. Prediction
# -------------------------

predictions = linear(X_test @ W + b)


# -------------------------
# 8. Metrics
# -------------------------

mse_value = mean_squared_error(y_test, predictions)

sse_value = sse(y_test, predictions)

r2 = r2_score(y_test, predictions)


# -------------------------
# 9. Results
# -------------------------

print("Weights:")
print(W)

print("\nBias:")
print(b)

print("\nPredictions:")
print(predictions)

print("\nActual:")
print(y_test)

print("\nMSE:")
print(mse_value)

print("\nSSE:")
print(sse_value)

print("\nR² Score:")
print(r2)


# -------------------------
# 10. Loss Plot
# -------------------------

plt.plot(range(1, epochs + 1), losses)

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss")

plt.show()
