import numpy as np


# -------------------------
# 1. Dataset
# -------------------------

X = np.array(
    [
        [1.0, 1.0],
        [1.0, 2.0],
        [2.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
        [4.0, 3.0],
        [3.0, 4.0],
        [4.0, 4.0],
    ]
)

y = np.array(
    [
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [1.0],
        [1.0],
        [1.0],
        [1.0],
    ]
)


# -------------------------
# 2. Activation Function
# -------------------------

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# -------------------------
# 3. Loss Function
# -------------------------

def binary_cross_entropy(y, y_pred):

    return -np.mean(
        y * np.log(y_pred)
        + (1 - y) * np.log(1 - y_pred)
    )


# -------------------------
# 4. Parameters
# -------------------------

n_features = X.shape[1]
n_outputs = y.shape[1]

W = np.random.randn(n_features, n_outputs)
b = np.zeros((1, n_outputs))

lr = 0.1


# -------------------------
# 5. Training
# -------------------------

epochs = 1000

for epoch in range(epochs):

    # -------------------------
    # Forward Pass
    # -------------------------

    Z = X @ W + b

    y_pred = sigmoid(Z)


    # -------------------------
    # Loss
    # -------------------------

    loss = binary_cross_entropy(y, y_pred)


    # -------------------------
    # Backward Pass
    # -------------------------

    N = X.shape[0]

    dZ = (y_pred - y) / N

    dW = X.T @ dZ

    db = np.sum(dZ)


    # -------------------------
    # Weight Update
    # -------------------------

    W -= lr * dW

    b -= lr * db


# -------------------------
# 6. Prediction
# -------------------------

Z = X @ W + b

probabilities = sigmoid(Z)

predictions = (probabilities >= 0.5).astype(float)


# -------------------------
# 7. Results
# -------------------------

print("Weights:")
print(W)

print("\nBias:")
print(b)

print("\nProbabilities:")
print(probabilities)

print("\nPredictions:")
print(predictions)

print("\nActual:")
print(y)