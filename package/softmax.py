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

y = np.array([0, 0, 0, 0, 1, 1, 1, 1])


# -------------------------
# 2. Functions
# -------------------------

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def categorical_cross_entropy(y, y_pred):
    return -np.mean(y * np.log(y_pred))


# -------------------------
# 3. Parameters
# -------------------------

N = X.shape[0]
n_features = X.shape[1]
n_classes = len(np.unique(y))

y_one_hot = np.eye(n_classes)[y]

W = np.random.randn(n_features, n_classes)
b = np.zeros((1, n_classes))

lr = 0.1


# -------------------------
# 4. Training
# -------------------------

for epoch in range(1000):

    # Forward
    Z = X @ W + b
    y_pred = softmax(Z)

    # Loss
    loss = categorical_cross_entropy(y_one_hot, y_pred)

    # Backward
    dZ = (y_pred - y_one_hot) / N
    dW = X.T @ dZ
    db = np.sum(dZ, axis=0, keepdims=True)

    # Update
    W -= lr * dW
    b -= lr * db


# -------------------------
# 5. Prediction
# -------------------------

probabilities = softmax(X @ W + b)

predictions = np.argmax(probabilities, axis=1)


# -------------------------
# 6. Results
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
