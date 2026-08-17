import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

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
# 2. Train-Test Split
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)


# -------------------------
# 3. Functions
# -------------------------


def relu(z):
    return np.maximum(0, z)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def binary_cross_entropy(y, y_pred):
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))


# -------------------------
# 4. Parameters
# -------------------------

N = X_train.shape[0]

n_features = X_train.shape[1]
n_hidden = 10
n_outputs = y_train.shape[1]

W1 = np.random.randn(n_features, n_hidden)

b1 = np.zeros((1, n_hidden))

W2 = np.random.randn(n_hidden, n_outputs)

b2 = np.zeros((1, n_outputs))

lr = 0.1

epochs = 1000


# -------------------------
# 5. Training
# -------------------------

losses = []

for epoch in range(epochs):

    # -------------------------
    # Forward Pass
    # -------------------------

    Z1 = X_train @ W1 + b1

    A1 = relu(Z1)

    Z2 = A1 @ W2 + b2

    y_pred = sigmoid(Z2)

    # -------------------------
    # Loss
    # -------------------------

    loss = binary_cross_entropy(y_train, y_pred)

    # -------------------------
    # Backward Pass
    # -------------------------

    N = X_train.shape[0]

    dZ2 = (y_pred - y_train) / N

    dW2 = A1.T @ dZ2

    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = dZ2 @ W2.T

    dZ1 = dA1 * (Z1 > 0)

    dW1 = X_train.T @ dZ1

    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # -------------------------
    # Weight Update
    # -------------------------

    W2 -= lr * dW2

    b2 -= lr * db2

    W1 -= lr * dW1

    b1 -= lr * db1

    # -------------------------
    # Epoch Loss
    # -------------------------

    losses.append(loss)


# -------------------------
# 6. Prediction
# -------------------------

Z1 = X_test @ W1 + b1

A1 = relu(Z1)

Z2 = A1 @ W2 + b2

probabilities = sigmoid(Z2)

predictions = (probabilities >= 0.5).astype(int)


# -------------------------
# 7. Metrics
# -------------------------

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(y_test, predictions, zero_division=0)

recall = recall_score(y_test, predictions, zero_division=0)

f1 = f1_score(y_test, predictions, zero_division=0)

auc = roc_auc_score(y_test, probabilities)


# -------------------------
# 8. Results
# -------------------------

print("W1:")
print(W1)

print("\nb1:")
print(b1)

print("\nW2:")
print(W2)

print("\nb2:")
print(b2)

print("\nProbabilities:")
print(probabilities)

print("\nPredictions:")
print(predictions)

print("\nActual:")
print(y_test)

print("\nAccuracy:")
print(accuracy)

print("\nPrecision:")
print(precision)

print("\nRecall:")
print(recall)

print("\nF1 Score:")
print(f1)

print("\nAUC:")
print(auc)


# -------------------------
# 9. Loss Plot
# -------------------------

plt.plot(range(1, epochs + 1), losses)

plt.xlabel("Epoch")
plt.ylabel("Binary Cross Entropy")
plt.title("Training Loss")

plt.show()
