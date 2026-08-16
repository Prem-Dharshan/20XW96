import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.utils import gen_batches

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
# 2. Train-Test Split
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)


# -------------------------
# 3. Activation Function
# -------------------------


def softmax(z):

    z = z - np.max(z, axis=1, keepdims=True)

    exp_z = np.exp(z)

    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# -------------------------
# 4. Loss Function
# -------------------------


def categorical_cross_entropy(y, y_pred):

    return -np.mean(y * np.log(y_pred))


# -------------------------
# 5. Parameters
# -------------------------

N = X_train.shape[0]

n_features = X_train.shape[1]
n_classes = len(np.unique(y))

y_train_one_hot = np.eye(n_classes)[y_train]

W = np.random.randn(n_features, n_classes)

b = np.zeros((1, n_classes))

lr = 0.1
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
    y_train_one_hot = y_train_one_hot[indices]

    # -------------------------
    # Batches
    # -------------------------

    for batch in gen_batches(N, batch_size):

        X_batch = X_train[batch]
        y_batch = y_train_one_hot[batch]

        n_batch = len(X_batch)

        # -------------------------
        # Forward Pass
        # -------------------------

        Z = X_batch @ W + b

        y_pred = softmax(Z)

        # -------------------------
        # Loss
        # -------------------------

        loss = categorical_cross_entropy(y_batch, y_pred)

        # -------------------------
        # Backward Pass
        # -------------------------

        dZ = (y_pred - y_batch) / n_batch

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

    y_epoch_pred = softmax(X_train @ W + b)

    losses.append(categorical_cross_entropy(y_train_one_hot, y_epoch_pred))


# -------------------------
# 7. Prediction
# -------------------------

probabilities = softmax(X_test @ W + b)

predictions = np.argmax(probabilities, axis=1)


# -------------------------
# 8. Metrics
# -------------------------

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(y_test, predictions, average="macro", zero_division=0)

recall = recall_score(y_test, predictions, average="macro", zero_division=0)

f1 = f1_score(y_test, predictions, average="macro", zero_division=0)

auc = roc_auc_score(y_test, probabilities, multi_class="ovr")


# -------------------------
# 9. Results
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
# 10. Loss Plot
# -------------------------

plt.plot(range(1, epochs + 1), losses)

plt.xlabel("Epoch")
plt.ylabel("Categorical Cross Entropy")
plt.title("Training Loss")

plt.show()
