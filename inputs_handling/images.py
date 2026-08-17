import os
import cv2 as cv
import numpy as np


# -------------------------
# 1. Dataset
# -------------------------

data_dir = "../dataset/images"

image_size = (64, 64)


# -------------------------
# 2. Read Images
# -------------------------

X = []
y = []


for filename in os.listdir(data_dir):

    path = os.path.join(
        data_dir,
        filename
    )

    # Read image as grayscale
    image = cv.imread(
        path,
        0
    )

    # Skip files that are not images
    if image is None:
        continue


    # -------------------------
    # Resize
    # -------------------------

    image = cv.resize(
        image,
        image_size
    )


    # -------------------------
    # Convert Image to Vector
    # -------------------------

    image = image.flatten()


    # -------------------------
    # Get Label and Store Sample
    # -------------------------

    if "cat" in filename.lower():

        X.append(image)
        y.append(0)

    elif "dog" in filename.lower():

        X.append(image)
        y.append(1)


# -------------------------
# 3. Convert to NumPy
# -------------------------

X = np.array(X)

y = np.array(y).reshape(-1, 1)


# -------------------------
# 4. Normalize
# -------------------------

X = X / 255.0


# -------------------------
# 5. Dataset Information
# -------------------------

print("X Shape:")
print(X.shape)

print("\ny Shape:")
print(y.shape)

print("\nX:")
print(X)

print("\ny:")
print(y)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.utils import gen_batches

# -------------------------
# 1. Dataset
# -------------------------

# X = np.array(
#     [
#         [1.0, 1.0],
#         [1.0, 2.0],
#         [2.0, 1.0],
#         [2.0, 2.0],
#         [3.0, 3.0],
#         [4.0, 3.0],
#         [3.0, 4.0],
#         [4.0, 4.0],
#     ]
# )

# y = np.array(
#     [
#         [0.0],
#         [0.0],
#         [0.0],
#         [0.0],
#         [1.0],
#         [1.0],
#         [1.0],
#         [1.0],
#     ]
# )


# -------------------------
# 2. Train-Test Split
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)


# -------------------------
# 3. Functions
# -------------------------


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def binary_cross_entropy(y, y_pred):
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))


# -------------------------
# 4. Parameters
# -------------------------

N = X_train.shape[0]

n_features = X_train.shape[1]
n_outputs = y_train.shape[1]

W = np.random.randn(n_features, n_outputs) / np.sqrt(n_features)

b = np.zeros((1, n_outputs))

lr = 0.1

epochs = 1000

batch_size = N
# batch_size = 1  → SGD
# batch_size = N  → Batch Gradient Descent


# -------------------------
# 5. Training
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

        y_pred = sigmoid(Z)

        # -------------------------
        # Loss
        # -------------------------

        loss = binary_cross_entropy(y_batch, y_pred)

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

    y_epoch_pred = sigmoid(X_train @ W + b)

    losses.append(binary_cross_entropy(y_train, y_epoch_pred))


# -------------------------
# 6. Prediction
# -------------------------

probabilities = sigmoid(X_test @ W + b)

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
# 9. Loss Plot
# -------------------------

plt.plot(range(1, epochs + 1), losses)

plt.xlabel("Epoch")
plt.ylabel("Binary Cross Entropy")
plt.title("Training Loss")

plt.show()
