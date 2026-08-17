import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize

# -------------------------
# Dataset
# -------------------------

X = torch.tensor(
    [
        [1.0, 1.0],
        [1.0, 2.0],
        [2.0, 1.0],
        [5.0, 5.0],
        [5.0, 6.0],
        [6.0, 5.0],
        [9.0, 1.0],
        [9.0, 2.0],
        [8.0, 1.0],
    ]
)

y = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])


# -------------------------
# MLP
# -------------------------

model = torch.nn.Sequential(
    torch.nn.Linear(2, 10), torch.nn.ReLU(), torch.nn.Linear(10, 3)
)


# -------------------------
# Loss
# -------------------------

loss_fn = torch.nn.CrossEntropyLoss()


# -------------------------
# Optimizer
# -------------------------

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


# -------------------------
# Training
# -------------------------

epochs = 1000

losses = []

for epoch in range(epochs):

    # Forward
    logits = model(X)

    # Loss
    loss = loss_fn(logits, y)

    # Store loss
    losses.append(loss.item())

    # Backward
    loss.backward()

    # Update
    optimizer.step()

    # Clear gradients
    optimizer.zero_grad()


# -------------------------
# Prediction
# -------------------------

with torch.no_grad():

    logits = model(X)

    probabilities = torch.softmax(logits, dim=1)

    predictions = torch.argmax(probabilities, dim=1)


print("Probabilities:")
print(probabilities)

print("\nPredictions:")
print(predictions)

print("\nActual:")
print(y)


# -------------------------
# Convert to NumPy
# -------------------------

y_true = y.numpy()
y_pred = predictions.numpy()
y_prob = probabilities.numpy()


# -------------------------
# Metrics
# -------------------------

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(y_true, y_pred, average="macro")

recall = recall_score(y_true, y_pred, average="macro")

f1 = f1_score(y_true, y_pred, average="macro")

auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")


print("\nMetrics:")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)
print("AUC      :", auc)


# -------------------------
# Confusion Matrix
# -------------------------

cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:")
print(cm)

ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=torch.unique(y).numpy()
).plot()

plt.title("Confusion Matrix")
plt.show()


# -------------------------
# Training Loss Plot
# -------------------------

plt.figure(figsize=(8, 5))

plt.plot(losses)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")

plt.show()


# -------------------------
# ROC Curve
# -------------------------

y_true_binary = label_binarize(y_true, classes=torch.unique(y).numpy())

plt.figure(figsize=(8, 6))

for i in range(3):

    fpr, tpr, _ = roc_curve(y_true_binary[:, i], y_prob[:, i])

    plt.plot(fpr, tpr, label=f"Class {i}")


plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()
