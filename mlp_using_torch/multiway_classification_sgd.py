import torch
from torch.utils.data import DataLoader, TensorDataset

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
# Dataset + DataLoader
# -------------------------

dataset = TensorDataset(X, y)

loader = DataLoader(dataset, batch_size=3, shuffle=True)


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
# SGD
# -------------------------

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


# -------------------------
# Training
# -------------------------

epochs = 1000

for epoch in range(epochs):

    for X_batch, y_batch in loader:

        # Forward
        logits = model(X_batch)

        # Loss
        loss = loss_fn(logits, y_batch)

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
