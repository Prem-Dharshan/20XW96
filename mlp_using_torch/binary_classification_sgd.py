import torch
from torch.utils.data import DataLoader, TensorDataset

# -------------------------
# 1. Dataset
# -------------------------

X = torch.tensor(
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

y = torch.tensor([[0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0]])


# -------------------------
# 2. Create Dataset
# -------------------------

dataset = TensorDataset(X, y)


# -------------------------
# 3. Create Batches
# -------------------------

loader = DataLoader(dataset, batch_size=2, shuffle=True)


# -------------------------
# 4. Nonlinear MLP
# -------------------------

model = torch.nn.Sequential(
    torch.nn.Linear(2, 10), torch.nn.ReLU(), torch.nn.Linear(10, 1), torch.nn.Sigmoid()
)


# -------------------------
# 5. Loss Function
# -------------------------

loss_fn = torch.nn.BCELoss()


# -------------------------
# 6. SGD Optimizer
# -------------------------

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


# -------------------------
# 7. Training
# -------------------------

epochs = 1000

for epoch in range(epochs):

    for X_batch, y_batch in loader:

        # Forward
        y_pred = model(X_batch)

        # Loss
        loss = loss_fn(y_pred, y_batch)

        # Backward
        loss.backward()

        # Update weights
        optimizer.step()

        # Clear gradients
        optimizer.zero_grad()


# -------------------------
# 8. Prediction
# -------------------------

with torch.no_grad():

    probabilities = model(X)

    predictions = (probabilities >= 0.5).float()


# -------------------------
# 9. Results
# -------------------------

print("Probabilities:")
print(probabilities)

print("\nPredictions:")
print(predictions)

print("\nActual:")
print(y)
