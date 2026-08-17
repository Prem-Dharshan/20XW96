import torch

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
# 2. MLP
# -------------------------

model = torch.nn.Sequential(
    torch.nn.Linear(2, 10), torch.nn.ReLU(), torch.nn.Linear(10, 1), torch.nn.Sigmoid()
)


# -------------------------
# 3. Binary Cross Entropy
# -------------------------

loss_fn = torch.nn.BCELoss()


# -------------------------
# 4. SGD optimizer
# -------------------------

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


# -------------------------
# 5. Training
# -------------------------

epochs = 1000

for epoch in range(epochs):

    # Forward
    y_pred = model(X)

    # Loss
    loss = loss_fn(y_pred, y)

    # Backward
    loss.backward()

    # Update
    optimizer.step()

    # Clear gradients
    optimizer.zero_grad()


# -------------------------
# 6. Prediction
# -------------------------

with torch.no_grad():

    probabilities = model(X)

    predictions = (probabilities >= 0.5).float()


print("Probabilities:")
print(probabilities)

print("\nPredictions:")
print(predictions)

print("\nActual:")
print(y)
