import torch

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

for epoch in range(epochs):

    # Forward
    logits = model(X)

    # Loss
    loss = loss_fn(logits, y)

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
