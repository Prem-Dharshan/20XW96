import torch
from torch.utils.data import TensorDataset, DataLoader

# -------------------------
# 1. Dataset
# -------------------------

X = torch.tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0],
    [5.0],
    [6.0]
])

y = torch.tensor([
    [3.0],
    [5.0],
    [7.0],
    [9.0],
    [11.0],
    [13.0]
])


# -------------------------
# 2. Create dataset
# -------------------------

dataset = TensorDataset(X, y)


# -------------------------
# 3. Create batches
# -------------------------

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)


# -------------------------
# 4. Linear model
# -------------------------

model = torch.nn.Linear(1, 1)


# -------------------------
# 5. Loss function
# -------------------------

loss_fn = torch.nn.MSELoss()


# -------------------------
# 6. SGD optimizer
# -------------------------

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)


# -------------------------
# 7. Training
# -------------------------

epochs = 100

for epoch in range(epochs):

    for X_batch, y_batch in loader:

        # Forward
        y_pred = model(X_batch)

        # Loss
        loss = loss_fn(y_pred, y_batch)

        # Backward
        loss.backward()

        # Update
        optimizer.step()

        # Clear gradients
        optimizer.zero_grad()


# -------------------------
# 8. Results
# -------------------------

print("Weight:", model.weight.item())
print("Bias:", model.bias.item())