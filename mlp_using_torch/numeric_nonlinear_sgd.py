import torch
from torch.utils.data import DataLoader, TensorDataset

# -------------------------
# 1. Dataset
# -------------------------

X = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])

y = torch.tensor([[1.0], [4.0], [9.0], [16.0], [25.0], [36.0]])


# -------------------------
# 2. Dataset
# -------------------------

dataset = TensorDataset(X, y)


# -------------------------
# 3. Batches
# -------------------------

loader = DataLoader(dataset, batch_size=2, shuffle=True)


# -------------------------
# 4. Nonlinear MLP
# -------------------------

model = torch.nn.Sequential(
    torch.nn.Linear(1, 10), torch.nn.ReLU(), torch.nn.Linear(10, 1)
)


# -------------------------
# 5. Loss
# -------------------------

loss_fn = torch.nn.MSELoss()


# -------------------------
# 6. SGD optimizer
# -------------------------

optimizer = torch.optim.SGD(model.parameters(), lr=0.001)


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

        # Update
        optimizer.step()

        # Clear gradients
        optimizer.zero_grad()


# -------------------------
# 8. Test
# -------------------------

X_test = torch.tensor([[7.0], [8.0]])

prediction = model(X_test)

print("Prediction:")
print(prediction)
