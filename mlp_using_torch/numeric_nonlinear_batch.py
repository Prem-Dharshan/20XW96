import torch

# -------------------------
# 1. Dataset
# -------------------------

X = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])

y = torch.tensor([[1.0], [4.0], [9.0], [16.0], [25.0], [36.0]])


# -------------------------
# 2. Nonlinear MLP
# -------------------------

model = torch.nn.Sequential(
    torch.nn.Linear(1, 10), torch.nn.ReLU(), torch.nn.Linear(10, 1)
)


# -------------------------
# 3. Loss
# -------------------------

loss_fn = torch.nn.MSELoss()


# -------------------------
# 4. SGD optimizer
# -------------------------

optimizer = torch.optim.SGD(model.parameters(), lr=0.001)


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
# 6. Test
# -------------------------

X_test = torch.tensor([[7.0], [8.0]])

prediction = model(X_test)

print("Prediction:")
print(prediction)
