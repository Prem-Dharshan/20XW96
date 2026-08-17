import torch

# -------------------------
# 1. Dataset
# -------------------------

X = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])

y = torch.tensor([[3.0], [5.0], [7.0], [9.0], [11.0], [13.0]])


# -------------------------
# 2. Linear model
# -------------------------

model = torch.nn.Linear(1, 1)


# -------------------------
# 3. Loss function
# -------------------------

loss_fn = torch.nn.MSELoss()


# -------------------------
# 4. SGD optimizer
# -------------------------

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


# -------------------------
# 5. Training
# -------------------------

epochs = 100

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
# 6. Results
# -------------------------

print("Weight:", model.weight.item())
print("Bias:", model.bias.item())
