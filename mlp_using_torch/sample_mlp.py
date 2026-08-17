import torch

# -------------------------
# 1. Input
# -------------------------

x = torch.tensor([1.0, 2.0, 3.0])


# -------------------------
# 2. Parameters
# -------------------------

W1 = torch.tensor(
    [[1.0, 2.0, 1.0], [2.0, 1.0, 1.0], [1.0, 1.0, 2.0]], requires_grad=True
)

b1 = torch.tensor([1.0, 0.0, -1.0], requires_grad=True)

W2 = torch.tensor([[1.0, 2.0, 1.0]], requires_grad=True)

b2 = torch.tensor([1.0], requires_grad=True)


# -------------------------
# 3. Target
# -------------------------

y = torch.tensor([30.0])


# -------------------------
# 4. Forward pass
# -------------------------

z1 = W1 @ x + b1

a1 = torch.relu(z1)

y_hat = W2 @ a1 + b2


# -------------------------
# 5. Loss
# -------------------------

loss = (y_hat - y) ** 2


print("Prediction:", y_hat)
print("Loss:", loss)


# -------------------------
# 6. Backpropagation
# -------------------------

loss.backward()


# -------------------------
# 7. Gradients
# -------------------------

print("\nW1 gradient:")
print(W1.grad)

print("\nb1 gradient:")
print(b1.grad)

print("\nW2 gradient:")
print(W2.grad)

print("\nb2 gradient:")
print(b2.grad)


# -------------------------
# 8. Update
# -------------------------

lr = 0.01

with torch.no_grad():
    W1 -= lr * W1.grad
    b1 -= lr * b1.grad
    W2 -= lr * W2.grad
    b2 -= lr * b2.grad


# -------------------------
# 9. Clear gradients
# -------------------------

W1.grad.zero_()
b1.grad.zero_()
W2.grad.zero_()
b2.grad.zero_()
