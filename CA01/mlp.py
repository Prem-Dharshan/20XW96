import torch
from torch.utils.data import DataLoader, TensorDataset


def train_mlp(
    X, y, task="linear", method="mini", batch_size=32, hidden=10, lr=0.01, epochs=1000
):

    # =========================================================
    # 1. Determine output size
    # =========================================================

    if task == "linear":
        output_size = 1

    elif task == "nonlinear":
        output_size = 1

    elif task == "binary":
        output_size = 1

    elif task == "multiway":
        output_size = len(torch.unique(y))

    else:
        raise ValueError("task must be: linear, nonlinear, binary, multiway")

    # =========================================================
    # 2. Create model
    # =========================================================

    input_size = X.shape[1]

    if task == "linear":

        model = torch.nn.Sequential(torch.nn.Linear(input_size, 1))

    else:

        model = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, output_size),
        )

    # =========================================================
    # 3. Loss function
    # =========================================================

    if task == "linear" or task == "nonlinear":

        loss_fn = torch.nn.MSELoss()

    elif task == "binary":

        loss_fn = torch.nn.BCEWithLogitsLoss()

    elif task == "multiway":

        loss_fn = torch.nn.CrossEntropyLoss()

    # =========================================================
    # 4. Optimizer
    # =========================================================

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # =========================================================
    # 5. Determine batch size
    # =========================================================

    sample_size = X.shape[0]

    if method == "sgd":

        batch_size = 1

    elif method == "batch":

        batch_size = sample_size

    elif method == "mini":

        if batch_size >= sample_size:

            raise ValueError(
                "For mini-batch, batch_size must be smaller "
                "than the number of samples."
            )

    else:

        raise ValueError("method must be: batch, sgd, mini")

    # =========================================================
    # 6. Create DataLoader
    # =========================================================

    dataset = TensorDataset(X, y)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # =========================================================
    # 7. Training
    # =========================================================

    for epoch in range(epochs):

        for X_batch, y_batch in loader:

            # -------------------------------------------------
            # Forward
            # -------------------------------------------------

            output = model(X_batch)

            # -------------------------------------------------
            # Prepare output / target
            # -------------------------------------------------

            if task == "binary":

                output = output.squeeze(1)

                y_batch = y_batch.float().squeeze(1)

            elif task == "multiway":

                y_batch = y_batch.long()

            # -------------------------------------------------
            # Loss
            # -------------------------------------------------

            loss = loss_fn(output, y_batch)

            # -------------------------------------------------
            # Backward
            # -------------------------------------------------

            loss.backward()

            # -------------------------------------------------
            # Update
            # -------------------------------------------------

            optimizer.step()

            # -------------------------------------------------
            # Clear gradients
            # -------------------------------------------------

            optimizer.zero_grad()

    return model


# =============================================================
# 1. LINEAR REGRESSION
# =============================================================

print("\n==============================")
print("1. LINEAR REGRESSION")
print("==============================")


X = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])

y = torch.tensor([[3.0], [5.0], [7.0], [9.0], [11.0]])


# ---------- Batch ----------

model = train_mlp(X, y, task="linear", method="batch", lr=0.01, epochs=1000)

with torch.no_grad():

    predictions = model(X)

print("\nLinear + Batch")
print("Predictions:")
print(predictions)
print("Actual:")
print(y)


# ---------- SGD ----------

model = train_mlp(X, y, task="linear", method="sgd", lr=0.001, epochs=1000)

with torch.no_grad():

    predictions = model(X)

print("\nLinear + SGD")
print("Predictions:")
print(predictions)
print("Actual:")
print(y)


# ---------- Mini-batch ----------

model = train_mlp(
    X, y, task="linear", method="mini", batch_size=2, lr=0.01, epochs=1000
)

with torch.no_grad():

    predictions = model(X)

print("\nLinear + Mini-batch SGD")
print("Predictions:")
print(predictions)
print("Actual:")
print(y)


# =============================================================
# 2. NONLINEAR REGRESSION
# =============================================================

print("\n==============================")
print("2. NONLINEAR REGRESSION")
print("==============================")


X = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]])

y = torch.tensor([[1.0], [4.0], [9.0], [16.0], [25.0], [36.0], [49.0], [64.0]])


# ---------- Batch ----------

model = train_mlp(
    X, y, task="nonlinear", method="batch", hidden=20, lr=0.0005, epochs=5000
)

with torch.no_grad():

    predictions = model(X)

print("\nNonlinear + Batch")
print("Predictions:")
print(predictions)
print("Actual:")
print(y)


# ---------- SGD ----------

model = train_mlp(
    X, y, task="nonlinear", method="sgd", hidden=20, lr=0.0001, epochs=5000
)

with torch.no_grad():

    predictions = model(X)

print("\nNonlinear + SGD")
print("Predictions:")
print(predictions)
print("Actual:")
print(y)


# ---------- Mini-batch ----------

model = train_mlp(
    X,
    y,
    task="nonlinear",
    method="mini",
    batch_size=2,
    hidden=20,
    lr=0.0005,
    epochs=5000,
)

with torch.no_grad():

    predictions = model(X)

print("\nNonlinear + Mini-batch SGD")
print("Predictions:")
print(predictions)
print("Actual:")
print(y)


# =============================================================
# 3. BINARY CLASSIFICATION
# =============================================================

print("\n==============================")
print("3. BINARY CLASSIFICATION")
print("==============================")


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


# ---------- Batch ----------

model = train_mlp(X, y, task="binary", method="batch", hidden=10, lr=0.1, epochs=1000)

with torch.no_grad():

    logits = model(X)

    probabilities = torch.sigmoid(logits)

    predictions = (probabilities >= 0.5).float()

print("\nBinary + Batch")
print("Probabilities:")
print(probabilities)

print("Predictions:")
print(predictions)

print("Actual:")
print(y)


# ---------- SGD ----------

model = train_mlp(X, y, task="binary", method="sgd", hidden=10, lr=0.01, epochs=1000)

with torch.no_grad():

    logits = model(X)

    probabilities = torch.sigmoid(logits)

    predictions = (probabilities >= 0.5).float()

print("\nBinary + SGD")
print("Probabilities:")
print(probabilities)

print("Predictions:")
print(predictions)

print("Actual:")
print(y)


# ---------- Mini-batch ----------

model = train_mlp(
    X, y, task="binary", method="mini", batch_size=2, hidden=10, lr=0.1, epochs=1000
)

with torch.no_grad():

    logits = model(X)

    probabilities = torch.sigmoid(logits)

    predictions = (probabilities >= 0.5).float()

print("\nBinary + Mini-batch SGD")
print("Probabilities:")
print(probabilities)

print("Predictions:")
print(predictions)

print("Actual:")
print(y)


# =============================================================
# 4. MULTIWAY CLASSIFICATION
# =============================================================

print("\n==============================")
print("4. MULTIWAY CLASSIFICATION")
print("==============================")


X = torch.tensor(
    [
        # Class 0
        [1.0, 1.0],
        [1.0, 2.0],
        [2.0, 1.0],
        # Class 1
        [5.0, 5.0],
        [5.0, 6.0],
        [6.0, 5.0],
        # Class 2
        [9.0, 1.0],
        [9.0, 2.0],
        [8.0, 1.0],
    ]
)

y = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])


# ---------- Batch ----------

model = train_mlp(
    X, y, task="multiway", method="batch", hidden=10, lr=0.01, epochs=1000
)

with torch.no_grad():

    logits = model(X)

    probabilities = torch.softmax(logits, dim=1)

    predictions = torch.argmax(probabilities, dim=1)

print("\nMultiway + Batch")
print("Probabilities:")
print(probabilities)

print("Predictions:")
print(predictions)

print("Actual:")
print(y)


# ---------- SGD ----------

model = train_mlp(X, y, task="multiway", method="sgd", hidden=10, lr=0.01, epochs=1000)

with torch.no_grad():

    logits = model(X)

    probabilities = torch.softmax(logits, dim=1)

    predictions = torch.argmax(probabilities, dim=1)

print("\nMultiway + SGD")
print("Probabilities:")
print(probabilities)

print("Predictions:")
print(predictions)

print("Actual:")
print(y)


# ---------- Mini-batch ----------

model = train_mlp(
    X, y, task="multiway", method="mini", batch_size=3, hidden=10, lr=0.01, epochs=1000
)

with torch.no_grad():

    logits = model(X)

    probabilities = torch.softmax(logits, dim=1)

    predictions = torch.argmax(probabilities, dim=1)

print("\nMultiway + Mini-batch SGD")
print("Probabilities:")
print(probabilities)

print("Predictions:")
print(predictions)

print("Actual:")
print(y)
