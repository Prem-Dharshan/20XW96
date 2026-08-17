### MLP revision matrix

| Method                              | **1. Data / Batch**             | **2. Linear / Forward**      | **3. Activation**         | **4. Output**            | **5. Loss**                          | **6. Backward**           | **7. Parameter Update**    |
| ----------------------------------- | ------------------------------- | ---------------------------- | ------------------------- | ------------------------ | ------------------------------------ | ------------------------- | -------------------------- |
| **Linear + Batch GD**               | `python X, y `                  | `python y_pred = model(X) `  | —                         | `python # y_pred `       | `python loss = loss_fn(y_pred, y) `  | `python loss.backward() ` | `python optimizer.step() ` |
| **Linear + Mini-batch SGD**         | `python for Xb, yb in loader: ` | `python y_pred = model(Xb) ` | —                         | `python # y_pred `       | `python loss = loss_fn(y_pred, yb) ` | `python loss.backward() ` | `python optimizer.step() ` |
| **Nonlinear MLP + Batch GD**        | `python X, y `                  | `python y_pred = model(X) `  | `python torch.nn.ReLU() ` | `python # final Linear ` | `python loss = loss_fn(y_pred, y) `  | `python loss.backward() ` | `python optimizer.step() ` |
| **Nonlinear MLP + Mini-batch SGD**  | `python for Xb, yb in loader: ` | `python y_pred = model(Xb) ` | `python torch.nn.ReLU() ` | `python # final Linear ` | `python loss = loss_fn(y_pred, yb) ` | `python loss.backward() ` | `python optimizer.step() ` |
| **Binary MLP + Batch GD**           | `python X, y `                  | `python logits = model(X) `  | `python torch.nn.ReLU() ` | `python # 1 output `     | `python loss = loss_fn(logits, y) `  | `python loss.backward() ` | `python optimizer.step() ` |
| **Binary MLP + Mini-batch SGD**     | `python for Xb, yb in loader: ` | `python logits = model(Xb) ` | `python torch.nn.ReLU() ` | `python # 1 output `     | `python loss = loss_fn(logits, yb) ` | `python loss.backward() ` | `python optimizer.step() ` |
| **Multiclass MLP + Batch GD**       | `python X, y `                  | `python logits = model(X) `  | `python torch.nn.ReLU() ` | `python # C outputs `    | `python loss = loss_fn(logits, y) `  | `python loss.backward() ` | `python optimizer.step() ` |
| **Multiclass MLP + Mini-batch SGD** | `python for Xb, yb in loader: ` | `python logits = model(Xb) ` | `python torch.nn.ReLU() ` | `python # C outputs `    | `python loss = loss_fn(logits, yb) ` | `python loss.backward() ` | `python optimizer.step() ` |

---

# Functions / syntax you need to remember

## 1. Dataset

### Numeric tensor

```python
X = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0]
])

y = torch.tensor([
    [0.0],
    [1.0]
])
```

For multiclass:

```python
y = torch.tensor([0, 1, 2])
```

---

## 2. Mini-batches

### Create dataset

```python
dataset = TensorDataset(X, y)
```

### Create DataLoader

```python
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)
```

### Iterate

```python
for X_batch, y_batch in loader:
```

---

# 3. MLP

### Linear model

```python
model = torch.nn.Linear(
    input_features,
    output_features
)
```

Example:

```python
model = torch.nn.Linear(2, 1)
```

### Nonlinear MLP

```python
model = torch.nn.Sequential(
    torch.nn.Linear(2, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 1)
)
```

### Multiclass

```python
model = torch.nn.Sequential(
    torch.nn.Linear(2, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 3)
)
```

---

# 4. Activations

### ReLU

```python
torch.nn.ReLU()
```

Mathematically:

[
f(x)=\max(0,x)
]

### Sigmoid

```python
torch.nn.Sigmoid()
```

or:

```python
torch.sigmoid(x)
```

Used for binary probabilities.

### Softmax

```python
torch.softmax(x, dim=1)
```

Used to **view multiclass probabilities**.

Usually **don't put Softmax before `CrossEntropyLoss`**.

---

# 5. Forward pass

### Regression

```python
y_pred = model(X)
```

### Binary classification

With `BCEWithLogitsLoss`:

```python
logits = model(X)
```

### Multiclass classification

```python
logits = model(X)
```

---

# 6. Loss functions

### Regression — MSE

```python
loss_fn = torch.nn.MSELoss()
```

```python
loss = loss_fn(y_pred, y)
```

### Binary classification

Preferred:

```python
loss_fn = torch.nn.BCEWithLogitsLoss()
```

```python
loss = loss_fn(logits, y)
```

If explicitly using Sigmoid:

```python
loss_fn = torch.nn.BCELoss()
```

```python
prob = torch.sigmoid(logits)
loss = loss_fn(prob, y)
```

### Multiclass classification

```python
loss_fn = torch.nn.CrossEntropyLoss()
```

```python
loss = loss_fn(logits, y)
```

---

# 7. Backpropagation

Same for **all of them**:

```python
loss.backward()
```

This calculates:

[
\frac{\partial L}{\partial W}
]

and:

[
\frac{\partial L}{\partial b}
]

for all trainable parameters.

---

# 8. SGD optimizer

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)
```

Update:

```python
optimizer.step()
```

Clear gradients:

```python
optimizer.zero_grad()
```

---

# 9. The two training loops to memorize

## Batch Gradient Descent

Entire dataset is used:

```python
for epoch in range(epochs):

    y_pred = model(X)

    loss = loss_fn(y_pred, y)

    loss.backward()

    optimizer.step()

    optimizer.zero_grad()
```

### Key idea

[
\boxed{\text{1 update per epoch}}
]

---

## Mini-batch SGD

Dataset is divided into batches:

```python
for epoch in range(epochs):

    for X_batch, y_batch in loader:

        y_pred = model(X_batch)

        loss = loss_fn(y_pred, y_batch)

        loss.backward()

        optimizer.step()

        optimizer.zero_grad()
```

### Key idea

[
\boxed{\text{1 update per batch}}
]

---

# 10. Prediction functions

### Binary

```python
probabilities = torch.sigmoid(logits)

predictions = (probabilities >= 0.5).float()
```

If using a model that already contains Sigmoid:

```python
probabilities = model(X)

predictions = (probabilities >= 0.5).float()
```

### Multiclass

```python
probabilities = torch.softmax(
    logits,
    dim=1
)
```

Then:

```python
predictions = torch.argmax(
    probabilities,
    dim=1
)
```

Or directly from logits:

```python
predictions = torch.argmax(
    logits,
    dim=1
)
```

---

# Final revision flow

This is probably the **most important table to memorize**:

| Task                          | Output | Activation during training | Loss                  |
| ----------------------------- | -----: | -------------------------- | --------------------- |
| **Linear regression**         |      1 | None                       | `MSELoss()`           |
| **Nonlinear regression**      |      1 | ReLU in hidden layers      | `MSELoss()`           |
| **Binary classification**     |      1 | Sigmoid conceptually       | `BCEWithLogitsLoss()` |
| **Multiclass classification** |      C | Softmax conceptually       | `CrossEntropyLoss()`  |

And the training mechanism is independent:

| Training           |  Batch size | Update          |
| ------------------ | ----------: | --------------- |
| **Batch GD**       |         `N` | Once per epoch  |
| **SGD**            |         `1` | Once per sample |
| **Mini-batch SGD** | `1 < B < N` | Once per batch  |

So the complete mental model is:

```text
DATA
 ↓
BATCHING
 ↓
FORWARD
 ↓
Linear
 ↓
Activation
 ↓
Linear
 ↓
OUTPUT
 ↓
LOSS
 ↓
BACKWARD
 ↓
GRADIENTS
 ↓
OPTIMIZER.STEP()
 ↓
ZERO_GRAD()
 ↓
NEXT BATCH
```

**One thing to keep especially clear:** `torch.optim.SGD` is the **optimizer**. Whether you're doing batch GD, pure SGD, or mini-batch SGD is determined by **how many samples you use to calculate each gradient/update**.
