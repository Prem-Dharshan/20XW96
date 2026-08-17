import numpy as np

from sklearn.model_selection import ParameterGrid
from sklearn.utils import gen_batches


# -------------------------
# 1. Model Training Function
# -------------------------

def train_model(
    X_train,
    y_train,
    lr,
    epochs,
    batch_size
):

    # -------------------------
    # Parameters
    # -------------------------

    N = X_train.shape[0]

    n_features = X_train.shape[1]

    n_outputs = y_train.shape[1]

    W = np.random.randn(
        n_features,
        n_outputs
    )

    b = np.zeros(
        (1, n_outputs)
    )


    # -------------------------
    # Training
    # -------------------------

    for epoch in range(epochs):

        # Shuffle
        indices = np.random.permutation(N)

        X_shuffled = X_train[indices]

        y_shuffled = y_train[indices]


        # Batches
        for batch in gen_batches(
            N,
            batch_size
        ):

            X_batch = X_shuffled[batch]

            y_batch = y_shuffled[batch]

            n_batch = len(X_batch)


            # -------------------------
            # Forward Pass
            # -------------------------

            Z = X_batch @ W + b

            y_pred = sigmoid(Z)


            # -------------------------
            # Backward Pass
            # -------------------------

            dZ = (
                y_pred - y_batch
            ) / n_batch

            dW = X_batch.T @ dZ

            db = np.sum(
                dZ,
                axis=0,
                keepdims=True
            )


            # -------------------------
            # Weight Update
            # -------------------------

            W -= lr * dW

            b -= lr * db


    return W, b


# -------------------------
# 2. Hyperparameter Grid
# -------------------------

param_grid = {

    "lr": [0.001, 0.01, 0.1],

    "epochs": [100, 500, 1000],

    "batch_size": [1, 2, 4]
}


# -------------------------
# 3. Grid Search
# -------------------------

best_loss = float("inf")

best_params = None

best_W = None

best_b = None


for params in ParameterGrid(
    param_grid
):

    # -------------------------
    # Train Model
    # -------------------------

    W, b = train_model(
        X_train,
        y_train,
        lr=params["lr"],
        epochs=params["epochs"],
        batch_size=params["batch_size"]
    )


    # -------------------------
    # Validation Prediction
    # -------------------------

    y_pred = sigmoid(
        X_test @ W + b
    )


    # -------------------------
    # Validation Loss
    # -------------------------

    loss = binary_cross_entropy(
        y_test,
        y_pred
    )


    # -------------------------
    # Store Best Model
    # -------------------------

    if loss < best_loss:

        best_loss = loss

        best_params = params.copy()

        best_W = W.copy()

        best_b = b.copy()


# -------------------------
# 4. Best Parameters
# -------------------------

print("Best Parameters:")

print(best_params)

print("\nBest Validation Loss:")

print(best_loss)


# -------------------------
# 5. Final Parameters
# -------------------------

W = best_W

b = best_b