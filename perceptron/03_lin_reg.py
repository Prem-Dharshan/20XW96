"""
LINEAR REGRESSION FROM SCRATCH  --  revision sheet

The whole thing is 4 moves, repeated:
    1. FORWARD    guess          ->  y_pred = X @ w + b
    2. LOSS       score the guess->  how wrong, as ONE number
    3. BACKWARD   assign blame   ->  which way does each w push the loss?
    4. UPDATE     step downhill  ->  w -= lr * grad

Loop 1-4 enough times, w stops being garbage.
"""

import numpy as np

# ======================================================================
# DATA
# ======================================================================
# X: rows = samples, cols = features.  ALWAYS this way round.
#    Shape (N, n_features) = (3, 3) here. Coincidence that both are 3.
#    Don't let that confuse you later.
inputs = np.array([[1, 2, 4],
                   [2, 3, 5],
                   [3, 4, 6]])

# One weight PER FEATURE -> shape (n_features,) = (3,)
# NOTE the float. If you write np.array([1,1,1]) you get an int array,
# and `weights -= lr * grad` CRASHES (can't cast float into int slot).
weights = np.array([1.0, 1.0, 1.0])

# Bias is ONE number, not one per sample.
# Broadcasting hands the same b to every sample for free.
# (Your version used np.array([0.5,0.5,0.5]) -- gives the same answer
#  by luck, but it's wrong: 3 samples today, 500 tomorrow, then it breaks.)
bias = 0.5

y_true = np.array([2.5, 3.5, 4.5])


# ======================================================================
# 1. FORWARD
# ======================================================================
def lin_reg_activation(x):
    """Identity. Linear regression has NO activation.

    This function exists as a placeholder so you can see the slot.
    Swap identity -> sigmoid  = logistic regression.
    Swap identity -> step     = perceptron.
    Same skeleton. Only this line changes.
    """
    return x


def predict(X, w, b):
    """y_pred = X @ w + b

    Shapes:   (N, f) @ (f,) -> (N,)      then + scalar broadcasts
    Result: one prediction per sample.

    If your output is a single number, you collapsed too far --
    you used np.sum() with no axis. Classic bug.
    """
    z = X @ w + b
    return lin_reg_activation(z)


# ======================================================================
# 2. LOSS  (minimise these)  and  METRICS (just report these)
# ======================================================================
# Why square the error at all?
#   (a) kills signs, so +3 and -3 don't cancel to 0
#   (b) PUNISHES BIG ERRORS HARDER. err 10 costs 100x what err 1 costs.
#       => model is pushed to fix outliers first. This is the exam answer.

def sse_loss(y_true, y_pred):
    """Sum of Squared Errors. Grows as you add data -> can't compare
    across datasets of different size."""
    return np.sum((y_true - y_pred) ** 2)


def mse_loss(y_true, y_pred):
    """SSE / N. Size-independent. This is what you differentiate."""
    return np.mean((y_true - y_pred) ** 2)


def rmse_loss(y_true, y_pred):
    """sqrt(MSE). Same UNITS as y -- that's the only reason it exists.
    If y is in rupees, RMSE is in rupees. MSE is in rupees-squared (meaningless)."""
    return np.sqrt(mse_loss(y_true, y_pred))


def r_squared(y_true, y_pred):
    """SCORE, not a loss. Never differentiated.

        R2 = 1 - (your error / lazy model's error)

    Lazy model = always predicts mean(y), ignores X completely.

        R2 = 1    perfect
        R2 = 0.75 cut the lazy model's error by 75%
        R2 = 0    no better than guessing the mean
        R2 < 0    WORSE than guessing the mean  (no floor, goes to -inf)

    Welded to squared error -- SS_res IS the SSE. There is no "R2 for MAE".
    Regression only. Never for classification.
    """
    ss_res = sse_loss(y_true, y_pred)
    ss_tot = sse_loss(y_true, np.mean(y_true))   # mean broadcasts to (N,)
    return 1 - (ss_res / ss_tot)


# ======================================================================
# 3. BACKWARD  -- the only real content
# ======================================================================
def gradients(X, y_true, y_pred):
    """How does the loss move if I nudge each weight?

    Derivation (memorise the result, understand the shapes):
        y_pred = w1*x1 + w2*x2 + w3*x3 + b
        L      = (1/N) * sum( (y_pred - y_true)^2 )

        dL/dy_pred = (2/N) * (y_pred - y_true)      <- power rule
        dy_pred/dw1 = x1                            <- everything else is constant
        dL/dw1      = (2/N) * error * x1            <- multiply (chain rule)

    ORDER MATTERS: error = pred - true.
    Flip it and both gradients flip sign, and your update walks UPHILL.
    """
    N = len(y_true)
    error = y_pred - y_true                  # (N,)

    # Why X.T and not X?
    #   You want ONE gradient PER FEATURE, not per sample.
    #   X   is (N, f).  X.T is (f, N).  (f,N) @ (N,) -> (f,)  <- matches w. correct.
    #   X @ error would be (N,f)@(N,) -> shape error / garbage.
    #
    # Intuition: X.T @ error pairs each FEATURE COLUMN with the error vector.
    #   big x1 -> feature 1 had a big say in the prediction -> big correction
    #   small x1 -> it barely mattered -> barely moves
    #   This is BLAME ASSIGNMENT BY FEATURE.
    grad_w = (2 / N) * (X.T @ error)         # (f,)

    # Bias touches EVERY sample equally (its "x" is always 1),
    # so every sample's error gets a vote -> just sum them.
    # Must be a scalar, because b is a scalar.
    grad_b = (2 / N) * np.sum(error)         # scalar

    return grad_w, grad_b


# ======================================================================
# 4. UPDATE  -- YOUR TURN. Do not read past the hints.
# ======================================================================
def train(X, y_true, w, b, lr=0.01, epochs=100, convergence_tol=1e-6):
    """
    THE BLIND MAN ON A FOGGY HILL.
    Can't see the valley. But you can feel the slope under your foot.
    Step downhill. Feel again. Step. 100 times -> you're at the bottom.

    Gradient = the slope. It points UPHILL.
    You want DOWN. Hence the MINUS:

        w = w - lr * grad_w

        slope positive -> minus a positive -> w shrinks
        slope negative -> minus a negative -> w grows
        one formula, both directions, automatically.

    Steepness is free step-sizing: far from the bottom = steep = big step.
    Near the bottom = flat = tiny step. Auto-braking.

    lr (learning rate) = how bold each step is.
        too small -> crawls, never arrives in 100 epochs
        too big   -> overshoots the valley, bounces, loss EXPLODES to nan
    """
    history = []

    for i in range(epochs):
        # TODO 1: forward pass -> y_pred
        # TODO 2: record mse_loss into history
        # TODO 3: gw, gb = gradients(...)
        # TODO 4: update w and b   (remember the minus)
        # TODO 5: every 10 epochs, print epoch + loss
        
        y_pred = predict(X, w, b)  # 1: forward pass

        loss = mse_loss(y_true, y_pred)  # 2: record loss
        history.append(loss)  # store loss in history

        gw, gb = gradients(X, y_true, y_pred)  # 3: compute gradients

        w -= lr * gw  # 4: update weights
        b -= lr * gb  # 4: update bias

        if i % 10 == 0:  # 5: print every 10 epochs
            print(f"Epoch {i}: Loss = {loss:.6f}")
            print(f"    w = {w}, b = {b:.4f}")

        if len(history) > 1 and abs(history[-1] - history[-2]) < convergence_tol:
            print(f"Converged at epoch {i}.")
            break

    return w, b, history


# ======================================================================
# RUN
# ======================================================================
if __name__ == "__main__":
    y_pred = predict(inputs, weights, bias)

    print(f"True:  {y_true}")
    print(f"Pred:  {y_pred}")
    print(f"SSE:   {sse_loss(y_true, y_pred):.4f}")
    print(f"MSE:   {mse_loss(y_true, y_pred):.4f}")
    print(f"RMSE:  {rmse_loss(y_true, y_pred):.4f}")
    print(f"R^2:   {r_squared(y_true, y_pred):.4f}")

    gw, gb = gradients(inputs, y_true, y_pred)
    print(f"\ngrad_w: {gw}   shape {gw.shape}  <- must match weights.shape")
    print(f"grad_b: {gb:.4f}")

    w, b, hist = train(inputs, y_true, weights.copy(), bias)
    print(f"\nloss  first -> last : {hist[0]:.4f} -> {hist[-1]:.4f}")
    print(f"learned w = {w}, b = {b:.4f}")
    print(f"final R^2 = {r_squared(y_true, predict(inputs, w, b)):.4f}")
