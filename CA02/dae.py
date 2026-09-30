import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("datasets/deep_learning_lab_test_dataset.csv")
df = df.dropna()

# Prepare data
X = pd.get_dummies(
    df.drop(columns="target"),
    dtype=float
)

X = StandardScaler().fit_transform(X)

# Split
X_train, X_test = train_test_split(
    X, test_size=0.2, random_state=42
)

# Add noise
X_train_noisy = X_train + 0.2 * np.random.normal(
    size=X_train.shape
)

X_test_noisy = X_test + 0.2 * np.random.normal(
    size=X_test.shape
)

# Autoencoder
autoencoder = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(X_train.shape[1])
])

# Compile
autoencoder.compile(
    optimizer="adam",
    loss="mse"
)

# Train: noisy input → clean output
autoencoder.fit(
    X_train_noisy,
    X_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

# Extract encoder
encoder = tf.keras.Sequential(
    autoencoder.layers[:3]
)

# Test
denoised = autoencoder.predict(
    X_test_noisy, verbose=0
)

encoded = encoder.predict(
    X_test_noisy, verbose=0
)

# Reconstruction error
error = np.mean(
    (X_test - denoised) ** 2,
    axis=1
)

print("Denoising Autoencoder completed.")
print("Test shape:", X_test.shape)
print("Encoded shape:", encoded.shape)
print("Mean reconstruction error:", error.mean())

# Plot
plt.plot(error, "o-")
plt.xlabel("Test Sample")
plt.ylabel("Reconstruction Error")
plt.title("Denoising Autoencoder Reconstruction Error")
plt.grid()
plt.show()
