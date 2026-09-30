import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("datasets/deep_learning_lab_test_dataset.csv")

# Remove missing values
df = df.dropna()

# Remove target and encode categorical features
X = pd.get_dummies(
    df.drop(columns="target"),
    dtype=float
)

# Standardize features
X = StandardScaler().fit_transform(X)

# Split into training and testing data
X_train, X_test = train_test_split(
    X, test_size=0.2, random_state=42
)

# Build autoencoder
autoencoder = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),  # Input
    tf.keras.layers.Dense(8, activation="relu"),       # Encoder
    tf.keras.layers.Dense(4, activation="relu"),       # Bottleneck
    tf.keras.layers.Dense(8, activation="relu"),       # Decoder
    tf.keras.layers.Dense(X_train.shape[1])            # Reconstruction
])

# Compile model
autoencoder.compile(
    optimizer="adam",
    loss="mse"
)

# Train autoencoder
autoencoder.fit(
    X_train,
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

# Reconstruct test data
reconstructed = autoencoder.predict(
    X_test, verbose=0
)

# Encode test data
encoded = encoder.predict(
    X_test, verbose=0
)

# Calculate reconstruction error for each sample
reconstruction_error = np.mean(
    (X_test - reconstructed) ** 2,
    axis=1
)

print("Autoencoder completed.")
print("Test shape:", X_test.shape)
print("Encoded shape:", encoded.shape)
print(
    "Mean reconstruction error:",
    reconstruction_error.mean()
)

# Plot reconstruction error
plt.plot(reconstruction_error, "o-")
plt.xlabel("Test Sample")
plt.ylabel("Reconstruction Error")
plt.title("Autoencoder Reconstruction Error")
plt.grid()
plt.show()
