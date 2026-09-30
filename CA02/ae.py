import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Variables
DATASET_PATH = "datasets/deep_learning_lab_test_dataset.csv"
TARGET = "target"

TEST_SIZE = 0.2
RANDOM_STATE = 42

ENCODER_UNITS = [8, 4]
DECODER_UNITS = [8]

ACTIVATION = "relu"
# Alternatives: sigmoid, tanh, linear, elu, selu, gelu

OPTIMIZER = "adam"
# Alternatives: sgd, rmsprop, adamax, nadam

LOSS = "mse"
# Alternatives: mae, binary_crossentropy, categorical_crossentropy

EPOCHS = 50
BATCH_SIZE = 16
VALIDATION_SPLIT = 0.2


# Load
df = pd.read_csv(DATASET_PATH).dropna()

X = pd.get_dummies(
    df.drop(columns=TARGET),
    dtype=float
)

X = StandardScaler().fit_transform(X)


# Split
X_train, X_test = train_test_split(
    X,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

INPUT_SIZE = X_train.shape[1]


# Autoencoder
autoencoder = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(INPUT_SIZE,)),

    tf.keras.layers.Dense(ENCODER_UNITS[0], activation=ACTIVATION),
    tf.keras.layers.Dense(ENCODER_UNITS[1], activation=ACTIVATION),

    tf.keras.layers.Dense(DECODER_UNITS[0], activation=ACTIVATION),

    # Output: linear activation by default
    tf.keras.layers.Dense(INPUT_SIZE)
])


autoencoder.compile(
    optimizer=OPTIMIZER,
    loss=LOSS
)


autoencoder.fit(
    X_train,
    X_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT
)


# Encoder
encoder = tf.keras.Sequential(
    autoencoder.layers[:len(ENCODER_UNITS) + 1]
)


# Reconstruction
reconstructed = autoencoder.predict(X_test, verbose=0)

# Encoding
encoded = encoder.predict(X_test, verbose=0)

# Reconstruction error
reconstruction_error = np.mean(
    (X_test - reconstructed) ** 2,
    axis=1
)


print("Test shape:", X_test.shape)
print("Encoded shape:", encoded.shape)
print("Mean error:", reconstruction_error.mean())


plt.plot(reconstruction_error, "o-")
plt.xlabel("Test Sample")
plt.ylabel("Reconstruction Error")
plt.title("Autoencoder Reconstruction Error")
plt.grid()
plt.show()
