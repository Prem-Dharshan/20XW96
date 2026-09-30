import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# Variables
DATASET_PATH = "datasets/deep_learning_lab_test_dataset.csv"
TARGET = "target"

TEST_SIZE = 0.2
RANDOM_STATE = 42

NOISE_LEVEL = 0.2

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


# Load data
df = pd.read_csv(DATASET_PATH).dropna()

# Keep target
y = df[TARGET]

# Prepare data
X = pd.get_dummies(
    df.drop(columns=TARGET),
    dtype=float
)

X = StandardScaler().fit_transform(X)

# Encode target if categorical
y = pd.factorize(y)[0]


# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)


# Add noise
X_train_noisy = X_train + NOISE_LEVEL * np.random.normal(
    size=X_train.shape
)

X_test_noisy = X_test + NOISE_LEVEL * np.random.normal(
    size=X_test.shape
)


# Autoencoder
INPUT_SIZE = X_train.shape[1]

autoencoder = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(INPUT_SIZE,)),
    tf.keras.layers.Dense(ENCODER_UNITS[0], activation=ACTIVATION),
    tf.keras.layers.Dense(ENCODER_UNITS[1], activation=ACTIVATION),
    tf.keras.layers.Dense(DECODER_UNITS[0], activation=ACTIVATION),
    tf.keras.layers.Dense(INPUT_SIZE)
])


autoencoder.compile(
    optimizer=OPTIMIZER,
    loss=LOSS
)


# Noisy input -> clean output
autoencoder.fit(
    X_train_noisy,
    X_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT
)


# Encoder
encoder = tf.keras.Sequential(
    autoencoder.layers[:len(ENCODER_UNITS) + 1]
)


# Encode clean and noisy data
encoded_clean_train = encoder.predict(X_train, verbose=0)
encoded_noisy_train = encoder.predict(X_train_noisy, verbose=0)

encoded_clean_test = encoder.predict(X_test, verbose=0)
encoded_noisy_test = encoder.predict(X_test_noisy, verbose=0)


# Classification WITHOUT noise
classifier_clean = LogisticRegression()
classifier_clean.fit(encoded_clean_train, y_train)

pred_clean = classifier_clean.predict(encoded_clean_test)

print("\nWITHOUT NOISE")
print(classification_report(y_test, pred_clean))


# Classification WITH noise
classifier_noisy = LogisticRegression()
classifier_noisy.fit(encoded_noisy_train, y_train)

pred_noisy = classifier_noisy.predict(encoded_noisy_test)

print("\nWITH NOISE")
print(classification_report(y_test, pred_noisy))


# Denoise
denoised = autoencoder.predict(
    X_test_noisy,
    verbose=0
)


# Reconstruction error
error = np.mean(
    (X_test - denoised) ** 2,
    axis=1
)

print("Mean reconstruction error:", error.mean())


# Plot
plt.plot(error, "o-")
plt.xlabel("Test Sample")
plt.ylabel("Reconstruction Error")
plt.title("Denoising Autoencoder Reconstruction Error")
plt.grid()
plt.show()
