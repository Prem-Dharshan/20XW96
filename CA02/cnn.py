import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


# Variables
NUM_IMAGES = 500
TEST_SIZE = 0.2
RANDOM_STATE = 42

FILTERS = [8, 16]
KERNEL_SIZE = 3
POOL_SIZE = 2
DENSE_UNITS = 32

ACTIVATION = "relu"
# Alternatives: sigmoid, tanh, linear, elu, selu, gelu

OUTPUT_ACTIVATION = "softmax"
# Alternatives: sigmoid (binary classification), linear (regression)

OPTIMIZER = "adam"
# Alternatives: sgd, rmsprop, adamax, nadam

LOSS = "sparse_categorical_crossentropy"
# Alternatives: categorical_crossentropy, binary_crossentropy

EPOCHS = 10
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.1


# Load dataset
data = load_digits()

images = data.images / data.images.max()
images = images[..., np.newaxis]
labels = data.target

images = images[:NUM_IMAGES]
labels = labels[:NUM_IMAGES]


# Split data
train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)


# CNN
IMAGE_SHAPE = train_images.shape[1:]

cnn = Sequential([
    tf.keras.Input(shape=IMAGE_SHAPE),

    Conv2D(FILTERS[0], KERNEL_SIZE, padding="same", activation=ACTIVATION),
    MaxPooling2D(POOL_SIZE),

    Conv2D(FILTERS[1], KERNEL_SIZE, padding="same", activation=ACTIVATION),
    MaxPooling2D(POOL_SIZE),

    Flatten(),
    Dense(DENSE_UNITS, activation=ACTIVATION),
    Dense(10, activation=OUTPUT_ACTIVATION)
])


# Compile
cnn.compile(
    optimizer=OPTIMIZER,
    loss=LOSS,
    metrics=["accuracy"]
)


# Train
history = cnn.fit(
    train_images,
    train_labels,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT
)


# Evaluate
loss, accuracy = cnn.evaluate(test_images, test_labels, verbose=0)

predictions = cnn.predict(test_images, verbose=0)
predicted_labels = np.argmax(predictions, axis=1)


print("CNN completed.")
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(test_labels, predicted_labels))


# Plot error
plt.plot(history.history["loss"], label="Training Error")
plt.plot(history.history["val_loss"], label="Validation Error")

plt.xlabel("Epoch")
plt.ylabel("Error")
plt.title("CNN Error")
plt.legend()
plt.grid()
plt.show()
