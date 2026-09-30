import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load dataset
data = load_digits()

images = data.images / data.images.max()
images = images[..., np.newaxis]
labels = data.target

# Use 500 images
images = images[:500]
labels = labels[:500]

# Split data
train_images, test_images, train_labels, test_labels = train_test_split(
    images,
    labels,
    test_size=0.2,
    random_state=42
)

# CNN
image_shape = train_images.shape[1:]

cnn = Sequential([
    tf.keras.Input(shape=image_shape),

    Conv2D(8, 3, padding="same", activation="relu"),
    MaxPooling2D(2),

    Conv2D(16, 3, padding="same", activation="relu"),
    MaxPooling2D(2),

    Flatten(),
    Dense(32, activation="relu"),
    Dense(10, activation="softmax")
])

# Compile
cnn.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train
history = cnn.fit(
    train_images,
    train_labels,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# Evaluate
loss, accuracy = cnn.evaluate(
    test_images,
    test_labels,
    verbose=0
)

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
