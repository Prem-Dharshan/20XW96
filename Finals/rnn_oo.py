from pathlib import Path

from _runtime import setup

args = setup(__file__)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the Fashion-MNIST dataset
(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.fashion_mnist.load_data()
)

if args.quick:
    x_train = x_train[:256]
    y_train = y_train[:256]
    x_test = x_test[:64]
    y_test = y_test[:64]


class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

print("Training images:", x_train.shape)
print("Training labels:", y_train.shape)
print("Testing images:", x_test.shape)
print("Testing labels:", y_test.shape)

# 2. Visualize a sample image
index = 0

plt.imshow(x_train[index], cmap="gray")
plt.title(f"Actual class: {class_names[y_train[index]]}")
plt.axis("off")
plt.show()
plt.figure()

# 3. Normalize pixel values to the range [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 4. Flatten each 28 x 28 image into 784 features
x_train_flat = x_train.reshape(-1, 784)
x_test_flat = x_test.reshape(-1, 784)

# 5. Add a timestep dimension for the SimpleRNN layer
x_train_rnn = x_train_flat[:, np.newaxis, :]
x_test_rnn = x_test_flat[:, np.newaxis, :]

print("RNN training input shape:", x_train_rnn.shape)
print("RNN testing input shape:", x_test_rnn.shape)

# 6. Build the one-to-one classification model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(1, 784)),
    tf.keras.layers.SimpleRNN(units=128,activation="tanh"),
    tf.keras.layers.Dense(units=10,activation="softmax")
])

# 7. Display model architecture and parameter counts
model.summary()

# 8. Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 9. Train the model
history = model.fit(
    x_train_rnn,
    y_train,
    epochs=1 if args.quick else 5,
    batch_size=64,
    validation_split=0.1,
    shuffle=True
)

# 10. Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(
    x_test_rnn,
    y_test,
    verbose=2
)

print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# 11. Plot training and validation loss and accuracy
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Training loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Training accuracy")
plt.plot(history.history["val_accuracy"], label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()

plt.tight_layout()
plt.show()
plt.figure()

# 12. Predict the class of one test image
index = 0

sample = x_test_rnn[index:index + 1]
probabilities = model.predict(sample, verbose=0)

predicted_label = np.argmax(probabilities[0])
actual_label = y_test[index]

plt.imshow(x_test[index], cmap="gray")
plt.title(
    f"Actual: {class_names[actual_label]}\n"
    f"Predicted: {class_names[predicted_label]}"
)
plt.axis("off")
plt.show()
plt.figure()

print("Actual class:", class_names[actual_label])
print("Predicted class:", class_names[predicted_label])
print("Class probabilities:", np.round(probabilities[0], 3))

# 13. Predict a batch of test images
predictions = model.predict(x_test_rnn[:10], verbose=0)
predicted_labels = np.argmax(predictions, axis=1)

for i in range(10):
    print(
        f"Image {i}: Actual = {class_names[y_test[i]]}, "
        f"Predicted = {class_names[predicted_labels[i]]}"
    )
