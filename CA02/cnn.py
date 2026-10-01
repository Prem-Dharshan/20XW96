import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


# Variables
DATA_PATH = "dataset"       # folder or CSV
IMAGE_SIZE = (64, 64)

TEST_SIZE = 0.2
RANDOM_STATE = 42

FILTERS = [8, 16]
KERNEL_SIZE = 3
POOL_SIZE = 2
DENSE_UNITS = 32

ACTIVATION = "relu"
OUTPUT_ACTIVATION = "softmax"
OPTIMIZER = "adam"
LOSS = "sparse_categorical_crossentropy"
EPOCHS = 10
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.1


# Load images
def load_data(path):
    images, labels = [], []

    if path.endswith(".csv"):
        data = pd.read_csv(path)

        for _, row in data.iterrows():
            image = cv2.imread(row["filepath"], cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, IMAGE_SIZE)
            images.append(image)
            labels.append(row["label"])

    else:
        for class_name in os.listdir(path):
            class_path = os.path.join(path, class_name)

            if not os.path.isdir(class_path):
                continue

            for filename in os.listdir(class_path):
                image = cv2.imread(
                    os.path.join(class_path, filename),
                    cv2.IMREAD_GRAYSCALE
                )

                if image is not None:
                    images.append(cv2.resize(image, IMAGE_SIZE))
                    labels.append(class_name)

    return np.array(images) / 255.0, np.array(labels)


images, labels = load_data(DATA_PATH)

# If filenames contain the class
# if not os.path.isdir(DATA_PATH) and not DATA_PATH.endswith(".csv"):
#     labels = np.array([
#         filename.split("_")[0]
#         for filename in os.listdir(DATA_PATH)
#     ])

labels = LabelEncoder().fit_transform(labels)
images = images[..., np.newaxis]


# Split
train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=labels
)


# CNN
cnn = Sequential([
    tf.keras.Input(shape=train_images.shape[1:]),

    Conv2D(FILTERS[0], KERNEL_SIZE, padding="same", activation=ACTIVATION),
    MaxPooling2D(POOL_SIZE),

    Conv2D(FILTERS[1], KERNEL_SIZE, padding="same", activation=ACTIVATION),
    MaxPooling2D(POOL_SIZE),

    Flatten(),
    Dense(DENSE_UNITS, activation=ACTIVATION),
    Dense(len(np.unique(labels)), activation=OUTPUT_ACTIVATION)
])


cnn.compile(
    optimizer=OPTIMIZER,
    loss=LOSS,
    metrics=["accuracy"]
)


history = cnn.fit(
    train_images,
    train_labels,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT
)

# Evaluate
loss, accuracy = cnn.evaluate(test_images, test_labels, verbose=0)

predicted_labels = np.argmax(
    cnn.predict(test_images, verbose=0),
    axis=1
)

print("Test Accuracy:", accuracy)
print(classification_report(test_labels, predicted_labels))


# Sample predictions
plt.figure(figsize=(10, 6))

for i in range(min(9, len(test_images))):
    plt.subplot(3, 3, i + 1)
    plt.imshow(test_images[i].squeeze(), cmap="gray")
    plt.title(f"True: {test_labels[i]} | Pred: {predicted_labels[i]}")
    plt.axis("off")

plt.tight_layout()
plt.show()


# Accuracy
plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()


# Loss
plt.plot(history.history["loss"], label="Training")
plt.plot(history.history["val_loss"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()

