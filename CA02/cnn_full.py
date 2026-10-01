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
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization


# ================= VARIABLES =================

DATA_PATH = "dataset"                 # Folder or CSV
IMAGE_SIZE = (64, 64)

TEST_SIZE = 0.2
RANDOM_STATE = 42

FILTERS = [8, 16]                     # Any number of Conv layers
KERNEL_SIZE = 3                       # Alternatives: 3, 5, 7
POOL_SIZE = 2                         # None = no pooling

DENSE_UNITS = [32]                    # Example: [128, 64]

ACTIVATION = "relu"
# Alternatives: sigmoid, tanh, elu, selu, gelu

OUTPUT_ACTIVATION = "softmax"
# Binary: sigmoid
# Regression: linear

OPTIMIZER = "adam"
# Alternatives: sgd, rmsprop, adamax, nadam

LOSS = "sparse_categorical_crossentropy"
# Multiclass integer labels: sparse_categorical_crossentropy
# Multiclass one-hot labels: categorical_crossentropy
# Binary: binary_crossentropy
# Regression: mse / mae

EPOCHS = 10
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.1

USE_DROPOUT = False
DROPOUT_RATE = 0.5

USE_BATCH_NORMALIZATION = False


# ================= LOAD DATA =================

def load_data(path):
    images, labels = [], []

    # CSV: label, filepath
    if path.endswith(".csv"):

        data = pd.read_csv(path)

        for _, row in data.iterrows():
            image = cv2.imread(
                row["filepath"],
                cv2.IMREAD_GRAYSCALE
            )

            if image is not None:
                images.append(cv2.resize(image, IMAGE_SIZE))
                labels.append(row["label"])

    # Folder: dataset/class/image.jpg
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


# If filename contains class instead:
# labels = np.array([
#     filename.split("_")[0]
#     for filename in os.listdir(DATA_PATH)
# ])


# RGB instead of grayscale:
# image = cv2.imread(path)
# Remove images[..., np.newaxis] below.


# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

images = images[..., np.newaxis]


# ================= SPLIT =================

train_images, test_images, train_labels, test_labels = train_test_split(
    images,
    labels,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=labels
)


# ================= CNN =================

layers = [
    tf.keras.Input(shape=train_images.shape[1:])
]


# Dynamic Conv layers
for filters in FILTERS:

    layers.append(
        Conv2D(
            filters,
            KERNEL_SIZE,
            padding="same",
            activation=ACTIVATION
        )
    )

    if USE_BATCH_NORMALIZATION:
        layers.append(BatchNormalization())

    if POOL_SIZE:
        layers.append(MaxPooling2D(POOL_SIZE))


layers.append(Flatten())


# Dynamic Dense layers
for units in DENSE_UNITS:

    layers.append(
        Dense(units, activation=ACTIVATION)
    )

    if USE_DROPOUT:
        layers.append(Dropout(DROPOUT_RATE))


# Output
NUM_CLASSES = len(np.unique(labels))

layers.append(
    Dense(
        NUM_CLASSES,
        activation=OUTPUT_ACTIVATION
    )
)

cnn = Sequential(layers)


# Binary classification:
# OUTPUT_ACTIVATION = "sigmoid"
# LOSS = "binary_crossentropy"
# Dense(1, activation="sigmoid")


# ================= COMPILE =================

cnn.compile(
    optimizer=OPTIMIZER,
    loss=LOSS,
    metrics=["accuracy"]
)


# ================= TRAIN =================

history = cnn.fit(
    train_images,
    train_labels,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT
)


# ================= EVALUATE =================

loss, accuracy = cnn.evaluate(
    test_images,
    test_labels,
    verbose=0
)

predictions = cnn.predict(
    test_images,
    verbose=0
)

predicted_labels = np.argmax(
    predictions,
    axis=1
)

print("Test Accuracy:", accuracy)
print(classification_report(
    test_labels,
    predicted_labels
))


# ================= SAMPLE PREDICTIONS =================

plt.figure(figsize=(10, 6))

for i in range(min(9, len(test_images))):

    plt.subplot(3, 3, i + 1)

    plt.imshow(
        test_images[i].squeeze(),
        cmap="gray"
    )

    true_label = label_encoder.inverse_transform(
        [test_labels[i]]
    )[0]

    predicted_label = label_encoder.inverse_transform(
        [predicted_labels[i]]
    )[0]

    plt.title(
        f"True: {true_label} | Pred: {predicted_label}"
    )

    plt.axis("off")

plt.tight_layout()
plt.show()


# ================= ACCURACY =================

plt.plot(
    history.history["accuracy"],
    label="Training"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()


# ================= LOSS =================

plt.plot(
    history.history["loss"],
    label="Training"
)

plt.plot(
    history.history["val_loss"],
    label="Validation"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()


# ================= CONFUSION MATRIX =================

# from sklearn.metrics import ConfusionMatrixDisplay
#
# ConfusionMatrixDisplay.from_predictions(
#     test_labels,
#     predicted_labels
# )
#
# plt.show()