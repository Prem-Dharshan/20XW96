
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Configuration
DATASET_PATH = "datasets/images"
LABEL_SOURCE = "subdir"       # "subdir", "filename", "csv"
CSV_PATH = "datasets/labels.csv"
CSV_IMAGE_COLUMN = "image_path"
CSV_LABEL_COLUMN = "label"
FILENAME_SEPARATOR = "_"       # cat_001.jpg -> label: cat

IMAGE_SIZE = 28
IMAGE_CHANNELS = 1             # 1 = grayscale, 3 = RGB
LATENT_DIM = 100
GENERATOR_UNITS = [256, 512]
DISCRIMINATOR_UNITS = [512, 256]

EPOCHS = 5
BATCH_SIZE = 128
LEARNING_RATE = 0.0002
BETA_1 = 0.5
NUM_GENERATED_IMAGES = 16
RANDOM_STATE = 42

# Supported image formats
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

tf.keras.utils.set_random_seed(RANDOM_STATE)

# Load image paths and labels
def load_image_paths():
    root = Path(DATASET_PATH)
    paths, labels = [], []

    if LABEL_SOURCE == "subdir":
        for path in root.rglob("*"):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(str(path))
                labels.append(path.parent.name)

    elif LABEL_SOURCE == "filename":
        for path in root.rglob("*"):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(str(path))
                labels.append(path.stem.rsplit(FILENAME_SEPARATOR, 1)[0])

    elif LABEL_SOURCE == "csv":
        with open(CSV_PATH, newline="", encoding="utf-8-sig") as file:
            for row in csv.DictReader(file):
                path = Path(row[CSV_IMAGE_COLUMN])
                if not path.is_absolute():
                    path = root / path
                paths.append(str(path))
                labels.append(row[CSV_LABEL_COLUMN])

    else:
        raise ValueError("LABEL_SOURCE must be subdir, filename, or csv")

    if not paths:
        raise ValueError("No images found. Check the dataset path and format.")

    if not all(Path(p).is_file() for p in paths):
        raise FileNotFoundError("Some image paths do not exist.")

    print("Images:", len(paths))
    print("Labels:", len(set(labels)))
    return paths


paths = load_image_paths()

# Preprocess images
def preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(
        image, channels=IMAGE_CHANNELS, expand_animations=False
    )
    image.set_shape([None, None, IMAGE_CHANNELS])
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.cast(image, tf.float32)
    return (image / 127.5) - 1.0


dataset = (
    tf.data.Dataset.from_tensor_slices(paths)
    .shuffle(len(paths), seed=RANDOM_STATE)
    .map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.AUTOTUNE)
)

if len(paths) < BATCH_SIZE:
    raise ValueError("Dataset must contain at least BATCH_SIZE images.")

# Generator
generator = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(LATENT_DIM,)),
    tf.keras.layers.Dense(GENERATOR_UNITS[0], activation="relu"),
    tf.keras.layers.Dense(GENERATOR_UNITS[1], activation="relu"),
    tf.keras.layers.Dense(
        IMAGE_SIZE * IMAGE_SIZE * IMAGE_CHANNELS, activation="tanh"
    ),
    tf.keras.layers.Reshape((IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS))
], name="generator")

# Discriminator
discriminator = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)),
    tf.keras.layers.Flatten(),
    *[
        layer
        for units in DISCRIMINATOR_UNITS
        for layer in (
            tf.keras.layers.Dense(units),
            tf.keras.layers.LeakyReLU(negative_slope=0.2)
        )
    ],
    tf.keras.layers.Dense(1)
], name="discriminator")

# Loss and optimizers
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

g_optimizer = tf.keras.optimizers.Adam(
    LEARNING_RATE, beta_1=BETA_1
)
d_optimizer = tf.keras.optimizers.Adam(
    LEARNING_RATE, beta_1=BETA_1
)

# Training step
@tf.function
def train_step(real_images):
    noise = tf.random.normal([tf.shape(real_images)[0], LATENT_DIM])

    with tf.GradientTape() as gt, tf.GradientTape() as dt:
        fake_images = generator(noise, training=True)

        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(fake_images, training=True)

        g_loss = loss_fn(tf.ones_like(fake_output), fake_output)
        d_loss = (
            loss_fn(tf.ones_like(real_output), real_output)
            + loss_fn(tf.zeros_like(fake_output), fake_output)
        )

    g_gradients = gt.gradient(g_loss, generator.trainable_variables)
    d_gradients = dt.gradient(d_loss, discriminator.trainable_variables)

    g_optimizer.apply_gradients(
        zip(g_gradients, generator.trainable_variables)
    )
    d_optimizer.apply_gradients(
        zip(d_gradients, discriminator.trainable_variables)
    )

    return g_loss, d_loss

# Train GAN
g_losses, d_losses = [], []

for epoch in range(EPOCHS):
    g_mean = tf.keras.metrics.Mean()
    d_mean = tf.keras.metrics.Mean()

    for real_images in dataset:
        g_loss, d_loss = train_step(real_images)
        g_mean.update_state(g_loss)
        d_mean.update_state(d_loss)

    g_losses.append(float(g_mean.result()))
    d_losses.append(float(d_mean.result()))

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"G Loss: {g_losses[-1]:.4f} | "
        f"D Loss: {d_losses[-1]:.4f}"
    )

# Plot losses
plt.plot(g_losses, label="Generator")
plt.plot(d_losses, label="Discriminator")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("GAN Training Loss")
plt.legend()
plt.grid()
plt.show()

# Generate images
noise = tf.random.normal([NUM_GENERATED_IMAGES, LATENT_DIM])
images = (generator(noise, training=False) + 1) / 2

fig, axes = plt.subplots(4, 4, figsize=(6, 6))
for ax, image in zip(axes.flat, images):
    ax.imshow(
        image[..., 0] if IMAGE_CHANNELS == 1 else image,
        cmap="gray" if IMAGE_CHANNELS == 1 else None,
        vmin=0, vmax=1
    )
    ax.axis("off")

plt.suptitle("Generated Images")
plt.tight_layout()
plt.show()
