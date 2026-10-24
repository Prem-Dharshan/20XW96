from pathlib import Path

from _runtime import setup

args = setup(__file__)

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# Variables
LATENT_DIM = 100
IMAGE_SIZE = 28
IMAGE_CHANNELS = 1
IMAGE_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)

GENERATOR_UNITS = [256, 512]
DISCRIMINATOR_UNITS = [512, 256]

ACTIVATION = "relu"
OUTPUT_ACTIVATION = "tanh"

OPTIMIZER = "adam"
LEARNING_RATE = 0.0002
BETA_1 = 0.5

LOSS = "binary_crossentropy"

EPOCHS = 1 if args.quick else 50
BATCH_SIZE = 128
BUFFER_SIZE = 60000

NUM_GENERATED_IMAGES = 16
RANDOM_STATE = 42


# Reproducibility
tf.keras.utils.set_random_seed(RANDOM_STATE)


# Load dataset
(x_train, _), (_, _) = tf.keras.datasets.fashion_mnist.load_data()

print("Training images:", x_train.shape)


if args.quick:
    x_train = x_train[:256]


# Preprocess images
x_train = x_train.astype("float32")
x_train = (x_train - 127.5) / 127.5
x_train = x_train[..., np.newaxis]

print("Preprocessed shape:", x_train.shape)
print("Minimum pixel value:", x_train.min())
print("Maximum pixel value:", x_train.max())


# Visualize real images
plt.figure(figsize=(8, 8))

for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(x_train[i].squeeze(), cmap="gray")
    plt.axis("off")

plt.suptitle("Real Fashion-MNIST Images")
plt.tight_layout()
plt.show()
plt.figure()


# Prepare training batches
train_dataset = (
    tf.data.Dataset.from_tensor_slices(x_train)
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.AUTOTUNE)
)


# Generator
generator = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(LATENT_DIM,)),

    tf.keras.layers.Dense(
        GENERATOR_UNITS[0],
        activation=ACTIVATION
    ),

    tf.keras.layers.Dense(
        GENERATOR_UNITS[1],
        activation=ACTIVATION
    ),

    tf.keras.layers.Dense(
        IMAGE_SIZE * IMAGE_SIZE * IMAGE_CHANNELS,
        activation=OUTPUT_ACTIVATION
    ),

    tf.keras.layers.Reshape(IMAGE_SHAPE)
], name="generator")


# Discriminator
discriminator = tf.keras.Sequential([
    tf.keras.layers.Input(shape=IMAGE_SHAPE),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        DISCRIMINATOR_UNITS[0]
    ),

    tf.keras.layers.LeakyReLU(negative_slope=0.2),

    tf.keras.layers.Dense(
        DISCRIMINATOR_UNITS[1]
    ),

    tf.keras.layers.LeakyReLU(negative_slope=0.2),

    tf.keras.layers.Dense(1)
], name="discriminator")


# Display model architectures
print("\nGenerator")
generator.summary()

print("\nDiscriminator")
discriminator.summary()


# Loss function
cross_entropy = tf.keras.losses.BinaryCrossentropy(
    from_logits=True
)


def generator_loss(fake_output):
    return cross_entropy(
        tf.ones_like(fake_output),
        fake_output
    )


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(
        tf.ones_like(real_output),
        real_output
    )

    fake_loss = cross_entropy(
        tf.zeros_like(fake_output),
        fake_output
    )

    return real_loss + fake_loss


# Optimizers
generator_optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE,
    beta_1=BETA_1
)

discriminator_optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE,
    beta_1=BETA_1
)


# Fixed noise for tracking progress
fixed_noise = tf.random.normal(
    shape=(NUM_GENERATED_IMAGES, LATENT_DIM),
    seed=RANDOM_STATE
)


# Training step
@tf.function
def train_step(real_images):
    noise = tf.random.normal(
        shape=(tf.shape(real_images)[0], LATENT_DIM)
    )

    with tf.GradientTape() as generator_tape, \
         tf.GradientTape() as discriminator_tape:

        generated_images = generator(
            noise,
            training=True
        )

        real_output = discriminator(
            real_images,
            training=True
        )

        fake_output = discriminator(
            generated_images,
            training=True
        )

        g_loss = generator_loss(fake_output)

        d_loss = discriminator_loss(
            real_output,
            fake_output
        )

    generator_gradients = generator_tape.gradient(
        g_loss,
        generator.trainable_variables
    )

    discriminator_gradients = discriminator_tape.gradient(
        d_loss,
        discriminator.trainable_variables
    )

    generator_optimizer.apply_gradients(
        zip(
            generator_gradients,
            generator.trainable_variables
        )
    )

    discriminator_optimizer.apply_gradients(
        zip(
            discriminator_gradients,
            discriminator.trainable_variables
        )
    )

    return g_loss, d_loss


# Generate images for visualization
def generate_images(model, noise):
    predictions = model(
        noise,
        training=False
    )

    predictions = (predictions + 1) / 2

    return tf.clip_by_value(
        predictions,
        0,
        1
    )


# Plot generated images
def plot_generated_images(model, noise, epoch):
    images = generate_images(model, noise)

    plt.figure(figsize=(8, 8))

    for i in range(len(images)):
        plt.subplot(4, 4, i + 1)

        plt.imshow(
            images[i, :, :, 0],
            cmap="gray",
            vmin=0,
            vmax=1
        )

        plt.axis("off")

    plt.suptitle(f"Generated Images - Epoch {epoch}")
    plt.tight_layout()
    plt.show()
plt.figure()


# Training loop
generator_losses = []
discriminator_losses = []

for epoch in range(EPOCHS):
    epoch_generator_loss = tf.keras.metrics.Mean()
    epoch_discriminator_loss = tf.keras.metrics.Mean()

    for real_images in train_dataset:
        g_loss, d_loss = train_step(real_images)

        epoch_generator_loss.update_state(g_loss)
        epoch_discriminator_loss.update_state(d_loss)

    g_loss_value = float(epoch_generator_loss.result())
    d_loss_value = float(epoch_discriminator_loss.result())

    generator_losses.append(g_loss_value)
    discriminator_losses.append(d_loss_value)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Generator Loss: {g_loss_value:.4f} | "
        f"Discriminator Loss: {d_loss_value:.4f}"
    )

    if (epoch + 1) % 5 == 0 or epoch == 0:
        plot_generated_images(
            generator,
            fixed_noise,
            epoch + 1
        )


# Plot generator and discriminator losses
plt.figure(figsize=(10, 5))

plt.plot(
    generator_losses,
    label="Generator Loss"
)

plt.plot(
    discriminator_losses,
    label="Discriminator Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("GAN Training Loss")
plt.legend()
plt.grid()
plt.show()
plt.figure()


# Generate final images
final_images = generate_images(
    generator,
    tf.random.normal(
        shape=(NUM_GENERATED_IMAGES, LATENT_DIM)
    )
)

print("\nGenerated image shape:", final_images.shape)


# Display final generated images
plt.figure(figsize=(8, 8))

for i in range(NUM_GENERATED_IMAGES):
    plt.subplot(4, 4, i + 1)

    plt.imshow(
        final_images[i, :, :, 0],
        cmap="gray",
        vmin=0,
        vmax=1
    )

    plt.axis("off")

plt.suptitle("Final GAN Generated Images")
plt.tight_layout()
plt.show()
plt.figure()
