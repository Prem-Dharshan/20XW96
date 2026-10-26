import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Variables
LATENT_DIM = 100
EPOCHS = 5
BATCH_SIZE = 128

# Load and preprocess dataset
(x_train, _), _ = tf.keras.datasets.fashion_mnist.load_data()
x_train = (x_train.astype("float32") - 127.5) / 127.5
x_train = x_train[..., np.newaxis]

dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(60000).batch(BATCH_SIZE, drop_remainder=True)

# Generator
generator = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(LATENT_DIM,)),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dense(784, activation="tanh"),
    tf.keras.layers.Reshape((28, 28, 1))
])

# Discriminator
discriminator = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(512),
    tf.keras.layers.LeakyReLU(negative_slope=0.2),
    tf.keras.layers.Dense(256),
    tf.keras.layers.LeakyReLU(negative_slope=0.2),
    tf.keras.layers.Dense(1)
])

loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
g_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)
d_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)

# Training
g_losses, d_losses = [], []

for epoch in range(EPOCHS):
    g_loss_avg, d_loss_avg = tf.keras.metrics.Mean(), tf.keras.metrics.Mean()

    for real in dataset:
        noise = tf.random.normal((BATCH_SIZE, LATENT_DIM))

        with tf.GradientTape() as gt, tf.GradientTape() as dt:
            fake = generator(noise, training=True)
            real_out = discriminator(real, training=True)
            fake_out = discriminator(fake, training=True)

            g_loss = loss_fn(tf.ones_like(fake_out), fake_out)
            d_loss = (
                loss_fn(tf.ones_like(real_out), real_out)
                + loss_fn(tf.zeros_like(fake_out), fake_out)
            )

        g_optimizer.apply_gradients(zip(
            gt.gradient(g_loss, generator.trainable_variables),
            generator.trainable_variables
        ))
        d_optimizer.apply_gradients(zip(
            dt.gradient(d_loss, discriminator.trainable_variables),
            discriminator.trainable_variables
        ))

        g_loss_avg.update_state(g_loss)
        d_loss_avg.update_state(d_loss)

    g_losses.append(g_loss_avg.result().numpy())
    d_losses.append(d_loss_avg.result().numpy())
    print(f"Epoch {epoch+1}/{EPOCHS} | G Loss: {g_losses[-1]:.4f} | D Loss: {d_losses[-1]:.4f}")

# Plot losses
plt.plot(g_losses, label="Generator Loss")
plt.plot(d_losses, label="Discriminator Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()

# Generate and display images
noise = tf.random.normal((16, LATENT_DIM))
images = (generator(noise, training=False) + 1) / 2

plt.figure(figsize=(6, 6))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(images[i, :, :, 0], cmap="gray")
    plt.axis("off")

plt.tight_layout()
plt.show()
