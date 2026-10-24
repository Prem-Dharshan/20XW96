from pathlib import Path

from _runtime import setup

args = setup(__file__)

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split


# Variables
DATA_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/data/"
    "shakespeare.txt"
)

SEQ_LEN = 100
STRIDE = 3
MAX_SAMPLES = 128 if args.quick else 50000

EMBEDDING_DIM = 32
LSTM_UNITS = 128

TEST_SIZE = 0.2
RANDOM_STATE = 42

ACTIVATION = "tanh"
RECURRENT_ACTIVATION = "sigmoid"
OUTPUT_ACTIVATION = "softmax"

OPTIMIZER = "adam"
LOSS = "sparse_categorical_crossentropy"

EPOCHS = 1 if args.quick else 10
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.1

TEMPERATURE = 0.8
GENERATE_LENGTH = 10 if args.quick else 300


# Reproducibility
tf.keras.utils.set_random_seed(RANDOM_STATE)


# Load dataset
path = tf.keras.utils.get_file(
    "shakespeare.txt",
    DATA_URL
)

with open(path, "r", encoding="utf-8") as file:
    text = file.read()

print("Total characters:", len(text))
print("\nSample text:")
print(text[:300])


# Create character vocabulary
vocab = sorted(set(text))
VOCAB_SIZE = len(vocab)

char_to_index = {
    char: index
    for index, char in enumerate(vocab)
}

index_to_char = np.array(vocab)

print("\nVocabulary size:", VOCAB_SIZE)


# Encode text
text_ids = np.array(
    [char_to_index[char] for char in text],
    dtype=np.int32
)


# Split text before creating sequences
split_index = int(len(text_ids) * (1 - TEST_SIZE))

train_text = text_ids[:split_index]
test_text = text_ids[split_index:]


# Create input and target sequences
def create_sequences(data, seq_len, stride, max_samples=None):
    X, y = [], []

    for i in range(0, len(data) - seq_len, stride):
        X.append(data[i:i + seq_len])
        y.append(data[i + 1:i + seq_len + 1])

        if max_samples is not None and len(X) >= max_samples:
            break

    return np.array(X), np.array(y)


X_train, y_train = create_sequences(
    train_text,
    SEQ_LEN,
    STRIDE,
    MAX_SAMPLES
)

X_test, y_test = create_sequences(
    test_text,
    SEQ_LEN,
    STRIDE,
    MAX_SAMPLES // 5
)

print("\nTraining input shape:", X_train.shape)
print("Training target shape:", y_train.shape)
print("Testing input shape:", X_test.shape)
print("Testing target shape:", y_test.shape)


# Inspect a sample sequence
print("\nInput sequence:")
print("".join(index_to_char[X_train[0]]))

print("\nTarget sequence:")
print("".join(index_to_char[y_train[0]]))


# Split training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=VALIDATION_SPLIT,
    random_state=RANDOM_STATE,
    shuffle=True
)


# Many-to-many LSTM
lstm = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(None,)),

    tf.keras.layers.Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=EMBEDDING_DIM
    ),

    tf.keras.layers.LSTM(
        units=LSTM_UNITS,
        activation=ACTIVATION,
        recurrent_activation=RECURRENT_ACTIVATION,
        return_sequences=True
    ),

    tf.keras.layers.Dense(
        units=VOCAB_SIZE,
        activation=OUTPUT_ACTIVATION
    )
])


# Display model architecture
lstm.summary()


# Compile model
lstm.compile(
    optimizer=OPTIMIZER,
    loss=LOSS,
    metrics=["accuracy"]
)


# Train model
history = lstm.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    shuffle=True
)


# Evaluate model
test_loss, test_accuracy = lstm.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\nTest Loss:", test_loss)
print("Test Character Accuracy:", test_accuracy)


# Predict test sequences
predictions = lstm.predict(
    X_test[:1],
    verbose=0
)

predicted_ids = np.argmax(
    predictions[0],
    axis=-1
)

print("\nInput sequence:")
print("".join(index_to_char[X_test[0]]))

print("\nActual next-character sequence:")
print("".join(index_to_char[y_test[0]]))

print("\nPredicted next-character sequence:")
print("".join(index_to_char[predicted_ids]))


# Plot accuracy
plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("LSTM Accuracy")
plt.legend()
plt.grid()
plt.show()
plt.figure()


# Plot loss
plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("LSTM Loss")
plt.legend()
plt.grid()
plt.show()
plt.figure()


# Generate text
def generate_text(seed_text, length=GENERATE_LENGTH, temperature=TEMPERATURE):
    if not seed_text:
        raise ValueError("Seed text cannot be empty.")

    if temperature <= 0:
        raise ValueError("Temperature must be greater than zero.")

    unknown_chars = [
        char for char in seed_text
        if char not in char_to_index
    ]

    if unknown_chars:
        raise ValueError(
            f"Characters not in vocabulary: {unknown_chars}"
        )

    generated = seed_text

    for _ in range(length):
        context = generated[-SEQ_LEN:]

        input_ids = np.array([
            [char_to_index[char] for char in context]
        ], dtype=np.int32)

        probabilities = lstm.predict(
            input_ids,
            verbose=0
        )[0, -1]

        logits = np.log(
            np.maximum(probabilities, 1e-8)
        ) / temperature

        logits -= np.max(logits)

        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()

        next_id = np.random.choice(
            VOCAB_SIZE,
            p=probabilities
        )

        generated += index_to_char[next_id]

    return generated


# Generate sample text
print("\nGenerated Text:")
print(
    generate_text(
        seed_text="ROMEO: ",
        length=GENERATE_LENGTH,
        temperature=TEMPERATURE
    )
)
