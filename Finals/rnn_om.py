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

OUTPUT_LEN = 50
EMBEDDING_DIM = 32
RNN_UNITS = 128

TEST_SIZE = 0.2
RANDOM_STATE = 42

ACTIVATION = "tanh"
OUTPUT_ACTIVATION = "softmax"

OPTIMIZER = "adam"
LOSS = "sparse_categorical_crossentropy"

EPOCHS = 1 if args.quick else 10
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.1

MAX_SAMPLES = 128 if args.quick else 50000
STRIDE = 20


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


# Character vocabulary
vocab = sorted(set(text))
VOCAB_SIZE = len(vocab)

char_to_index = {
    char: index
    for index, char in enumerate(vocab)
}

index_to_char = np.array(vocab)

print("\nVocabulary size:", VOCAB_SIZE)
print("Vocabulary:", "".join(vocab))


# Encode text
text_ids = np.array(
    [char_to_index[char] for char in text],
    dtype=np.int32
)


# Split text before creating training sequences
split_index = int(len(text_ids) * (1 - TEST_SIZE))

train_text = text_ids[:split_index]
test_text = text_ids[split_index:]


# Create one-to-many input and target pairs
def create_sequences(data, output_len, stride, max_samples=None):
    X, y = [], []

    end = len(data) - output_len

    for i in range(0, end, stride):
        X.append(data[i:i + 1])
        y.append(data[i + 1:i + output_len + 1])

        if max_samples is not None and len(X) >= max_samples:
            break

    return np.array(X), np.array(y)


X_train, y_train = create_sequences(
    train_text,
    OUTPUT_LEN,
    STRIDE,
    MAX_SAMPLES
)

X_test, y_test = create_sequences(
    test_text,
    OUTPUT_LEN,
    STRIDE,
    MAX_SAMPLES // 5
)

print("\nTraining input shape:", X_train.shape)
print("Training target shape:", y_train.shape)
print("Testing input shape:", X_test.shape)
print("Testing target shape:", y_test.shape)


# Inspect a training example
print("\nInput character:")
print(index_to_char[X_train[0, 0]])

print("\nExpected output sequence:")
print("".join(index_to_char[y_train[0]]))


# Split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=VALIDATION_SPLIT,
    random_state=RANDOM_STATE,
    shuffle=True
)

print("\nFinal training shape:", X_train.shape)
print("Validation shape:", X_val.shape)


# One-to-many RNN
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),

    tf.keras.layers.Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=EMBEDDING_DIM
    ),

    tf.keras.layers.Flatten(),

    tf.keras.layers.RepeatVector(OUTPUT_LEN),

    tf.keras.layers.SimpleRNN(
        units=RNN_UNITS,
        activation=ACTIVATION,
        return_sequences=True
    ),

    tf.keras.layers.Dense(
        units=VOCAB_SIZE,
        activation=OUTPUT_ACTIVATION
    )
])


# Display model architecture
model.summary()


# Compile model
model.compile(
    optimizer=OPTIMIZER,
    loss=LOSS,
    metrics=["accuracy"]
)


# Train model
history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    shuffle=True
)


# Evaluate model
test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\nTest Loss:", test_loss)
print("Test Character Accuracy:", test_accuracy)


# Training and validation loss
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
plt.title("One-to-Many RNN Loss")
plt.legend()
plt.grid()
plt.show()
plt.figure()


# Training and validation accuracy
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
plt.title("One-to-Many RNN Accuracy")
plt.legend()
plt.grid()
plt.show()
plt.figure()


# Generate a sequence from one starting character
def generate_text(start_char, temperature=0.8):
    if len(start_char) != 1:
        raise ValueError("Provide exactly one starting character.")

    if start_char not in char_to_index:
        raise ValueError("Starting character is not in the vocabulary.")

    if temperature <= 0:
        raise ValueError("Temperature must be greater than zero.")

    input_ids = np.array(
        [[char_to_index[start_char]]],
        dtype=np.int32
    )

    probabilities = model.predict(
        input_ids,
        verbose=0
    )[0]

    # Temperature controls the randomness of sampling
    logits = np.log(
        np.maximum(probabilities, 1e-8)
    ) / temperature

    logits = logits - np.max(logits)

    probabilities = np.exp(logits)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)

    # Sample one character at each output timestep
    generated_ids = [
        np.random.choice(VOCAB_SIZE, p=probabilities[t])
        for t in range(OUTPUT_LEN)
    ]

    return "".join(index_to_char[generated_ids])


# Generate sample outputs
for start_char in ["T", "A", "W"]:
    print(f"\nStarting character: {start_char}")
    print("Generated text:")
    print(start_char + generate_text(start_char))
    