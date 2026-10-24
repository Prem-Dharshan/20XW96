
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import nltk

from sklearn.model_selection import train_test_split


# Variables

DATASET_PATH = "datasets/shakespeare.txt"
SEQ_LEN = 100
STRIDE = 3
TRAIN_LIMIT = 20000
TEST_LIMIT = 10000
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1

EMBEDDING_DIM = 32
RNN_UNITS = 128
RNN_LAYERS = 1
ACTIVATION = "tanh"
OUTPUT_ACTIVATION = "softmax"
OPTIMIZER = "adam"
LOSS = "sparse_categorical_crossentropy"
METRICS = ["accuracy"]

EPOCHS = 30
BATCH_SIZE = 128
RANDOM_STATE = 42


# Load dataset

with open(DATASET_PATH, encoding="utf-8") as file:
    text = file.read()

print("Characters:", len(text))

vocab = sorted(nltk.FreqDist(text))
VOCAB_SIZE = len(vocab)
char_to_index = {c: i for i, c in enumerate(vocab)}
index_to_char = np.array(vocab)
text_ids = np.array([char_to_index[c] for c in text])

split = int(len(text_ids) * TRAIN_SPLIT)
train_text, test_text = text_ids[:split], text_ids[split:]


# Create one-to-many sequences

def create_sequences(data):
    sequences = np.lib.stride_tricks.sliding_window_view(
        data, SEQ_LEN + 1
    )[::STRIDE]

    return sequences[:, 0], sequences[:, 1:].copy()

X_train, y_train = create_sequences(train_text)
X_test, y_test = create_sequences(test_text)

X_train, y_train = X_train[:TRAIN_LIMIT], y_train[:TRAIN_LIMIT]
X_test, y_test = X_test[:TEST_LIMIT], y_test[:TEST_LIMIT]

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=VAL_SPLIT, random_state=RANDOM_STATE
)

print("Train:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)


# Build RNN

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM),
    tf.keras.layers.RepeatVector(SEQ_LEN),
    *[
        tf.keras.layers.SimpleRNN(
            RNN_UNITS, return_sequences=True, activation=ACTIVATION
        )
        for _ in range(RNN_LAYERS)
    ],
    tf.keras.layers.Dense(VOCAB_SIZE, activation=OUTPUT_ACTIVATION)
])

model.compile(
    optimizer=OPTIMIZER,
    loss=LOSS,
    metrics=METRICS
)

model.summary()


# Train

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)


# Evaluate

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)


# Plot results

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Validation")
plt.title("Loss")
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("Accuracy")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()


# Predict sequence

prediction = model.predict(X_test[:1], verbose=0)[0].argmax(axis=-1)

print("Input:    ", index_to_char[X_test[0]])
print("Actual:   ", "".join(index_to_char[y_test[0]]))
print("Predicted:", "".join(index_to_char[prediction]))


# Generate text

def generate_text(seed, length=300):
    generated = seed

    for _ in range(length):
        X = np.array([[char_to_index[generated[-1]]]])
        probabilities = model.predict(X, verbose=0)[0, -1]
        next_id = np.random.choice(VOCAB_SIZE, p=probabilities)
        generated += index_to_char[next_id]

    return generated

print(generate_text(text[0], length=300))
