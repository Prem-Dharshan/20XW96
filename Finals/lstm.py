import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import nltk

from sklearn.model_selection import train_test_split
from numpy.lib.stride_tricks import sliding_window_view


DATASET_PATH = "datasets/alice.txt"
SEQ_LEN = 100
STRIDE = 3
TRAIN_LIMIT = 20000
TEST_LIMIT = 10000
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1

EMBEDDING_DIM = 32
LSTM_UNITS = 128
LSTM_LAYERS = 1
ACTIVATION = "softmax"
OPTIMIZER = "adam"
LOSS = "sparse_categorical_crossentropy"
METRICS = ["accuracy"]

EPOCHS = 30
BATCH_SIZE = 128

GENERATION_LENGTH = 300
TEMPERATURE = 0.7
RANDOM_STATE = 42


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

def create_sequences(data):
    sequences = sliding_window_view(data, SEQ_LEN + 1)[::STRIDE]
    return sequences[:, :-1].copy(), sequences[:, 1:].copy()

X_train, y_train = create_sequences(train_text)
X_test, y_test = create_sequences(test_text)

X_train, y_train = X_train[:TRAIN_LIMIT], y_train[:TRAIN_LIMIT]
X_test, y_test = X_test[:TEST_LIMIT], y_test[:TEST_LIMIT]

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=VAL_SPLIT, random_state=RANDOM_STATE
)

print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(SEQ_LEN,)),
    tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM),
    *[
        tf.keras.layers.LSTM(LSTM_UNITS, return_sequences=True)
        for _ in range(LSTM_LAYERS)
    ],
    tf.keras.layers.Dense(VOCAB_SIZE, activation=ACTIVATION)
])

model.compile(
    optimizer=OPTIMIZER,
    loss=LOSS,
    metrics=METRICS
)

model.summary()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

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

prediction = model.predict(X_test[:1], verbose=0)[0].argmax(axis=-1)

print("Input:    ", "".join(index_to_char[X_test[0]]))
print("Actual:   ", "".join(index_to_char[y_test[0]]))
print("Predicted:", "".join(index_to_char[prediction]))

def generate_text(seed, length=GENERATION_LENGTH, temperature=TEMPERATURE):

    generated = seed

    for _ in range(length):
        context = generated[-SEQ_LEN:]
        X = np.array([[char_to_index[c] for c in context]])

        probabilities = model.predict(X, verbose=0)[0, -1]
        probabilities = np.log(np.maximum(probabilities, 1e-8)) / temperature
        probabilities = np.exp(probabilities - np.max(probabilities))
        probabilities /= probabilities.sum()

        next_id = np.random.choice(VOCAB_SIZE, p=probabilities)
        generated += index_to_char[next_id]

    return generated

print(generate_text(text[:SEQ_LEN]))
