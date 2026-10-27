
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import nltk



# Variables

DATASET_PATH = "datasets/shakespeare.txt"
TRAIN_LIMIT = 20000
TEST_LIMIT = 10000
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1

EMBEDDING_DIM = 32
RNN_UNITS = 128
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

# Split text before creating sequences to keep dataset boundaries disjoint.
train_end = int(len(text_ids) * TRAIN_SPLIT)
val_end = int(len(text_ids) * (TRAIN_SPLIT + VAL_SPLIT))
train_text = text_ids[:train_end]
val_text = text_ids[train_end:val_end]
test_text = text_ids[val_end:]

print("Train characters:", len(train_text))
print("Validation characters:", len(val_text))
print("Test characters:", len(test_text))


# Create one-to-one sequences

def create_sequences(data):
    return data[:-1].reshape(-1, 1), data[1:]

X_train, y_train = create_sequences(train_text)
X_val, y_val = create_sequences(val_text)
X_test, y_test = create_sequences(test_text)

X_train, y_train = X_train[:TRAIN_LIMIT], y_train[:TRAIN_LIMIT]
X_test, y_test = X_test[:TEST_LIMIT], y_test[:TEST_LIMIT]

print("Train:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)


# Build RNN

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM),
    tf.keras.layers.SimpleRNN(RNN_UNITS, activation=ACTIVATION),
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


# Predict next character

prediction = model.predict(X_test[:1], verbose=0)[0].argmax()

print("Input:    ", index_to_char[X_test[0, 0]])
print("Actual:   ", index_to_char[y_test[0]])
print("Predicted:", index_to_char[prediction])
