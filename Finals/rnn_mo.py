from pathlib import Path

from _runtime import setup

args = setup(__file__)

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


# Variables
MAX_WORDS = 10000
MAX_LEN = 200

EMBEDDING_DIM = 64
RNN_UNITS = 64

TEST_SIZE = 0.2
RANDOM_STATE = 42

ACTIVATION = "tanh"
OUTPUT_ACTIVATION = "sigmoid"

OPTIMIZER = "adam"
LOSS = "binary_crossentropy"

EPOCHS = 1 if args.quick else 10
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.2


# Load dataset
(X_train, y_train), (X_test, y_test) = (
    tf.keras.datasets.imdb.load_data(
        num_words=MAX_WORDS
    )
)

if args.quick:
    X_train, y_train = X_train[:256], y_train[:256]
    X_test, y_test = X_test[:64], y_test[:64]

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Training labels:", y_train.shape)
print("Testing labels:", y_test.shape)


# Pad sequences to a fixed length
X_train = pad_sequences(
    X_train,
    maxlen=MAX_LEN,
    padding="post",
    truncating="post"
)

X_test = pad_sequences(
    X_test,
    maxlen=MAX_LEN,
    padding="post",
    truncating="post"
)

print("Training input shape:", X_train.shape)
print("Testing input shape:", X_test.shape)

print("Positive reviews:", np.sum(y_train == 1))
print("Negative reviews:", np.sum(y_train == 0))


# Decode a sample review
word_index = tf.keras.datasets.imdb.get_word_index()

index_to_word = {
    index + 3: word
    for word, index in word_index.items()
}

index_to_word[0] = "<PAD>"
index_to_word[1] = "<START>"
index_to_word[2] = "<UNK>"
index_to_word[3] = "<UNUSED>"


def decode_review(sequence):
    return " ".join(
        index_to_word.get(index, "?")
        for index in sequence
        if index != 0
    )


print("\nSample review:")
print(decode_review(X_train[0])[:1000])
print("Sentiment:", "Positive" if y_train[0] == 1 else "Negative")


# Split training data into training and validation sets
rng = np.random.default_rng(RANDOM_STATE)

indices = rng.permutation(len(X_train))
split_index = int(len(X_train) * (1 - VALIDATION_SPLIT))

train_indices = indices[:split_index]
val_indices = indices[split_index:]

X_train_final = X_train[train_indices]
y_train_final = y_train[train_indices]

X_val = X_train[val_indices]
y_val = y_train[val_indices]

print("\nFinal training shape:", X_train_final.shape)
print("Validation shape:", X_val.shape)


# Many-to-one RNN
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(MAX_LEN,)),

    tf.keras.layers.Embedding(
        input_dim=MAX_WORDS,
        output_dim=EMBEDDING_DIM,
        mask_zero=True
    ),

    tf.keras.layers.SimpleRNN(
        units=RNN_UNITS,
        activation=ACTIVATION
    ),

    tf.keras.layers.Dense(
        units=1,
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
    X_train_final,
    y_train_final,
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
print("Test Accuracy:", test_accuracy)


# Predict test labels
y_prob = model.predict(
    X_test,
    verbose=0
).ravel()

y_pred = (y_prob >= 0.5).astype(int)


# Classification report
print("\nClassification Report")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Negative", "Positive"],
        digits=4
    )
)


# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Negative", "Positive"]
)

disp.plot()
plt.title("IMDB Sentiment Classification")
plt.grid(False)
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
plt.title("RNN Accuracy")
plt.legend()
plt.grid()
plt.show()
plt.figure()


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
plt.title("RNN Loss")
plt.legend()
plt.grid()
plt.show()
plt.figure()


# Display sample predictions
for i in range(10):
    actual = "Positive" if y_test[i] == 1 else "Negative"
    predicted = "Positive" if y_pred[i] == 1 else "Negative"

    print(
        f"\nReview {i + 1}"
        f"\nActual: {actual}"
        f"\nPredicted: {predicted}"
        f"\nPositive probability: {y_prob[i]:.4f}"
        f"\nReview: {decode_review(X_test[i])[:300]}"
    )


# Predict a new review
new_review = (
    "This movie was excellent, the acting was great "
    "and the story was wonderful"
)


# Convert words to IMDB word indices
new_review_words = new_review.lower().split()

new_review_sequence = [
    word_index.get(word, -1) + 3
    if word_index.get(word, -1) + 3 < MAX_WORDS
    else 2
    for word in new_review_words
]

new_review_padded = pad_sequences(
    [new_review_sequence],
    maxlen=MAX_LEN,
    padding="post",
    truncating="post"
)

# Predict sentiment
new_probability = model.predict(
    new_review_padded,
    verbose=0
)[0][0]

new_sentiment = (
    "Positive" if new_probability >= 0.5 else "Negative"
)

print("\nNew Review:", new_review)
print("Predicted Sentiment:", new_sentiment)
print("Positive Probability:", round(float(new_probability), 4))
