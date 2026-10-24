import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split


# Variables
DATASET_PATH = "datasets/mammogram.csv"
TEXT_COLUMN = "Full_Report"

MAX_WORDS = 12000
MAX_LENGTH = 20
EMBEDDING_DIM = 64
LSTM_UNITS = 128
MAX_SAMPLES = 100000

TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1
RANDOM_STATE = 42

ACTIVATION = "tanh"
RECURRENT_ACTIVATION = "sigmoid"
OUTPUT_ACTIVATION = "softmax"

OPTIMIZER = "adam"
LOSS = "sparse_categorical_crossentropy"

EPOCHS = 10
BATCH_SIZE = 128

GENERATE_WORDS = 50
TEMPERATURE = 0.8


# Reproducibility
tf.keras.utils.set_random_seed(RANDOM_STATE)


# Load dataset
df = pd.read_csv(DATASET_PATH, sep=";")

df = df[[TEXT_COLUMN]].dropna()
df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).str.strip()
df = df[df[TEXT_COLUMN] != ""]

reports = df[TEXT_COLUMN].tolist()

if len(reports) < 10:
    raise ValueError("Not enough reports to train and evaluate the model.")

print("Number of reports:", len(reports))
print("\nSample report:")
print(reports[0][:1000])


# Split reports before creating word sequences
train_reports, test_reports = train_test_split(
    reports,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

train_reports, val_reports = train_test_split(
    train_reports,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE
)

print("\nTraining reports:", len(train_reports))
print("Validation reports:", len(val_reports))
print("Testing reports:", len(test_reports))


# Tokenize using training reports only
tokenizer = tf.keras.preprocessing.text.Tokenizer(
    num_words=MAX_WORDS,
    oov_token="<OOV>",
    filters=""
)

tokenizer.fit_on_texts(train_reports)

VOCAB_SIZE = min(
    MAX_WORDS,
    len(tokenizer.word_index) + 1
)

print("Vocabulary size:", VOCAB_SIZE)


# Convert reports into next-word training examples
def create_sequences(reports, tokenizer, max_len, max_samples=None):
    X, y = [], []

    for report in reports:
        sequence = tokenizer.texts_to_sequences([report])[0]

        for i in range(1, len(sequence)):
            context = sequence[max(0, i - max_len):i]
            target = sequence[i]

            X.append(context)
            y.append(target)

            if max_samples is not None and len(X) >= max_samples:
                break

        if max_samples is not None and len(X) >= max_samples:
            break

    X = tf.keras.utils.pad_sequences(
        X,
        maxlen=max_len,
        padding="pre",
        truncating="pre"
    )

    return np.asarray(X, dtype=np.int32), np.asarray(y, dtype=np.int32)


X_train, y_train = create_sequences(
    train_reports, tokenizer, MAX_LENGTH, MAX_SAMPLES
)

X_val, y_val = create_sequences(
    val_reports, tokenizer, MAX_LENGTH, MAX_SAMPLES // 5
)

X_test, y_test = create_sequences(
    test_reports, tokenizer, MAX_LENGTH, MAX_SAMPLES // 5
)

if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
    raise ValueError(
        "No next-word sequences were created. Check report lengths and dataset."
    )

print("\nTraining input:", X_train.shape)
print("Training target:", y_train.shape)
print("Validation input:", X_val.shape)
print("Testing input:", X_test.shape)


# Display a sample training example
index_to_word = {
    index: word
    for word, index in tokenizer.word_index.items()
    if index < VOCAB_SIZE
}


def decode_sequence(sequence):
    return " ".join(
        index_to_word.get(int(index), "<OOV>")
        for index in sequence
        if index != 0
    )


print("\nSample context:", decode_sequence(X_train[0]))
print("Expected next word:", index_to_word.get(int(y_train[0]), "<OOV>"))


# LSTM language model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(MAX_LENGTH,)),

    tf.keras.layers.Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=EMBEDDING_DIM,
        mask_zero=True
    ),

    tf.keras.layers.LSTM(
        units=LSTM_UNITS,
        activation=ACTIVATION,
        recurrent_activation=RECURRENT_ACTIVATION
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
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# Evaluate model
test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

perplexity = np.exp(test_loss)

print("\nTest Cross-Entropy Loss:", test_loss)
print("Test Perplexity:", perplexity)
print("Test Next-Word Accuracy:", test_accuracy)


# Plot training and validation loss
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("LSTM Language Model Loss")
plt.legend()
plt.grid()
plt.show()


# Plot training and validation accuracy
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Next-Word Accuracy")
plt.title("LSTM Language Model Accuracy")
plt.legend()
plt.grid()
plt.show()


# Predict the next word
def predict_next_word(text):
    sequence = tokenizer.texts_to_sequences([text])[0]

    if not sequence:
        raise ValueError("Input text contains no recognized or OOV tokens.")

    sequence = sequence[-MAX_LENGTH:]

    padded = tf.keras.utils.pad_sequences(
        [sequence],
        maxlen=MAX_LENGTH,
        padding="pre",
        truncating="pre"
    )

    probabilities = model.predict(padded, verbose=0)[0].copy()
    probabilities[0] = 0.0

    predicted_id = int(np.argmax(probabilities))

    return (
        index_to_word.get(predicted_id, "<OOV>"),
        float(probabilities[predicted_id])
    )


prompt = "no evidence of"

word, probability = predict_next_word(prompt)

print("\nPrompt:", prompt)
print("Predicted next word:", word)
print("Probability:", probability)


# Generate text using autoregressive sampling
def generate_text(seed_text, length=GENERATE_WORDS, temperature=TEMPERATURE):
    if temperature <= 0:
        raise ValueError("Temperature must be greater than zero.")

    generated = seed_text.strip()

    if not generated:
        raise ValueError("Seed text cannot be empty.")

    for _ in range(length):
        sequence = tokenizer.texts_to_sequences([generated])[0]

        if not sequence:
            break

        sequence = sequence[-MAX_LENGTH:]

        padded = tf.keras.utils.pad_sequences(
            [sequence],
            maxlen=MAX_LENGTH,
            padding="pre",
            truncating="pre"
        )

        probabilities = model.predict(padded, verbose=0)[0]

        logits = np.log(np.maximum(probabilities, 1e-8)) / temperature
        logits[0] = -1e9
        logits -= np.max(logits)

        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()

        predicted_id = int(
            np.random.choice(VOCAB_SIZE, p=probabilities)
        )

        next_word = index_to_word.get(predicted_id, "<OOV>")

        if next_word in ("<OOV>", "<PAD>"):
            continue

        generated += " " + next_word

    return generated


print("\nGenerated report text:")
print(
    generate_text(
        "breast tissue demonstrates",
        length=GENERATE_WORDS
    )
)
