
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split


# Variables

DATASET_PATH = "datasets/mammogram.csv"
TEXT_COLUMN = "Full_Report"
CSV_SEPARATOR = ";"

MAX_WORDS = 12000
SEQ_LEN = 20
STRIDE = 1
TRAIN_LIMIT = 100000

EMBEDDING_DIM = 64
RNN_UNITS = 128
RNN_LAYERS = 1
RECURRENT_LAYERS = ["SimpleRNN", "LSTM"]

ACTIVATION = "tanh"
OUTPUT_ACTIVATION = "softmax"
OPTIMIZER = "adam"
LOSS = "sparse_categorical_crossentropy"
METRICS = ["accuracy"]

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
EPOCHS = 10
BATCH_SIZE = 128
RANDOM_STATE = 42

PROMPT_WORDS = 5
GENERATION_LENGTH = 50
TEMPERATURE = 0.8


# Load dataset

df = pd.read_csv(DATASET_PATH, sep=CSV_SEPARATOR, encoding="utf-8")

reports = df[TEXT_COLUMN].dropna().astype(str).str.strip()
reports = reports[reports != ""].drop_duplicates().tolist()

print("Reports:", len(reports))


# Split dataset

train_reports, test_reports = train_test_split(
    reports, test_size=1 - TRAIN_SPLIT, random_state=RANDOM_STATE
)

train_reports, val_reports = train_test_split(
    train_reports, test_size=VAL_SPLIT, random_state=RANDOM_STATE
)


# Tokenize text

tokenizer = tf.keras.preprocessing.text.Tokenizer(
    num_words=MAX_WORDS, oov_token="<OOV>"
)

tokenizer.fit_on_texts(train_reports)

VOCAB_SIZE = min(MAX_WORDS, len(tokenizer.word_index) + 1)

train_sequences = tokenizer.texts_to_sequences(train_reports)
val_sequences = tokenizer.texts_to_sequences(val_reports)
test_sequences = tokenizer.texts_to_sequences(test_reports)


# Create many-to-many sequences

def create_sequences(sequences):
    X, y = [], []

    for sequence in sequences:
        for i in range(SEQ_LEN, len(sequence), STRIDE):
            X.append(sequence[i - SEQ_LEN:i])
            y.append(sequence[i])

            if len(X) >= TRAIN_LIMIT:
                break

        if len(X) >= TRAIN_LIMIT:
            break

    return np.array(X, dtype=np.int32), np.array(y, dtype=np.int32)


X_train, y_train = create_sequences(train_sequences)
X_val, y_val = create_sequences(val_sequences)
X_test, y_test = create_sequences(test_sequences)

print("Train:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)


# Train and compare models

results = []
best_loss = float("inf")
best_model = None
best_name = None

for layer_name in RECURRENT_LAYERS:

    tf.keras.backend.clear_session()

    recurrent_layer = (
        tf.keras.layers.SimpleRNN
        if layer_name == "SimpleRNN"
        else tf.keras.layers.LSTM
    )

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(SEQ_LEN,)),
        tf.keras.layers.Embedding(
            VOCAB_SIZE, EMBEDDING_DIM, mask_zero=True
        ),
        *[
            recurrent_layer(RNN_UNITS, return_sequences=True, activation=ACTIVATION)
            for _ in range(RNN_LAYERS - 1)
        ],
        recurrent_layer(RNN_UNITS, activation=ACTIVATION),
        tf.keras.layers.Dense(VOCAB_SIZE, activation=OUTPUT_ACTIVATION)
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
        batch_size=BATCH_SIZE,
        verbose=1
    )

    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    val_perplexity = np.exp(val_loss)
    test_perplexity = np.exp(test_loss)

    results.append({
        "Model": layer_name,
        "Validation Loss": val_loss,
        "Validation Perplexity": val_perplexity,
        "Validation Accuracy": val_accuracy,
        "Test Loss": test_loss,
        "Test Perplexity": test_perplexity,
        "Test Accuracy": test_accuracy
    })

    print(f"\n{layer_name}")
    print("Validation Cross-Entropy:", val_loss)
    print("Validation Perplexity:", val_perplexity)
    print("Test Cross-Entropy:", test_loss)
    print("Test Perplexity:", test_perplexity)
    print("Test Accuracy:", test_accuracy)

    if val_loss < best_loss:
        best_loss = val_loss
        best_model = tf.keras.models.clone_model(model)
        best_model.set_weights(model.get_weights())
        best_name = layer_name

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.title(f"{layer_name} Loss")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Validation")
    plt.title(f"{layer_name} Accuracy")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()


# Compare models

results_df = pd.DataFrame(results)

print("\nModel comparison:")
print(results_df.to_string(index=False))

print("\nBest model:", best_name)
print("Best validation cross-entropy:", best_loss)
print("Best validation perplexity:", np.exp(best_loss))


# Predict next word

def predict_next_word(prompt):
    sequence = tokenizer.texts_to_sequences([prompt])
    sequence = sequence[0][-SEQ_LEN:]
    sequence = tf.keras.utils.pad_sequences(
        [sequence], maxlen=SEQ_LEN, padding="pre"
    )

    probabilities = best_model.predict(sequence, verbose=0)[0]
    return tokenizer.index_word.get(int(np.argmax(probabilities)), "<UNK>")


# Generate text

def generate_text(seed, length=GENERATION_LENGTH, temperature=TEMPERATURE):
    if temperature <= 0:
        raise ValueError("Temperature must be greater than zero.")

    generated = seed

    for _ in range(length):
        sequence = tokenizer.texts_to_sequences([generated])[0][-SEQ_LEN:]
        sequence = tf.keras.utils.pad_sequences(
            [sequence], maxlen=SEQ_LEN, padding="pre"
        )

        probabilities = best_model.predict(sequence, verbose=0)[0]
        logits = np.log(np.maximum(probabilities, 1e-8)) / temperature
        logits[:2] = -np.inf
        probabilities = np.exp(logits - np.max(logits))
        probabilities /= probabilities.sum()

        next_id = np.random.choice(VOCAB_SIZE, p=probabilities)
        next_word = tokenizer.index_word.get(next_id, "<UNK>")
        generated += " " + next_word

    return generated


prompt = " ".join(test_reports[0].split()[:PROMPT_WORDS])

print("\nPrompt:", prompt)
print("Predicted next word:", predict_next_word(prompt))
print("Generated report:", generate_text(prompt))


# Compare cross-entropy and perplexity

plt.figure(figsize=(8, 4))
plt.bar(
    results_df["Model"],
    results_df["Test Loss"],
    label="Cross-Entropy"
)
plt.title("Test Cross-Entropy Comparison")
plt.ylabel("Cross-Entropy Loss")
plt.legend()
plt.grid(axis="y")
plt.show()

plt.figure(figsize=(8, 4))
plt.bar(
    results_df["Model"],
    results_df["Test Perplexity"],
    label="Perplexity"
)
plt.title("Test Perplexity Comparison")
plt.ylabel("Perplexity")
plt.legend()
plt.grid(axis="y")
plt.show()
