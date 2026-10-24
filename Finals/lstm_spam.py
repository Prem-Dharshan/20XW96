
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, roc_auc_score, ConfusionMatrixDisplay
)


# Variables

DATASET_PATH = "datasets/spam.csv"
TEXT_COLUMN = "text"
TARGET_COLUMN = "label"

MAX_WORDS = 10000
MAX_LENGTHS = [50, 100, 200]
HIDDEN_UNITS = [32, 64, 128]
EMBEDDING_DIM = 64

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.2
RANDOM_STATE = 42

ACTIVATION = "tanh"
OUTPUT_ACTIVATION = "sigmoid"
OPTIMIZER = "adam"
LOSS = "binary_crossentropy"
METRICS = ["accuracy"]

EPOCHS = 10
BATCH_SIZE = 32


# Load dataset

df = pd.read_csv(DATASET_PATH, encoding="latin-1").rename(
    columns={"v1": TARGET_COLUMN, "v2": TEXT_COLUMN}
)

df = df[[TEXT_COLUMN, TARGET_COLUMN]].dropna()
df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).str.strip()
df = df[df[TEXT_COLUMN] != ""]

X = df[TEXT_COLUMN].to_numpy()
labels = df[TARGET_COLUMN].astype(str).str.strip().str.lower()

label_mapping = {
    "ham": 0, "not spam": 0, "legitimate": 0, "normal": 0, "0": 0,
    "spam": 1, "1": 1
}

if set(labels.unique()).issubset(label_mapping):
    y = labels.map(label_mapping).to_numpy(dtype=np.int32)
else:
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

print("Dataset shape:", df.shape)
print("Class distribution:", np.bincount(y))


# Split dataset

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=1 - TRAIN_SPLIT,
    random_state=RANDOM_STATE,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train,
    test_size=VAL_SPLIT,
    random_state=RANDOM_STATE,
    stratify=y_train
)


# Tokenize text

tokenizer = tf.keras.preprocessing.text.Tokenizer(
    num_words=MAX_WORDS,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_val = tokenizer.texts_to_sequences(X_val)
X_test = tokenizer.texts_to_sequences(X_test)

VOCAB_SIZE = min(MAX_WORDS, len(tokenizer.word_index) + 1)

print("Vocabulary size:", VOCAB_SIZE)


# Train and compare LSTM models

results = []
best_auc = -1
best_model = None
best_length = None
best_units = None

for MAX_LEN in MAX_LENGTHS:

    X_train_pad = tf.keras.utils.pad_sequences(
        X_train, maxlen=MAX_LEN, padding="post", truncating="post"
    )
    X_val_pad = tf.keras.utils.pad_sequences(
        X_val, maxlen=MAX_LEN, padding="post", truncating="post"
    )
    X_test_pad = tf.keras.utils.pad_sequences(
        X_test, maxlen=MAX_LEN, padding="post", truncating="post"
    )

    for LSTM_UNITS in HIDDEN_UNITS:

        tf.keras.backend.clear_session()

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(MAX_LEN,)),
            tf.keras.layers.Embedding(
                VOCAB_SIZE, EMBEDDING_DIM, mask_zero=True
            ),
            tf.keras.layers.LSTM(
                LSTM_UNITS, activation=ACTIVATION
            ),
            tf.keras.layers.Dense(
                1, activation=OUTPUT_ACTIVATION
            )
        ])

        model.compile(
            optimizer=OPTIMIZER,
            loss=LOSS,
            metrics=METRICS
        )

        history = model.fit(
            X_train_pad, y_train,
            validation_data=(X_val_pad, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0
        )

        val_prob = model.predict(X_val_pad, verbose=0).ravel()
        val_auc = roc_auc_score(y_val, val_prob)

        results.append({
            "Max Length": MAX_LEN,
            "Hidden Units": LSTM_UNITS,
            "Validation Accuracy": max(history.history["val_accuracy"]),
            "Validation AUC": val_auc
        })

        print(
            f"Length={MAX_LEN} | Units={LSTM_UNITS} "
            f"| Validation AUC={val_auc:.4f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            best_model = tf.keras.models.clone_model(model)
            best_model.set_weights(model.get_weights())
            best_length = MAX_LEN
            best_units = LSTM_UNITS


# Best configuration

results_df = pd.DataFrame(results)

print("\nExperiment results:")
print(results_df.to_string(index=False))

print("\nBest configuration:")
print("Maximum length:", best_length)
print("Hidden units:", best_units)
print("Validation AUC:", best_auc)


# Evaluate model

X_test_pad = tf.keras.utils.pad_sequences(
    X_test, maxlen=best_length, padding="post", truncating="post"
)

y_prob = best_model.predict(X_test_pad, verbose=0).ravel()
y_pred = (y_prob >= 0.5).astype(int)

print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("Test Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Test Recall:", recall_score(y_test, y_pred, zero_division=0))
print("Test F1:", f1_score(y_test, y_pred, zero_division=0))
print("Test ROC-AUC:", roc_auc_score(y_test, y_prob))


# Plot results

fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.4f}")
plt.plot([0, 1], [0, 1], "--")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid()

plt.subplot(1, 3, 2)
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=["Legitimate", "Spam"],
    ax=plt.gca()
)
plt.title("Confusion Matrix")

plt.subplot(1, 3, 3)
for units in HIDDEN_UNITS:
    subset = results_df[results_df["Hidden Units"] == units]
    plt.plot(
        subset["Max Length"], subset["Validation AUC"],
        marker="o", label=f"{units} units"
    )

plt.title("Hyperparameter Comparison")
plt.xlabel("Maximum Sequence Length")
plt.ylabel("Validation AUC")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
