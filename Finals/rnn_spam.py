from pathlib import Path

from _runtime import setup

args = setup(__file__)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, roc_auc_score, classification_report,
    ConfusionMatrixDisplay
)


# Variables
DATASET_PATH = Path(__file__).resolve().parents[1] / "datasets" / "spam.csv"
TEXT_COLUMN = "text"
TARGET_COLUMN = "label"

MAX_WORDS = 10000
MAX_LENGTHS = [50, 100, 200]
HIDDEN_UNITS = [32, 64, 128]
EMBEDDING_DIM = 64

TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
RANDOM_STATE = 42

ACTIVATION = "tanh"
OUTPUT_ACTIVATION = "sigmoid"
OPTIMIZER = "adam"
LOSS = "binary_crossentropy"

EPOCHS = 1 if args.quick else 10
BATCH_SIZE = 32


# Reproducibility
tf.keras.utils.set_random_seed(RANDOM_STATE)


# Load data
df = pd.read_csv(DATASET_PATH, encoding="latin-1").rename(
    columns={"v1": TARGET_COLUMN, "v2": TEXT_COLUMN}
)

df = df[[TEXT_COLUMN, TARGET_COLUMN]].dropna()
df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).str.strip()
df = df[df[TEXT_COLUMN] != ""]

print("Dataset shape:", df.shape)
print(df.head())


# Encode labels
labels = df[TARGET_COLUMN].astype(str).str.strip().str.lower()

label_mapping = {
    "ham": 0,
    "not spam": 0,
    "legitimate": 0,
    "normal": 0,
    "0": 0,
    "spam": 1,
    "1": 1
}

if set(labels.unique()).issubset(label_mapping):
    y = labels.map(label_mapping).to_numpy(dtype=np.int32)
else:
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    if len(encoder.classes_) != 2:
        raise ValueError("Spam classification requires exactly two classes.")

    print("Label mapping:", dict(enumerate(encoder.classes_)))

X = df[TEXT_COLUMN].to_numpy()

if args.quick:
    X, _, y, _ = train_test_split(
        X, y, train_size=256, stratify=y, random_state=RANDOM_STATE
    )

print("Class distribution:", np.bincount(y))


# Split into train and test sets
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)


# Split training data into training and validation sets
X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X_train_raw, y_train,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_train
)


# Tokenize using training data only
tokenizer = tf.keras.preprocessing.text.Tokenizer(
    num_words=MAX_WORDS,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(X_train_raw)

train_sequences = tokenizer.texts_to_sequences(X_train_raw)
val_sequences = tokenizer.texts_to_sequences(X_val_raw)
test_sequences = tokenizer.texts_to_sequences(X_test_raw)

VOCAB_SIZE = min(
    MAX_WORDS,
    len(tokenizer.word_index) + 1
)

print("Vocabulary size:", VOCAB_SIZE)


# Store experiment results
results = []
best_auc = -1
best_model = None
best_length = None
best_units = None


# Train models for different sequence lengths and hidden units
for MAX_LEN in MAX_LENGTHS:
    X_train = tf.keras.utils.pad_sequences(
        train_sequences, maxlen=MAX_LEN,
        padding="post", truncating="post"
    )

    X_val = tf.keras.utils.pad_sequences(
        val_sequences, maxlen=MAX_LEN,
        padding="post", truncating="post"
    )

    X_test = tf.keras.utils.pad_sequences(
        test_sequences, maxlen=MAX_LEN,
        padding="post", truncating="post"
    )

    for RNN_UNITS in HIDDEN_UNITS:
        tf.keras.backend.clear_session()

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(MAX_LEN,)),

            tf.keras.layers.Embedding(
                input_dim=VOCAB_SIZE,
                output_dim=EMBEDDING_DIM,
                mask_zero=True
            ),

            tf.keras.layers.SimpleRNN(
                units=RNN_UNITS,
                activation=ACTIVATION
            ),

            tf.keras.layers.Dense(
                1,
                activation=OUTPUT_ACTIVATION
            )
        ])

        model.compile(
            optimizer=OPTIMIZER,
            loss=LOSS,
            metrics=[
                "accuracy",
                tf.keras.metrics.AUC(name="auc")
            ]
        )

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0
        )

        val_prob = model.predict(X_val, verbose=0).ravel()
        val_auc = roc_auc_score(y_val, val_prob)

        results.append({
            "Max Length": MAX_LEN,
            "Hidden Units": RNN_UNITS,
            "Best Validation Accuracy": max(history.history["val_accuracy"]),
            "Validation AUC": val_auc
        })

        print(
            f"RNN | Length={MAX_LEN} | Units={RNN_UNITS} "
            f"| Validation AUC={val_auc:.4f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            best_model = tf.keras.models.clone_model(model)
            best_model.set_weights(model.get_weights())
            best_length = MAX_LEN
            best_units = RNN_UNITS


# Compare configurations
results_df = pd.DataFrame(results)

print("\nExperiment results:")
print(results_df.sort_values("Validation AUC", ascending=False).to_string(index=False))

print("\nBest configuration:")
print("Maximum length:", best_length)
print("Hidden units:", best_units)
print("Validation AUC:", best_auc)


# Prepare test input for the selected configuration
X_test = tf.keras.utils.pad_sequences(
    test_sequences,
    maxlen=best_length,
    padding="post",
    truncating="post"
)


# Evaluate the selected model
y_prob = best_model.predict(X_test, verbose=0).ravel()
y_pred = (y_prob >= 0.5).astype(int)

print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("Test Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Test Recall:", recall_score(y_test, y_pred, zero_division=0))
print("Test F1:", f1_score(y_test, y_pred, zero_division=0))
print("Test ROC-AUC:", roc_auc_score(y_test, y_prob))

print("\nClassification report:")
print(classification_report(
    y_test, y_pred,
    target_names=["Legitimate", "Spam"],
    zero_division=0
))


# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)

plt.plot(fpr, tpr, label=f"SimpleRNN AUC = {auc_score:.4f}")
plt.plot([0, 1], [0, 1], "--", label="Random classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Spam Classification ROC Curve — SimpleRNN")
plt.legend()
plt.grid()
plt.show()
plt.figure()


# Confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=["Legitimate", "Spam"]
)
plt.title("SimpleRNN Confusion Matrix")
plt.grid(False)
plt.show()
plt.figure()


# Compare validation AUC across configurations
for units in HIDDEN_UNITS:
    subset = results_df[results_df["Hidden Units"] == units]

    plt.plot(
        subset["Max Length"],
        subset["Validation AUC"],
        marker="o",
        label=f"{units} units"
    )

plt.xlabel("Maximum Sequence Length")
plt.ylabel("Validation ROC-AUC")
plt.title("SimpleRNN Hyperparameter Comparison")
plt.legend()
plt.grid()
plt.show()
plt.figure()
