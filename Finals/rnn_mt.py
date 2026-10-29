import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# Variables
DATASET_PATH = "datasets/translation.tsv"
TARGET_COLUMN = None       # Column name or zero-based index; None = auto
TARGET_LANGUAGE_KEY = None # JSON language key, e.g. "fr"

MAX_VOCAB = 20000
MAX_LENGTHS = [10, 20, 30]
HIDDEN_UNITS = [64, 128]
EMBEDDING_DIM = 64
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TRAIN_LIMIT = 50000
VAL_LIMIT = 10000
TEST_LIMIT = 10000
EPOCHS = 10
BATCH_SIZE = 128
RANDOM_STATE = 42
BEST_MODEL_PATH = "best_translation_model.keras"


# Load local text from TXT, CSV, TSV, JSON, or JSONL
def clean(value):
    return value.strip() if isinstance(value, str) and value.strip() else None


def json_text(row):
    if isinstance(row, str):
        return clean(row)
    if isinstance(row, list):
        try:
            index = int(TARGET_COLUMN) if TARGET_COLUMN is not None else -1
            return json_text(row[index])
        except (ValueError, TypeError, IndexError):
            return next((v for x in reversed(row) if (v := json_text(x))), None)
    if not isinstance(row, dict):
        return None

    if TARGET_LANGUAGE_KEY:
        value = row.get(TARGET_LANGUAGE_KEY)
        if value is None:
            for key in ("translation", "translations"):
                nested = row.get(key)
                if isinstance(nested, dict) and TARGET_LANGUAGE_KEY in nested:
                    value = nested[TARGET_LANGUAGE_KEY]
                    break
        result = json_text(value)
        if result:
            return result

    if TARGET_COLUMN is not None:
        value = row
        for part in str(TARGET_COLUMN).split("."):
            value = value.get(part) if isinstance(value, dict) else None
        result = json_text(value)
        if result:
            return result

    keys = ("target_text", "target", "translated_text", "translation",
            "translated", "french", "fr", "german", "de", "spanish",
            "es", "text", "sentence")
    for key in keys:
        if key in row:
            result = json_text(row[key])
            if result:
                return result
    return next((v for x in reversed(list(row.values()))
                 if (v := clean(x))), None)


def load_dataset(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")

    ext = path.suffix.lower()
    if ext in (".txt", ".text"):
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        if any("\t" in line for line in lines):
            rows = list(csv.reader(lines, delimiter="\t"))
            return load_rows(rows, path.name)
        texts = [line.strip() for line in lines if line.strip()]
    elif ext in (".csv", ".tsv", ".tab"):
        delimiter = "," if ext == ".csv" else "\t"
        with path.open(encoding="utf-8-sig", newline="") as file:
            texts = load_rows(list(csv.reader(file, delimiter=delimiter)), path.name)
    elif ext in (".json", ".jsonl", ".ndjson"):
        with path.open(encoding="utf-8-sig") as file:
            if ext in (".jsonl", ".ndjson"):
                records = []
                for line_no, line in enumerate(file, 1):
                    if line.strip():
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError as error:
                            raise ValueError(f"Invalid JSON at line {line_no}: {error}") from error
            else:
                data = json.load(file)
                records = data if isinstance(data, list) else next(
                    (data[k] for k in ("data", "rows", "records", "examples")
                     if isinstance(data, dict) and isinstance(data.get(k), list)),
                    [data] if isinstance(data, dict) else []
                )
        texts = [text for row in records if (text := json_text(row))]
    else:
        raise ValueError("Supported formats: .txt, .csv, .tsv, .tab, .json, .jsonl, .ndjson")

    texts = [text for text in texts if text]
    if len(texts) < 3:
        raise ValueError("Fewer than 3 usable texts found; check the file and target column.")
    print(f"Dataset: {path.name} | usable texts: {len(texts)}")
    print("Sample:", texts[0][:200])
    return texts


def load_rows(rows, name):
    rows = [row for row in rows if row and any(cell.strip() for cell in row)]
    if not rows:
        raise ValueError(f"No usable rows in {name}")

    normalize = lambda value: value.strip().lower().lstrip("\\ufeff")
    headers = [normalize(v) for v in rows[0]]
    target_names = {"target", "target_text", "translation", "translated",
                    "translated_text", "french", "fr", "german", "de",
                    "spanish", "es", "sentence", "text"}
    has_header = bool(target_names.intersection(headers)) or (
        isinstance(TARGET_COLUMN, str) and TARGET_COLUMN.lower() in headers
    )
    data = rows[1:] if has_header else rows
    if not data:
        raise ValueError(f"No data rows in {name}")

    if isinstance(TARGET_COLUMN, int):
        column = TARGET_COLUMN
    elif isinstance(TARGET_COLUMN, str):
        if not has_header or TARGET_COLUMN.lower() not in headers:
            raise ValueError(f"Target column '{TARGET_COLUMN}' not found in headers.")
        column = headers.index(TARGET_COLUMN.lower())
    elif has_header:
        preferred = ("target", "target_text", "translation", "translated_text",
                     "translated", "french", "fr", "german", "de",
                     "spanish", "es", "sentence", "text")
        column = next((headers.index(k) for k in preferred if k in headers), len(headers) - 1)
    else:
        column = len(data[0]) - 1

    texts = [clean(row[column]) for row in data if -len(row) <= column < len(row)]
    texts = [text for text in texts if text]
    if len(texts) < 3:
        raise ValueError("Fewer than 3 usable texts; check TARGET_COLUMN.")
    print(f"Dataset: {name} | usable texts: {len(texts)}")
    print("Sample:", texts[0][:200])
    return texts


texts = load_dataset(DATASET_PATH)


# Split records before creating next-word examples
rng = np.random.default_rng(RANDOM_STATE)
indices = rng.permutation(len(texts))
train_end = int(len(texts) * TRAIN_SPLIT)
val_end = int(len(texts) * (TRAIN_SPLIT + VAL_SPLIT))
train_texts = [texts[i] for i in indices[:train_end]]
val_texts = [texts[i] for i in indices[train_end:val_end]]
test_texts = [texts[i] for i in indices[val_end:]]
if not train_texts or not val_texts or not test_texts:
    raise ValueError("Dataset needs enough records for train/validation/test splits.")

print("Train / validation / test records:",
      len(train_texts), len(val_texts), len(test_texts))


# Fit vocabulary on training data only
vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=MAX_VOCAB, standardize="lower",
    split="whitespace", output_mode="int"
)
vectorizer.adapt(tf.data.Dataset.from_tensor_slices(train_texts).batch(1024))
vocabulary = vectorizer.get_vocabulary()
VOCAB_SIZE = len(vocabulary)
if VOCAB_SIZE < 3:
    raise ValueError("Vocabulary is too small; check the target text.")
print("Vocabulary size:", VOCAB_SIZE)


# Each prefix predicts the next word; pre-padding supports masking
def create_sequences(sentences, max_length, limit):
    X, y = [], []
    for sentence in sentences:
        ids = vectorizer(tf.constant([sentence]))[0].numpy()
        ids = ids[ids != 0]
        for i in range(1, len(ids)):
            context = ids[max(0, i - max_length):i]
            padded = np.zeros(max_length, dtype=np.int32)
            padded[-len(context):] = context
            X.append(padded)
            y.append(ids[i])
            if len(X) >= limit:
                break
        if len(X) >= limit:
            break
    if not X:
        raise ValueError("No next-word examples. Check the dataset and target column.")
    X, y = np.asarray(X), np.asarray(y)
    order = rng.permutation(len(X))
    return X[order], y[order]


def build_model(max_length, units, kind):
    recurrent = (tf.keras.layers.SimpleRNN(units, activation="tanh")
                 if kind == "RNN" else tf.keras.layers.LSTM(units))
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(max_length,)),
        tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, mask_zero=True),
        recurrent,
        tf.keras.layers.Dense(VOCAB_SIZE, activation="softmax")
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["sparse_categorical_accuracy"])
    return model


# Train configurations and select by validation cross-entropy/perplexity
results = []
best_loss = float("inf")
best_config = None
best_history = None

for kind in ("RNN", "LSTM"):
    for max_length in MAX_LENGTHS:
        X_train, y_train = create_sequences(train_texts, max_length, TRAIN_LIMIT)
        X_val, y_val = create_sequences(val_texts, max_length, VAL_LIMIT)

        for units in HIDDEN_UNITS:
            print(f"\\nTraining {kind}: units={units}, max_length={max_length}")
            tf.keras.utils.set_random_seed(RANDOM_STATE)
            model = build_model(max_length, units, kind)
            history = model.fit(
                X_train, y_train, validation_data=(X_val, y_val),
                epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1,
                callbacks=[tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=2, restore_best_weights=True
                )]
            )
            val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
            result = {
                "model": kind, "units": units, "max_length": max_length,
                "val_loss": float(val_loss),
                "val_perplexity": float(np.exp(min(val_loss, 80))),
                "val_accuracy": float(val_accuracy)
            }
            results.append(result)
            print(result)

            if val_loss < best_loss:
                best_loss = val_loss
                best_config = result.copy()
                best_history = {k: v.copy() for k, v in history.history.items()}
                model.save(BEST_MODEL_PATH)
                print("Saved new best model:", BEST_MODEL_PATH)
            del model


# Compare models
results.sort(key=lambda item: item["val_perplexity"])
print("\\nMODEL COMPARISON")
for row in results:
    print(f"{row['model']:4} units={row['units']:3} length={row['max_length']:2} "
          f"loss={row['val_loss']:.4f} PPL={row['val_perplexity']:.4f} "
          f"accuracy={row['val_accuracy']:.4f}")
print("\\nBEST CONFIGURATION:", best_config)

best_model = tf.keras.models.load_model(BEST_MODEL_PATH)
best_length = best_config["max_length"]
X_test, y_test = create_sequences(test_texts, best_length, TEST_LIMIT)
test_loss, test_accuracy = best_model.evaluate(X_test, y_test, verbose=0)
test_perplexity = float(np.exp(min(test_loss, 80)))
print("\\nTEST RESULTS")
print("Cross-entropy:", round(test_loss, 4))
print("Perplexity:", round(test_perplexity, 4))
print("Accuracy:", round(test_accuracy, 4))


# Plot best model loss and validation perplexity comparison
plt.figure(figsize=(8, 4))
plt.plot(best_history["loss"], label="Training")
plt.plot(best_history["val_loss"], label="Validation")
plt.title("Best Model: Cross-Entropy Loss")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

labels = [f"{r['model']}-{r['units']}-L{r['max_length']}" for r in results]
plt.figure(figsize=(10, 4))
plt.bar(labels, [r["val_perplexity"] for r in results])
plt.title("Validation Perplexity (lower is better)")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Perplexity")
plt.grid(axis="y")
plt.tight_layout()
plt.show()


# Predict next word and generate text with the selected model
def encode_context(text):
    ids = vectorizer(tf.constant([text]))[0].numpy()
    ids = ids[ids != 0][-best_length:]
    if len(ids) == 0:
        raise ValueError("Seed text must contain at least one known token.")
    X = np.zeros((1, best_length), dtype=np.int32)
    X[0, -len(ids):] = ids
    return X


def predict_next_word(seed):
    probabilities = best_model.predict(encode_context(seed), verbose=0)[0].copy()
    probabilities[:2] = 0  # Do not predict padding or [UNK].
    return vocabulary[int(np.argmax(probabilities))]


def generate_text(seed, length=50, temperature=0.8):
    if not seed.strip():
        raise ValueError("Seed text cannot be empty.")
    if length < 0 or temperature <= 0:
        raise ValueError("length must be >= 0 and temperature must be > 0.")

    generated = seed.strip()
    for _ in range(length):
        probabilities = best_model.predict(encode_context(generated), verbose=0)[0]
        probabilities[:2] = 0
        logits = np.log(np.maximum(probabilities, 1e-8)) / temperature
        logits -= np.max(logits)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()
        next_id = np.random.choice(VOCAB_SIZE, p=probabilities)
        generated += " " + vocabulary[next_id]
    return generated


print("\\nNEXT WORD:", predict_next_word("the patient"))
print("\\nGENERATED TEXT:")
print(generate_text("the patient", length=50))
print("\\nRelationship: perplexity = exp(cross-entropy); cross-entropy = ln(perplexity).")
print("Test cross-entropy:", round(test_loss, 4))
print("Test perplexity:", round(test_perplexity, 4))
