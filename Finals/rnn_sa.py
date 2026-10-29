import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, classification_report
)


# Variables

DATASET_PATH = "datasets/sentiment.csv"

# Set these to match your dataset.
# CSV/TSV: column name or integer index.
TEXT_COLUMN = None       # e.g. "review" or "text"
LABEL_COLUMN = None      # e.g. "sentiment" or "label"
POSITIVE_LABEL = None    # e.g. "positive"; None uses common label detection

MAX_VOCAB = 20000
MAX_LENGTHS = [50, 100, 200]
HIDDEN_UNITS = [32, 64, 128]
EMBEDDING_DIM = 64

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TRAIN_LIMIT = 50000
VAL_LIMIT = 10000
TEST_LIMIT = 10000

EPOCHS = 10
BATCH_SIZE = 128
RANDOM_STATE = 42
BEST_MODEL_PATH = "best_sentiment_model.keras"


# Dataset loading

TEXT_NAMES = ("text", "review", "sentence", "content", "message", "comment")
LABEL_NAMES = ("label", "sentiment", "target", "class", "rating", "polarity")


def clean(value):
    if value is None:
        return None
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        value = str(value).strip()
        return value if value else None
    return None


def load_records(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")

    ext = path.suffix.lower()
    if ext in (".csv", ".tsv", ".tab"):
        delimiter = "," if ext == ".csv" else "\t"
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            rows = list(csv.reader(file, delimiter=delimiter))
        if not rows:
            raise ValueError("Dataset is empty.")

        normalize = lambda s: s.strip().lower().lstrip("\\ufeff")
        first = [normalize(v) for v in rows[0]]
        has_header = (
            any(v in TEXT_NAMES + LABEL_NAMES for v in first)
            or (isinstance(TEXT_COLUMN, str) and TEXT_COLUMN.lower() in first)
            or (isinstance(LABEL_COLUMN, str) and LABEL_COLUMN.lower() in first)
        )
        headers = first if has_header else None
        data = rows[1:] if has_header else rows

        def get_column(setting, names, default):
            if isinstance(setting, int):
                return setting
            if isinstance(setting, str):
                if headers is None or setting.lower() not in headers:
                    raise ValueError(f"Column '{setting}' not found. Headers: {headers}")
                return headers.index(setting.lower())
            if headers:
                return next((headers.index(n) for n in names if n in headers), default)
            return default

        text_idx = get_column(TEXT_COLUMN, TEXT_NAMES, 0)
        label_idx = get_column(LABEL_COLUMN, LABEL_NAMES, 1)
        records = []
        for row in data:
            if max(text_idx, label_idx) >= len(row):
                continue
            text, label = clean(row[text_idx]), clean(row[label_idx])
            if text and label:
                records.append((text, label))
        return records

    if ext in (".txt", ".text"):
        records = []
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("\\t")
            if len(parts) >= 2:
                # Default format: text<TAB>label. Override indices if needed.
                ti = TEXT_COLUMN if isinstance(TEXT_COLUMN, int) else 0
                li = LABEL_COLUMN if isinstance(LABEL_COLUMN, int) else 1
                if max(ti, li) < len(parts):
                    text, label = clean(parts[ti]), clean(parts[li])
                    if text and label:
                        records.append((text, label))
            else:
                raise ValueError(
                    "Plain TXT needs one 'text<TAB>label' record per line, "
                    "or use CSV/TSV/JSON."
                )
        return records

    if ext in (".json", ".jsonl", ".ndjson"):
        if ext in (".jsonl", ".ndjson"):
            data = []
            for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as error:
                        raise ValueError(f"Invalid JSON on line {line_no}: {error}") from error
        else:
            parsed = json.loads(path.read_text(encoding="utf-8-sig"))
            data = parsed if isinstance(parsed, list) else parsed.get(
                "data", parsed.get("records", parsed.get("examples", [parsed]))
            )

        records = []
        for item in data:
            if not isinstance(item, dict):
                continue
            keys = {str(k).lower(): k for k in item}
            text_key = TEXT_COLUMN if isinstance(TEXT_COLUMN, str) else next(
                (keys[n] for n in TEXT_NAMES if n in keys), None
            )
            label_key = LABEL_COLUMN if isinstance(LABEL_COLUMN, str) else next(
                (keys[n] for n in LABEL_NAMES if n in keys), None
            )
            if isinstance(TEXT_COLUMN, int) or isinstance(LABEL_COLUMN, int):
                values = list(item.values())
                text_key = TEXT_COLUMN if isinstance(TEXT_COLUMN, int) else text_key
                label_key = LABEL_COLUMN if isinstance(LABEL_COLUMN, int) else label_key
                text = clean(values[text_key]) if isinstance(text_key, int) and -len(values) <= text_key < len(values) else clean(item.get(text_key))
                label = clean(values[label_key]) if isinstance(label_key, int) and -len(values) <= label_key < len(values) else clean(item.get(label_key))
            else:
                text, label = clean(item.get(text_key)), clean(item.get(label_key))
            if text and label:
                records.append((text, label))
        return records

    raise ValueError("Supported formats: CSV, TSV, TAB, TXT (tab-separated), JSON, JSONL, NDJSON.")


records = load_records(DATASET_PATH)
if len(records) < 10:
    raise ValueError(f"Only {len(records)} usable records found; check text/label columns.")

texts = np.array([r[0] for r in records], dtype=object)
raw_labels = np.array([r[1] for r in records], dtype=object)
classes = sorted(set(raw_labels.tolist()))

if len(classes) != 2:
    raise ValueError(f"Binary sentiment classification expects 2 labels; found {classes}")

positive = POSITIVE_LABEL
if positive is None:
    common_positive = ("positive", "pos", "1", "positive sentiment")
    positive = next((c for c in classes if c.lower() in common_positive), classes[-1])
if positive not in classes:
    raise ValueError(f"POSITIVE_LABEL={positive!r} not in labels {classes}")

label_to_id = {classes[0]: 0, classes[1]: 1}
# Force positive class to be 1 for precision/recall/F1/AUC.
label_to_id = {c: int(c == positive) for c in classes}
y = np.array([label_to_id[v] for v in raw_labels], dtype=np.int32)

print("Dataset:", DATASET_PATH)
print("Records:", len(texts), "| Labels:", classes, "| Positive label:", positive)
print("Sample:", texts[0][:160], "=>", raw_labels[0])


# Split records before vectorization; stratify manually for both classes.

rng = np.random.default_rng(RANDOM_STATE)
train_idx, val_idx, test_idx = [], [], []
for class_id in (0, 1):
    idx = np.where(y == class_id)[0]
    rng.shuffle(idx)
    a = int(len(idx) * TRAIN_SPLIT)
    b = int(len(idx) * (TRAIN_SPLIT + VAL_SPLIT))
    train_idx.extend(idx[:a])
    val_idx.extend(idx[a:b])
    test_idx.extend(idx[b:])

rng.shuffle(train_idx)
rng.shuffle(val_idx)
rng.shuffle(test_idx)

train_texts, y_train_all = texts[train_idx].tolist(), y[train_idx]
val_texts, y_val_all = texts[val_idx].tolist(), y[val_idx]
test_texts, y_test_all = texts[test_idx].tolist(), y[test_idx]

if min(len(train_idx), len(val_idx), len(test_idx)) == 0:
    raise ValueError("Not enough data for train/validation/test splits.")

vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=MAX_VOCAB, standardize="lower",
    split="whitespace", output_mode="int",
    output_sequence_length=max(MAX_LENGTHS)
)
vectorizer.adapt(tf.data.Dataset.from_tensor_slices(train_texts).batch(1024))
VOCAB_SIZE = len(vectorizer.get_vocabulary())
print("Vocabulary size:", VOCAB_SIZE)


def make_arrays(text_list, labels, max_length, limit):
    # Vectorize once to the maximum configured length, then trim to this length.
    ids = vectorizer(tf.constant(text_list)).numpy()[:, :max_length]
    if limit and len(ids) > limit:
        ids, labels = ids[:limit], labels[:limit]
    return ids.astype(np.int32), np.asarray(labels, dtype=np.int32)


def build_model(max_length, hidden_units, model_type):
    recurrent = (
        tf.keras.layers.SimpleRNN(hidden_units, activation="tanh")
        if model_type == "RNN"
        else tf.keras.layers.LSTM(hidden_units)
    )
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(max_length,), dtype="int32"),
        tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, mask_zero=True),
        recurrent,
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


# Train configurations and compare validation metrics.

results = []
best_score = -1.0
best_config = None
best_history = None

for model_type in ("RNN", "LSTM"):
    for max_length in MAX_LENGTHS:
        X_train, y_train = make_arrays(train_texts, y_train_all, max_length, TRAIN_LIMIT)
        X_val, y_val = make_arrays(val_texts, y_val_all, max_length, VAL_LIMIT)

        for units in HIDDEN_UNITS:
            print(f"\\nTraining {model_type}: units={units}, max_length={max_length}")
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(RANDOM_STATE)
            model = build_model(max_length, units, model_type)

            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1,
                callbacks=[tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=2, restore_best_weights=True
                )]
            )

            probabilities = model.predict(X_val, verbose=0).ravel()
            predictions = (probabilities >= 0.5).astype(int)
            metrics = {
                "model": model_type, "units": units, "max_length": max_length,
                "accuracy": accuracy_score(y_val, predictions),
                "precision": precision_score(y_val, predictions, zero_division=0),
                "recall": recall_score(y_val, predictions, zero_division=0),
                "f1": f1_score(y_val, predictions, zero_division=0),
                "auc": roc_auc_score(y_val, probabilities) if len(np.unique(y_val)) == 2 else float("nan"),
                "val_loss": float(model.evaluate(X_val, y_val, verbose=0)[0]),
                "epochs": len(history.history["loss"])
            }
            results.append(metrics)
            print(metrics)

            # Select by validation F1; break ties using AUC.
            score = metrics["f1"] + (0.0001 * (metrics["auc"] if np.isfinite(metrics["auc"]) else 0))
            if score > best_score:
                best_score = score
                best_config = metrics.copy()
                best_history = {k: v.copy() for k, v in history.history.items()}
                model.save(BEST_MODEL_PATH)
                print("Saved current best model.")

            del model


# Results table

results.sort(key=lambda r: (r["f1"], r["auc"]), reverse=True)
print("\\nVALIDATION COMPARISON")
print(f"{'Model':<6} {'Units':>5} {'Length':>6} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")
for r in results:
    print(f"{r['model']:<6} {r['units']:>5} {r['max_length']:>6} "
          f"{r['accuracy']:>10.4f} {r['precision']:>10.4f} {r['recall']:>10.4f} "
          f"{r['f1']:>10.4f} {r['auc']:>10.4f}")

print("\\nBEST CONFIGURATION:", best_config)


# Final test evaluation using the selected model.

best_model = tf.keras.models.load_model(BEST_MODEL_PATH)
X_test, y_test = make_arrays(test_texts, y_test_all, best_config["max_length"], TEST_LIMIT)
test_prob = best_model.predict(X_test, verbose=0).ravel()
test_pred = (test_prob >= 0.5).astype(int)

print("\\nTEST RESULTS")
print("Accuracy :", round(accuracy_score(y_test, test_pred), 4))
print("Precision:", round(precision_score(y_test, test_pred, zero_division=0), 4))
print("Recall   :", round(recall_score(y_test, test_pred, zero_division=0), 4))
print("F1-score :", round(f1_score(y_test, test_pred, zero_division=0), 4))
print("ROC-AUC  :", round(roc_auc_score(y_test, test_prob), 4))
print("\\nClassification report:")
print(classification_report(y_test, test_pred, target_names=["Negative", "Positive"], zero_division=0))


# Plots

plt.figure(figsize=(8, 4))
plt.plot(best_history["loss"], label="Training loss")
plt.plot(best_history["val_loss"], label="Validation loss")
plt.title("Best Model: Loss")
plt.xlabel("Epoch")
plt.ylabel("Binary cross-entropy")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
labels = [f"{r['model']}-{r['units']}-L{r['max_length']}" for r in results]
plt.bar(labels, [r["f1"] for r in results])
plt.title("Validation F1-score Comparison")
plt.xlabel("Model - Hidden Units - Max Length")
plt.ylabel("F1-score (higher is better)")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y")
plt.tight_layout()
plt.show()

fpr, tpr, _ = roc_curve(y_test, test_prob)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc_score(y_test, test_prob):.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("Best Model: ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


# Try a custom sentiment prediction.

def predict_sentiment(text):
    probability = float(best_model.predict(
        vectorizer(tf.constant([text]))[:, :best_config["max_length"]],
        verbose=0
    )[0, 0])
    label = "Positive" if probability >= 0.5 else "Negative"
    return {"sentiment": label, "positive_probability": round(probability, 4)}


print("\\nCUSTOM PREDICTION")
print(predict_sentiment("This product is excellent and I really enjoyed it."))
