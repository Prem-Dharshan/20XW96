# Datasets and input formats

[Repository overview](../README.md) · [Experiment index](../INDEX.md)

Run Python experiments from the repository root. Most paths below are relative to that directory. Dataset paths and relevant columns are configured near the top of each script.

## Included files

| File | Format / role | Used by |
| --- | --- | --- |
| [deep_learning_lab_test_dataset.csv](deep_learning_lab_test_dataset.csv) | Tabular CSV with `age`, `salary`, `experience`, `department`, `performance`, and `target`. | PCA, autoencoder, and denoising-autoencoder examples in `CA02`. |
| [alice.txt](alice.txt) | Character-level text corpus. | `Finals/rnn_mm.py` and `Finals/lstm.py`. |
| [shakespeare.txt](shakespeare.txt) | Character-level text corpus. | `Finals/rnn_oo.py`, `rnn_om.py`, and `rnn_mo.py`. |
| [spam.csv](spam.csv) | CSV whose source columns are `v1` and `v2`; spam scripts normalize the label/text names and read with Latin-1 encoding. | `Finals/rnn_spam.py` and `Finals/lstm_spam.py`. |
| [mammogram.csv](mammogram.csv) | Semicolon-separated CSV; the report-text scripts use `Full_Report`. | `Finals/rnn_mammogram.py` and `Finals/lstm_mammogram.py`. |
| [mammogram_text.txt](mammogram_text.txt) | Additional report-text corpus. | Available for experiments; not the default CSV input of the report-text scripts. |
| [spam_text.txt](spam_text.txt) | Additional text corpus. | Available for experiments; spam classifiers use `spam.csv` by default. |

The report-text examples perform sequence modeling on text. Their file names do not imply mammogram-image classification.

## Inputs you need to supply

| Experiment | Default expected path | Preparation |
| --- | --- | --- |
| `CA02/cnn.py`, `CA02/cnn_full.py` | `dataset` at the repository root | Class-subfolder images, or change `DATA_PATH` to a CSV containing `filepath` and `label`. The configured default folder is not included. |
| `Finals/gan_full.py` | `datasets/images` | Custom images. Select `LABEL_SOURCE` as `subdir`, `filename`, or `csv`; CSV mode also expects the configured label file and column names. |
| `Finals/rnn_sa.py` | `datasets/sentiment.csv` | Labeled text records. Configure the text column, label column, and positive-label mapping for your data. |
| `Finals/rnn_mt.py` | `datasets/translation.tsv` | Translation-style records or another supported local text format. Configure the selected target column or language key. |

An example class-subfolder layout is:

```text
dataset/
├── class_a/
│   └── image_001.jpg
└── class_b/
    └── image_002.jpg
```

For a CNN CSV, the expected columns are:

```csv
filepath,label
dataset/class_a/image_001.jpg,class_a
dataset/class_b/image_002.jpg,class_b
```

The listed images are illustrative paths, not bundled samples.

## Other data sources

- `Finals/gan.py` loads Fashion-MNIST through `tf.keras.datasets`; its first run may download the dataset.
- The `CA01` classifier examples use small datasets defined directly in their scripts.
- `CA02/cbow_sg.py` uses in-script text and NLTK tokenization. Tokenizer resources can be installed with `uv run python -m nltk.downloader punkt punkt_tab`.
- The browser lecture generates its warning and healthy sequences locally and does not read these dataset files.

To identify the task and preprocessing used by a particular file, start with the [full index](../INDEX.md) and then its configuration section.
