# Repository index

[Repository overview](README.md) · [Live RNN/LSTM lecture](https://prem-dharshan.github.io/20XW96/) · [Mathematical guide](https://prem-dharshan.github.io/20XW96/docs.html) · [Dataset guide](datasets/README.md)

Use this index to choose an experiment by topic and see what input it needs. Python commands should be run from the repository root after `uv sync`.

## Suggested learning path

1. Train a [sigmoid classifier](CA01/sigmoid.py), then a [softmax classifier](CA01/softmax.py).
2. Inspect the [NumPy MLP](CA01/plainmlp.py) and compare it with the [PyTorch experiments](CA01/mlp.py).
3. Explore [PCA](CA02/pca.py), [autoencoding](CA02/ae.py), and [denoising](CA02/dae.py).
4. Study [word embeddings](CA02/cbow_sg.py) and [CNN image classification](CA02/cnn_full.py).
5. Compare the [recurrent sequence layouts](#recurrent-sequence-layouts) and the [character-level LSTM](Finals/lstm.py).
6. Use the [interactive lecture](https://prem-dharshan.github.io/20XW96/) to connect recurrent equations, memory, gradients, and training.
7. Explore the [application experiments](#sequence-applications) and [GANs](#generative-adversarial-networks).

## Foundations · CA01

| File | Focus | Input |
| --- | --- | --- |
| [sigmoid.py](CA01/sigmoid.py) | NumPy sigmoid classifier, binary cross-entropy, gradient-based training, metrics, and loss plots. | Small in-script dataset. |
| [softmax.py](CA01/softmax.py) | NumPy softmax classifier, categorical cross-entropy, training, and evaluation. | Small in-script dataset. |
| [plainmlp.py](CA01/plainmlp.py) | NumPy MLP with hidden activation, binary output, training, and evaluation. | Small in-script dataset. |
| [mlp.py](CA01/mlp.py) | PyTorch MLP experiments for linear/nonlinear regression, binary/multiway classification, and batch/SGD/mini-batch training. | In-script examples. |
| [portions.md](CA01/portions.md) | Assessment topic notes and evaluation measures. | Reading notes. |

## Representations and vision · CA02

| File | Focus | Input |
| --- | --- | --- |
| [pca.py](CA02/pca.py) | PCA via centering, covariance, eigenvectors, component selection, and explained variance. | `datasets/deep_learning_lab_test_dataset.csv`; expects `target`. |
| [ae.py](CA02/ae.py) | Dense autoencoder, learned encoding, reconstruction, and reconstruction error. | Same tabular CSV; excludes the configured target from features. |
| [dae.py](CA02/dae.py) | Denoising autoencoder trained on noisy-to-clean inputs, plus classification comparisons. | Same tabular CSV. |
| [cbow_sg.py](CA02/cbow_sg.py) | NumPy CBOW and skip-gram word embeddings with similarity comparisons. | In-script text; NLTK tokenizer data. |
| [cnn.py](CA02/cnn.py) | CNN image classification with evaluation and learning curves. | Configured image folder or CSV; default path `dataset` is not included. |
| [cnn_full.py](CA02/cnn_full.py) | Expanded CNN configuration, flexible layer settings, data loading, metrics, and plots. | Class-subfolder images or a CSV containing `filepath` and `label`. |

## Recurrent sequence layouts

The names `oo`, `om`, `mo`, and `mm` refer to one-to-one, one-to-many, many-to-one, and many-to-many input/output layouts. These scripts use character-level text examples.

| File | Focus | Default input |
| --- | --- | --- |
| [rnn_oo.py](Finals/rnn_oo.py) | One-to-one RNN: one character predicts the following character. | `datasets/shakespeare.txt` |
| [rnn_om.py](Finals/rnn_om.py) | One-to-many RNN sequence prediction and text generation. | `datasets/shakespeare.txt` |
| [rnn_mo.py](Finals/rnn_mo.py) | Many-to-one RNN: context sequence predicts the next character. | `datasets/shakespeare.txt` |
| [rnn_mm.py](Finals/rnn_mm.py) | Many-to-many RNN with shifted character sequences, evaluation, and sampled generation. | `datasets/alice.txt` |
| [lstm.py](Finals/lstm.py) | Character-level LSTM with shifted sequence targets and text generation. | `datasets/alice.txt` |

Example commands:

```powershell
uv run python Finals/rnn_mm.py
uv run python Finals/lstm.py
```

## Sequence applications

| File | Focus | Default input / preparation |
| --- | --- | --- |
| [rnn_spam.py](Finals/rnn_spam.py) | SimpleRNN spam classification, configuration comparisons, and classification metrics. | Included `datasets/spam.csv`. |
| [lstm_spam.py](Finals/lstm_spam.py) | LSTM spam classification across sequence lengths and hidden-unit settings. | Included `datasets/spam.csv`. |
| [rnn_mammogram.py](Finals/rnn_mammogram.py) | Report-text next-word prediction and generation; compares SimpleRNN and LSTM configurations. | Included semicolon-delimited `datasets/mammogram.csv`, column `Full_Report`. |
| [lstm_mammogram.py](Finals/lstm_mammogram.py) | Report-text modeling and SimpleRNN/LSTM comparisons. | Same report CSV and text column. |
| [rnn_sa.py](Finals/rnn_sa.py) | Sentiment classification with SimpleRNN/LSTM comparisons and validation-based selection. | Supply `datasets/sentiment.csv` or configure the loader and label mapping; default file is not included. |
| [rnn_mt.py](Finals/rnn_mt.py) | Next-word prediction and generation on selected target-language text from translation-style records; compares SimpleRNN and LSTM. | Supply `datasets/translation.tsv` or configure its supported text/record loader; default file is not included. |

The mammogram scripts consume **report text**. The translation-style experiment models selected text through next-word prediction. Read the configuration and preprocessing sections to understand the exact task before interpreting results.

## Generative adversarial networks

| File | Focus | Input |
| --- | --- | --- |
| [gan.py](Finals/gan.py) | Dense generator/discriminator training, loss curves, and generated image display. | Fashion-MNIST, loaded through Keras; may download on first use. |
| [gan_full.py](Finals/gan_full.py) | GAN with configurable custom-image loading, image dimensions, model sizes, and training settings. | Supply `datasets/images`; loader supports label sources from subdirectories, filenames, or `datasets/labels.csv`. |

## Interactive RNN/LSTM lecture

**[Launch the lecture](https://prem-dharshan.github.io/20XW96/)** · **[Open the Docs page](https://prem-dharshan.github.io/20XW96/docs.html)**

The browser experiment compares memory and learning on a synthetic battery-warning sequence. It includes expandable cells, forward/backward passes, weight-update inspection, training curves, configurable settings, and a worked numerical epoch with notation tooltips.

| File | Responsibility |
| --- | --- |
| [LSTM-demo/README.md](LSTM-demo/README.md) | Presentation controls, model assumptions, setup, and deployment notes. |
| [index.html](LSTM-demo/dist/index.html) | Main lecture page. |
| [battery-app.js](LSTM-demo/dist/battery-app.js) | Network drawings, animations, navigation, and controls. |
| [battery-model.js](LSTM-demo/dist/battery-model.js) | Numerical RNN/LSTM forward passes, backpropagation, gradient clipping, and SGD. |
| [battery.css](LSTM-demo/dist/battery.css) | Lecture styling. |
| [docs.html](LSTM-demo/dist/docs.html), [docs.js](LSTM-demo/dist/docs.js), [docs.css](LSTM-demo/dist/docs.css) | Mathematical guide, notation tooltips, and calculated epoch tables. |
| [color-memory.html](LSTM-demo/dist/color-memory.html) | Earlier color-memory lesson; uses [app.js](LSTM-demo/dist/app.js) and [excalidraw.css](LSTM-demo/dist/excalidraw.css). |
| [serve.cjs](LSTM-demo/serve.cjs) | Optional local static server on port 8787. |
| [package.json](LSTM-demo/package.json) | Local `npm start` command; no frontend dependencies to install. |
| [vendor/](LSTM-demo/dist/vendor/) | Bundled fonts, Rough.js, KaTeX, and third-party notices. |
| [deploy-lstm-pages.yml](.github/workflows/deploy-lstm-pages.yml) | Deploys `LSTM-demo/dist` to GitHub Pages. |

## Data and environment

| Location | Purpose |
| --- | --- |
| [datasets/README.md](datasets/README.md) | Dataset inventory and expected formats. |
| [pyproject.toml](pyproject.toml) | Project metadata, Python requirement, and dependencies. |
| [uv.lock](uv.lock) | Locked dependency resolution. |
| [.python-version](.python-version) | Selected Python version: 3.13. |
| [main.py](main.py) | Starter greeting; experiments run through their individual scripts. |
| [.gitignore](.gitignore), [demo ignore rules](LSTM-demo/.gitignore) | Ignore local environments, generated files, logs, and process IDs; retain the demo's static source. |

Return to the [repository overview](README.md).
