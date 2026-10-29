# 20XW96 · Deep Learning Lab

Neural-network fundamentals, representation learning, generative models, and sequence modeling in one coursework repository. Explore NumPy implementations, PyTorch and TensorFlow experiments, and an interactive RNN/LSTM lecture with actual forward passes, backpropagation, and weight updates.

[![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](.python-version)
[![Frameworks](https://img.shields.io/badge/Frameworks-PyTorch%20%7C%20TensorFlow-526779)](pyproject.toml)
[![Live lecture](https://img.shields.io/badge/Live-RNN%20%26%20LSTM%20lecture-2F8452)](https://prem-dharshan.github.io/20XW96/)

**[Open the interactive lecture](https://prem-dharshan.github.io/20XW96/)** · **[Read the LSTM guide](https://prem-dharshan.github.io/20XW96/docs.html)** · **[Browse the full index](INDEX.md)**

## Start here

| Explore | What you will find |
| --- | --- |
| [CA01 — Foundations](CA01/) | Sigmoid and softmax classifiers, a NumPy MLP, and PyTorch MLP experiments covering regression and classification. |
| [CA02 — Representations and vision](CA02/) | PCA, autoencoders, denoising, CBOW/skip-gram word embeddings, and CNN image classification. |
| [Finals — Model experiments](Finals/) | GANs, recurrent sequence layouts, character generation, report-text modeling, spam classification, and sentiment experiments. |
| [LSTM-demo — Visual lecture](LSTM-demo/) | An animated comparison of RNN and LSTM memory, gradients, and learning, with mathematical documentation. |
| [Datasets and input formats](datasets/README.md) | Included corpora, CSV schemas, and the extra data required by configurable examples. |

For individual files, model types, and data requirements, use the [repository index](INDEX.md).

## Run the Python experiments

The repository selects **Python 3.13** through `.python-version` and declares **Python >= 3.13** in `pyproject.toml`. Dependencies and their lockfile are managed with **uv**.

```powershell
git clone https://github.com/Prem-Dharshan/20XW96.git
cd 20XW96
uv sync
uv run python CA01/sigmoid.py
```

Run experiments from the **repository root** so relative paths such as `datasets/alice.txt` resolve correctly. Most scripts configure their dataset, model, and training settings through constants near the top of the file, rather than command-line arguments.

Examples:

```powershell
uv run python CA01/plainmlp.py
uv run python CA02/pca.py
uv run python Finals/rnn_mm.py
uv run python Finals/lstm.py
```

For the NLTK tokenizer used by the CBOW/skip-gram example, install its tokenizer data if needed:

```powershell
uv run python -m nltk.downloader punkt punkt_tab
uv run python CA02/cbow_sg.py
```

Some examples expect additional local image, sentiment, or translation-style data. Check the [dataset guide](datasets/README.md) before running those scripts. `main.py` is the starter greeting; the experiments are the individual scripts listed in the index.

## Explore the browser lecture

The [live lecture](https://prem-dharshan.github.io/20XW96/) runs entirely in the browser. Each visitor has an independent model and training state.

- Compare an RNN and LSTM on the same synthetic battery-warning memory task.
- Expand cells to inspect gates, or collapse them to advance one cell at a time.
- Follow forward calculations, backward gradients, and recorded parameter updates.
- Configure sequence length, hidden units, initialization, learning rate, epochs, convergence, and gradient clipping.
- Use the [Docs page](https://prem-dharshan.github.io/20XW96/docs.html) for notation tooltips and a complete worked training epoch.

To serve it locally, install Node.js and run:

```powershell
cd LSTM-demo
npm start
```

Open **http://127.0.0.1:8787**. No `npm install` or frontend build is required. You can also open `LSTM-demo/dist/index.html` directly for offline use.

## Repository layout

```text
20XW96/
├── CA01/                  # Neural-network foundations
├── CA02/                  # PCA, autoencoders, embeddings, CNNs
├── Finals/                # GAN and recurrent-model experiments
├── datasets/              # Local data and input-format guide
├── LSTM-demo/             # Static lecture, docs, and local server
├── .github/workflows/     # GitHub Pages deployment
├── INDEX.md               # File-by-file learning and experiment index
├── pyproject.toml         # Python dependencies
└── uv.lock                # Locked dependency resolution
```

## Publishing the lecture

[GitHub Actions](https://github.com/Prem-Dharshan/20XW96/actions/workflows/deploy-lstm-pages.yml) publishes `LSTM-demo/dist` to GitHub Pages when that directory changes on `main`. The deployment can also be started manually. Repository Markdown documentation is separate from the browser lecture and its Docs page.

See the [demo README](LSTM-demo/README.md) for its model assumptions, presentation controls, and training behavior. Bundled font and JavaScript-library notices are included under `LSTM-demo/dist/vendor/`.
