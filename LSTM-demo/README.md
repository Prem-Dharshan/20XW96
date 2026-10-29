# Battery Memory Lab — RNN vs LSTM

Open `dist/index.html` in Chrome or Edge. All assets are local, including Nunito and Rough.js, so no build, server, or internet connection is needed. The prior color-memory lesson remains available at `dist/color-memory.html`.

The navigation's **Docs** item opens `dist/docs.html`: detailed gate equations and intuition, dimensions and notation, a step-by-step backward pass, gradient accumulation, clipping, SGD, and a complete numerical epoch. Hover, keyboard focus, or tap a notation to see its definition. The worked epoch independently calculates both examples, all 32 time steps per example, and all 22 scalar parameter updates using the simulation's model code. KaTeX and its fonts are bundled locally.

The synthetic task is to remember an abnormal heating precursor through a long run of normal readings, then make an inspection decision. This is a simplified sequence-memory lesson, not a measured battery temperature model.

## Presenting

- Forward pass shows the RNN state and LSTM gates side by side. The first two cells start expanded. Select any time step and expand or collapse it to change the stepping.
- Backward pass traverses the actual sample-loss derivatives in reverse time. Expanded LSTM cells show the derivatives of every gate, propagation to both previous states, and accumulation of shared-parameter gradients.
- Weight updates show the averaged two-example batch gradient, its clipping factor, and actual SGD before/after values. Before a training step, they are clearly marked as a preview. After a step, they show the last recorded update. Next/Previous traverses parameter groups.
- Apply one epoch performs one update per model, up to the configured epoch cap. Run configured epochs trains each model until its loss target, plateau stopping criterion, numerical divergence, or epoch limit is reached.
- Hide controls expands the canvas. Fullscreen opens the lecture for presenting. Drag to pan, scroll to zoom, and use Focus step after an overview. Arrow keys step through the pass; Space plays/pauses; F toggles fullscreen.

## Configurable settings

Sequence gap (2–100), hidden units (1–4), recurrent initialization, LSTM forget-bias initialization, inspection threshold, learning rate, maximum epochs, loss target, plateau patience, minimum loss change, and clipping norm. Apply settings resets the parameters and training history to make runs reproducible.

Input size is 3 and recurrent depth is 1. Both models train on one healthy and one warning sequence per epoch, with equal weighting. All weights and biases are updated, including recurrent matrices and readout parameters.

## Default demonstration

30 normal readings, 1 hidden unit, SGD learning rate 0.04, maximum 80 epochs, loss target 0.08, clipping norm 1, inspection threshold 0.65. Initial parameters are chosen to reveal the memory mechanism. With these defaults the RNN misses the early warning, whereas the LSTM retains it. After configured training, RNN average loss is approximately 0.7116 at epoch 80; LSTM reaches average loss approximately 0.0798 at epoch 78. Outcomes are calculated from current parameters and can change when settings change.

The exploding-gradient preset uses the healthy control and recurrent initialization 1.35, so a zero, unsaturated RNN state has an amplifying backward chain. It starts with clipping disabled. The long-memory preset illustrates forgetting and vanishing gradients. These are distinct controlled settings of the same task; LSTM is not represented as a universal cure for exploding gradients.

## Mathematics and verification

`battery-model.js` contains vector forward passes, full backpropagation through time, full-batch gradient averaging, global gradient-norm clipping, and SGD. The green path highlights the direct LSTM cell-state route; the actual backward calculation also includes indirect recurrent gate paths.

Analytic gradients were checked against centered finite differences for every parameter in both two-unit models. Maximum absolute errors were below 1e-8. Browser checks covered default training outcomes, early stopping, expansion/collapse, forward/backward/updates, exploding gradients, configurable hidden dimensions, sidebar toggling, and mobile layout.

Local font/library license notices are in `dist/vendor/`.

## Run from this folder

```powershell
cd D:\Academics\20XW96\LSTM-demo
npm start
```

Open http://127.0.0.1:8787 in your browser. Node.js is the only requirement; no npm install is necessary.

To share with ngrok, keep the server running and use a second terminal:

```powershell
ngrok http http://127.0.0.1:8787
```

Alternatively, open dist/index.html directly for offline use.

## GitHub Pages

This demo is part of https://github.com/Prem-Dharshan/20XW96.

Live lecture: https://prem-dharshan.github.io/20XW96/

Documentation: https://prem-dharshan.github.io/20XW96/docs.html

The repository-root workflow .github/workflows/deploy-lstm-pages.yml publishes LSTM-demo/dist. Pushing changes to that directory on main redeploys the lecture. It can also be run manually from Actions. No build or npm install is needed.

To publish future edits:

```powershell
cd D:\Academics\20XW96
git add LSTM-demo .github/workflows/deploy-lstm-pages.yml
git commit -m "Update LSTM lecture"
git push
```

Local server logs and process IDs are ignored. GitHub Pages runs the static app directly in each visitor's browser, independently of the local server and ngrok.