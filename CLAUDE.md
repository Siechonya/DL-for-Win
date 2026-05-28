# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Plasma physics coherent structure classification using a BiAutoencoder (1D-CNN + Bi-LSTM) with physical contrastive learning. 4-channel magnetic field time series → 64-dim latent space → 8-class prediction (sheet, vortex chain, c vortex, l vortex, hole, soliton, shock, alfen dis) + noise rejection via adaptive thresholds.

## Repository is notebook-based

All code lives in Jupyter notebooks. There is no `setup.py`, `requirements.txt`, build step, or test suite. Notebooks must be run in order (cells depend on prior cells). Notebooks **cannot** be imported as modules — code reuse is done by copying cells or running the full notebook.

## Key notebooks

- **`src/train_20240101-0130.ipynb`** — Main training pipeline (Euclidean distance). Defines `load_data`, model, loss functions, `test_clustering`, training loop, t-SNE, and all visualization cells. Run top-to-bottom.
- **`src/train_20240101-0130_cosine.ipynb`** — Cosine distance variant. Same architecture but `physical_contrastive_loss` and `test_clustering` use cosine distance (L2-normalized embeddings + `cdist`). Keeps input normalization to keep MSE channels balanced.
- **`src/prediction_for_arbitrary_df.ipynb`** — Loads trained weights and runs inference on new data. Defines `PhysicalPredictor` with `distance_metric` parameter (`'euclidean'` or `'cosine'`). Includes batch prediction runner with tqdm and visualization cells. Self-contained after first run.
- **`src/clean_sample.ipynb`** — Preprocessing for prototype samples.

## Architecture decisions that matter

- **No contrastive warmup**: `start_lambda_contrastive = max_lambda_contrastive = 0.1`. Warmup caused the decoder to lock into a reconstruction-only latent space; contrastive loss couldn't recover. Both losses co-evolve from epoch 0.
- **Normalization in `load_data`**: B channel centered then divided by max; perturbations divided by global max. This is a linear transform — required for balanced MSE channels, harmless for cosine distance.
- **`max_shift=50` in `calc_invariant_mse`**: Intentional large tolerance for translation+reflection invariance (structures aren't centered in the time window).
- **Physical features**: 10 summary statistics (polarization ratio, compressibility index, correlation, kurtosis, etc.) computed by `extract_physical_features_batch`. These inform the contrastive loss's pair-masking logic via similarity thresholds.
- **Classification**: Prototype-center nearest-neighbor with per-class dynamic thresholds (`mean + n_std * std`). Unknown/noise samples assigned `'neither'` if distance exceeds threshold.
- **Training data**: 48,573 parquet files from `trainset_*` folders. Only prototypes (31 raw × 2 augmented = 62) have labels. Training samples are unlabeled (`labels = -1` in contrastive loss).
- **Test data has no labels** — evaluation is qualitative (t-SNE, top-K waveform inspection, distance histograms).

## Data format

Each `.parquet` file is a time series with columns: `B`, `b_z`, `b_max`, `b_min`. Prototype samples live in `samples_clean/<class_name>/`. Training data in `trainset_*` folders (git-ignored). Adding a new class requires: (1) folder in `samples_clean`, (2) add name to `classes` list in training notebooks.

## Model I/O convention

- **Input**: `[batch, 300, 4]` — interpolated to 300 time steps, 4 channels
- **Latent**: `[batch, 64]` — from `BiAutoencoder.encode()`
- **Output**: `[batch, 300, 4]` — reconstructed waveform
- Saved artifacts: `bi_model.pth` (state_dict), `proto_emb.npy` (dict of class→center_vector), `thresholds.npy` (dict of class→float), `target_pts.npy` (array with one element: 300)
