# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Plasma physics coherent structure classification using a BiAutoencoder (1D-CNN + Bi-LSTM) with physical contrastive learning. 4-channel magnetic field time series → 64-dim latent space → 8-class prediction (sheet, vortex chain, c vortex, l vortex, hole, soliton, shock, alfen dis) + noise rejection via adaptive thresholds.

## Repository is notebook-based

All code lives in Jupyter notebooks. There is no `setup.py`, `requirements.txt`, build step, or test suite. Notebooks must be run in order (cells depend on prior cells). Notebooks **cannot** be imported as modules — code reuse is done by copying cells or running the full notebook.

**Jupytext pairing**: Every `.ipynb` has a synced `.py` file in `percent` format (`# %%` cell separators). VS Code Jupytext extension auto-syncs on save. **Prefer reading/editing `.py` files** — they are plain Python, avoiding JSON escaping bugs. After editing `.py`, jupytext auto-updates `.ipynb` (or run `jupytext --sync <file>` manually).

## Key notebooks

- **`src/train_20240101-0130.ipynb`** — Main training pipeline (Euclidean distance). Defines `load_data`, model, loss functions, `test_clustering`, training loop, t-SNE, and all visualization cells. Run top-to-bottom.
- **`src/train_20240101-0130_cosine.ipynb`** — Cosine distance variant. Same architecture but `physical_contrastive_loss` and `test_clustering` use cosine distance (L2-normalized embeddings + `cdist`). Keeps input normalization to keep MSE channels balanced.
- **`src/prediction_for_arbitrary_df.ipynb`** — Loads trained weights and runs inference on new data. Defines `PhysicalPredictor` with `distance_metric` parameter (`'euclidean'` or `'cosine'`). Includes batch prediction runner with tqdm and visualization cells. Self-contained after first run.
- **`src/clean_sample.ipynb`** — Preprocessing for prototype samples.

## Architecture decisions that matter

- **No contrastive warmup**: `start_lambda_contrastive = max_lambda_contrastive = 0.1`. Warmup caused the decoder to lock into a reconstruction-only latent space; contrastive loss couldn't recover. Both losses co-evolve from epoch 0.
- **Normalization in `load_data`**: B channel centered then divided by max; perturbations divided by global max. This is a linear transform — required for balanced MSE channels, harmless for cosine distance.
- **`max_shift=50` in `calc_invariant_mse`**: Intentional large tolerance for translation+reflection invariance (structures aren't centered in the time window).
- **Physical features**: 18 features computed by `extract_physical_features_batch` (see table below). Gating masks (`mask_alfven` / `mask_comp`) zero out irrelevant features per structure type. Features are z-score normalized inside `physical_contrastive_loss` and used for pair-masking via similarity thresholds (<0.8 similar, >4.0 dissimilar).
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

## Physical feature groups (18 features, 0-indexed)

Features computed by `extract_physical_features_batch` and returned in fixed order. Comments in code use 0-based indices matching the return order for direct cross-reference with `alfven_indices`.

| idx | feature | group | gate |
| --- | --- | --- | --- |
| 0 | pol_ratio | Alfven | mask_alfven |
| 1 | comp_index | Compressible | mask_comp |
| 2 | bz_dip | Compressible | mask_comp |
| 3 | B_dip | Compressible | mask_comp |
| 4 | corr_bmax_bmin | Alfven | mask_alfven |
| 5 | dom_freq | Alfven | mask_alfven |
| 6 | max_grad_bz | Compressible | mask_comp |
| 7 | R_jump | Compressible | mask_comp |
| 8 | peakiness_dot_bmax | Alfven | mask_alfven |
| 9 | b_max_flipscore | Alfven | mask_alfven |
| 10 | kurt_dot_B | Compressible | mask_comp |
| 11 | kurt_dot_bz | Compressible | mask_comp |
| 12 | complexity_index_bz | Compressible | mask_comp |
| 13 | complexity_index_bmax | Alfven | mask_alfven |
| 14 | corr_shock_B | Compressible | mask_comp |
| 15 | abs_skew_grad_B | Compressible | mask_comp |
| 16 | abs_skew_grad_bz | Compressible | mask_comp |
| 17 | abs_skew_grad_bmax | Alfven | mask_alfven |

`alfven_indices = [0, 4, 5, 8, 9, 13, 17]` — used by `extract_physical_features_batch` for gate assignment. Alfven features are gated by `mask_alfven` (`comp_index < 1.0`), compressible features by `mask_comp` (`comp_index > 0.5`).

`int_dot_bz_window` (formerly index ~11) was removed — its computation code is deleted, not just commented out.

The contrastive loss (`physical_contrastive_loss` in cell `e54b11c5`) is a plain function with NO learnable weights. Physical features are z-score normalized, then Euclidean distance in physical space gates positive/negative pairs (thresholds: <0.8 similar, >4.0 dissimilar). This is intentionally simple — learnable feature weights were attempted but introduced instability without proportional benefit.

## Environment

- **OS**: Windows 11 Pro (NOT WSL, NOT pure Linux)
- **Shell**: Use **PowerShell** for Python scripting and notebook modifications. Bash is available but its Python environment is unreliable (exit code 49, encoding issues). Prefer PowerShell `python -c "..."` or `python script.py`.
- **Conda**: `D:\Anaconda\envs\dplearn` (Python 3.12.3, PyTorch with CUDA 13.2)
- **GPU**: NVIDIA GeForce RTX 5060 (8 GB VRAM), driver 596.21, WDDM mode
- **Git remote**: `git@github.com:Siechonya/DL-for-Win.git`, branch `main`
- **Kernel**: `dplearn` registered at `C:\Users\predor\AppData\Roaming\jupyter\kernels\dplearn\kernel.json`

## Code editing rules (CRITICAL)

**Default workflow: sync .ipynb → .py, edit .py, sync .py → .ipynb.** Jupytext pairing keeps them in sync bidirectionally.

**Step 1 — before editing**: Sync .ipynb to .py to get the latest notebook changes:
```
jupytext --sync src\<file>.ipynb
```
**Step 2**: Edit the `.py` file with the Edit tool (plain Python, no JSON escaping).
**Step 3 — after editing**: Sync .py back to .ipynb:
```
jupytext --sync src\<file>.py
```

The jupytext executable is `D:\Anaconda\envs\jupytext\python.exe -m jupytext`.

1. **`.py` editing — NEVER `replace_all` on trivial strings**: `.py` files use `# %%` as cell separators — these are critical markers that jupytext uses to split cells. NEVER do `replace_all: true` with trivial strings like `[]`, `:`, `#`, or short variable names — they appear in hundreds of innocuous locations and will corrupt the entire file. Always use context-rich, unique `old_string` patterns (at least 2-3 lines, with surrounding code) and keep `replace_all: false` unless you are absolutely certain the match only occurs in the intended locations.

2. **After editing `.py`, sync to `.ipynb`**: `& 'D:\Anaconda\envs\jupytext\python.exe' -m jupytext --sync src\<file>.py`

3. **When things go wrong**: `git checkout -- <file>` to restore from the last commit. The repo is at commit `c406543` ("Revert global phys_feat_std normalization") pushed to remote. All local uncommitted changes are experimental.

### Legacy: direct `.ipynb` editing (only when `.py` is unavailable)

4. **Never use `NotebookEdit`** on cells larger than ~20 lines — it replaces the ENTIRE cell content, not a portion.

5. **Always verify JSON validity** after any notebook edit: `python -c "import json; json.load(open('path.ipynb'))"`

6. **`\n` escaping — the #1 cause of notebook corruption**: In modification scripts, use `"\\n"` (double-backslash) to produce literal `\n` in source code. json.dump handles the rest. NEVER use `"\n"` inside a Python script that writes notebook source — it becomes a real newline and corrupts JSON.

7. **Surgical editing pattern** (when .ipynb editing is unavoidable):
```python
import json
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)
for cell in nb['cells']:
    if cell.get('id') == 'TARGET_ID':
        source = cell['source']
        new_source = []
        for line in source:
            if 'old pattern' in line:
                line = line.replace('old', 'new')
            new_source.append(line)
        cell['source'] = new_source
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
```

## GPU memory

- **Training**: `batch_size=128` is stable and does NOT cause OOM. `cdist(emb, emb)` inside contrastive loss is manageable at this size.
- **Prediction OOM**: `run_batch_predictions` in `prediction_for_arbitrary_df.ipynb` was calling `predict(df)` then `encode(df)` separately — double the GPU inferences per sample with no `torch.cuda.empty_cache()`. Fixed by: (1) `predict(df, return_embedding=True)` returns the embedding to avoid double `encode()`, (2) `torch.cuda.empty_cache()` every 500 samples, (3) batch prediction no longer leaks GPU memory.
- **Zombie VRAM**: After OOM crash on Windows WDDM, GPU memory may not release until reboot. Check with `nvidia-smi`.
- **Kernel hang on startup**: If `dplearn` Python process hangs (e.g., torch tries to init CUDA on a zombie GPU), the Jupyter kernel will time out. Symptoms: "Failed to start the Kernel", "timeout waiting for ports". Fix: reboot, or kill zombie python processes + reinstall kernel spec via base conda Python.
