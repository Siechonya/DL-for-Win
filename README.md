# DL for Win

Deep learning framework for plasma physics coherent structure classification. A BiAutoencoder (1D-CNN + Bi-LSTM) with physical contrastive learning identifies coherent structures from 4-channel magnetic field time series. 64-dim latent space → 9-class prediction (8 structure types + noise rejection via adaptive thresholds).

## Project structure

```
├── src/
│   ├── train_20240101-0130.ipynb / .py        # Training pipeline (Euclidean distance)
│   ├── train_20240101-0130_cosine.ipynb / .py # Training pipeline (cosine distance variant)
│   ├── prediction_for_arbitrary_df.ipynb / .py# Inference on arbitrary data
│   └── clean_sample.ipynb / .py               # Sample data preprocessing
├── samples_clean/                              # Labeled prototype waveforms
├── output/images/eu/                           # Training output visualizations
├── bi_model.pth / proto_emb.npy                # Trained model weights / prototype embeddings
├── thresholds.npy / target_pts.npy            # Classification thresholds / interpolation pts
└── trainset*/                                 # Training datasets (git-ignored)
```

> Each `.ipynb` has a paired `.py` file (percent format, `# %%` cell separators) managed by [Jupytext](https://github.com/mwouts/jupytext). Edit `.py` files for safer code changes — sync back with `jupytext --sync <file>.py`.

## Model architecture

- **1D-CNN frontend** — extracts local gradient and spike features (kernel=5, padding=2)
- **Bi-LSTM encoder** — 2-layer, hidden=128, bidirectional, captures temporal dependencies
- **Mixed pooling** (max + mean) — preserves both peak signatures and background trends → 512-dim
- **FC reduction** — 512 → 64-dim latent space
- **Bi-LSTM decoder** — reconstructs 4-channel input from 64-dim latent

## Training approach

**Hybrid loss**: `loss = (1-λ) × reconstruction_MSE + λ × contrastive`

- **Reconstruction**: shift-invariant MSE (±50 points, forward + flipped), weighted per-channel (B:0.5, b_z:2.0, b_max,b_min:1.0)
- **Physical contrastive loss**: 18 hand-crafted physical features (z-score normalized per-batch) gate positive/negative pairs in the latent space via similarity thresholds
- **`λ = 0.05`** — no warmup, both losses co-evolve from epoch 0
- **Feature gating**: soft gates with overlap zone — `mask_alfven` (`comp_index < 1`) / `mask_comp` (`comp_index > 0.707`). Intermediate structures (0.707~1.0) activate both feature groups
- **Classification**: prototype-center nearest-neighbor with per-class dynamic thresholds (`mean + n_std × std`). Unknown/noise → `'neither'`
- **Training data**: ~70K samples, unlabeled. Prototypes (31 raw × 2 augmented = 62) are the only labeled samples

## Classification targets

| Class | Description |
| --- | --- |
| `sheet` | Current sheet |
| `vortex chain` | Chain of magnetic vortices |
| `c vortex` | Circular polarized vortex |
| `l vortex` | Linear polarized vortex |
| `hole` | Magnetic hole |
| `soliton` | Magnetic soliton |
| `shock` | Shock / shock-like structure |
| `alfen dis` | Alfvén discontinuity |
| `noise` | Rejected / unclassified |

## Usage

### Prediction

```python
predictor = PhysicalPredictor(
    model_path='bi_model.pth',
    proto_emb_path='proto_emb.npy',
    thresholds_path='thresholds.npy',
)

df = pd.read_parquet('your_data.parquet')
label, distance, details, is_neither = predictor.predict(df)
```

### Batch prediction (with OOM protection)

```python
results = run_batch_predictions(predictor, 'trainset/', fraction=0.1)
```

## Environment

| Component | Detail |
| --- | --- |
| OS | Windows 11 Pro |
| Python | 3.12.3 (conda env `dplearn`) |
| GPU | NVIDIA GeForce RTX 5060 (8 GB VRAM), CUDA 13.2 |
| Shell | PowerShell |
| Jupytext | Dedicated conda env `jupytext` (minimal, no ML deps) |
| Sync command | `jupytext --sync src/<file>.py` |

## Key dependencies

- PyTorch (CUDA), pandas, numpy
- pyarrow (parquet I/O), scipy (cdist)
- tqdm, matplotlib, scikit-learn (t-SNE)
