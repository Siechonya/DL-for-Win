# DL for Win

Deep learning framework for plasma physics coherent structure classification. Two encoder architectures (Bi-LSTM and CNN+Transformer) with physical contrastive learning identify coherent structures from 4-channel magnetic field time series. 64-dim latent space → 8-class prediction + noise rejection via adaptive thresholds.

## Project structure

```
├── src/
│   ├── train.ipynb / .py                          # Training pipeline (Bi-LSTM encoder)
│   ├── train_cnn_transformer.ipynb / .py          # Training pipeline (Transformer encoder)
│   ├── prediction_for_arbitrary_df.ipynb / .py   # Inference (Bi-LSTM weights)
│   ├── prediction_for_arbitrary_df_transformer... # Inference (Transformer weights)
│   └── clean_sample.ipynb / .py                   # Prototype feature analysis
├── samples_clean/                                  # Labeled prototype waveforms
├── output/images/{timestamp}/                      # Per-run training output images
├── output/train_result/{timestamp}/                # Per-run classification results
├── bi_model.pth / proto_emb.npy                    # Trained model weights / prototype embeddings
├── thresholds.npy / target_pts.npy                # Classification thresholds / interpolation pts
└── trainset/                                       # Training datasets (git-ignored)
```

> Each `.ipynb` has a paired `.py` file (percent format, `# %%` cell separators) managed by [Jupytext](https://github.com/mwouts/jupytext). Edit `.py` files for safer code changes — sync with `jupytext --sync <file>.py`.

## Model architecture

**Bi-LSTM variant (`train`)**: 1D-CNN → Bi-LSTM encoder → Mixed Pooling → 64-dim latent → Bi-LSTM decoder
**Transformer variant (`train_cnn_transformer`)**: 1D-CNN → +Positional Encoding → 3-layer Transformer encoder (d=20, nhead=4) → Mixed Pooling → 64-dim latent → Bi-LSTM decoder

Both share the same decoder and loss design. Transformer's self-attention excels at global structure comparison (shock vs soliton); Bi-LSTM excels at local phase/pattern recognition.

## Training approach

**Hybrid loss**: `loss = (1-λ) × reconstruction_MSE + λ × contrastive`

- **Reconstruction**: shift-invariant MSE (±50 points, forward + flipped), weighted per-channel (B:0.5, b_z:2.0, b_max,b_min:1.0)
- **Physical contrastive loss**: 18 hand-crafted physical features (z-score normalized per-batch) gate positive/negative pairs in the latent space via similarity thresholds (<0.8 similar, >4.0 dissimilar). Features include asym_B (shock detection), comp_index, kurtosis, skewness, etc.
- **`λ = 0.05`**, margin = 1.0 — no warmup, both losses co-evolve from epoch 0
- **Feature gating**: `mask_alfven` (`comp_index < 1`) / `mask_comp` (`comp_index > 0.5`). 4 features unmasked, 14 gated
- **Classification**: prototype-center nearest-neighbor with per-class dynamic thresholds (`mean + n_std × std`). Unknown/noise → `'neither'`
- **Training data**: ~70K samples (2023–2024, 2 days/month), unlabeled. Prototypes (31 raw × 2 augmented = 62) are the only labeled samples

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
| `alfven dis` | Alfvén discontinuity |
| `neither` | Rejected / unclassified |

## Usage

### Prediction (Bi-LSTM)

```python
predictor = PhysicalPredictor(
    model_path='bi_model.pth',
    proto_emb_path='proto_emb.npy',
    thresholds_path='thresholds.npy',
)

df = pd.read_parquet('your_data.parquet')
label, distance, details, is_neither = predictor.predict(df)
```

### Batch prediction

```python
results = run_batch_predictions(predictor, 'testset/', fraction=0.1)
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
