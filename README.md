# DL for Win

Deep learning framework for plasma physics structure classification. A BiAutoencoder (CNN + Bi-LSTM) identifies coherent structures — shocks, vortex chains, solitons, holes, and other patterns — from 4-channel magnetic field time series.

## Project structure

```
├── src/
│   ├── train_20240101-0130.ipynb          # Model training pipeline
│   ├── prediction_for_arbitrary_df.ipynb  # Inference on arbitrary data
│   └── clean_sample.ipynb                 # Sample data preprocessing
├── samples/                             # Labeled prototype waveforms
├── samples_clean/                       # Cleaned prototype waveforms
├── bi_model.pth                         # Trained model weights
├── trainset_*/                          # Training datasets (git-ignored)
├── proto_emb.npy / target_pts.npy       # Cached embeddings
└── thresholds.npy / max_len.npy         # Classification thresholds
```

## Model architecture

- **1D-CNN frontend** — extracts local gradient and spike features
- **Bi-LSTM encoder** — captures bidirectional temporal dependencies
- **Mixed pooling** (max + mean) — preserves both peak signatures and background trends
- **Bi-LSTM decoder** — reconstructs the 4-channel input from a 64-dim latent space

## Classification targets

| Class | Description |
|---|---|
| `sheet` | Current sheet / Alfvénic structure |
| `vortex chain` | Chain of magnetic vortices |
| `c vortex` | Compressible vortex |
| `l vortex` | Line vortex / linear structure |
| `hole` | Magnetic hole / dip region |
| `soliton` | Solitary wave |
| `shock` | Shock front / compressible jump |

## Usage

Open `src/prediction_for_arbitrary_df.ipynb`, run all cells (this defines the model and `PhysicalPredictor` class), then classify as needed:

```python
predictor = PhysicalPredictor(
    model_path='bi_model.pth',
    proto_emb_path='proto_emb.npy',
    thresholds_path='thresholds.npy',
    target_pts=300,
)

df = pd.read_parquet('your_data.parquet')
label, distance, details, is_neither = predictor.predict(df)

print(f"Prediction: {label} (distance: {distance:.4f})")
```

## Requirements

- Python 3.8+
- PyTorch
- pandas, numpy
- pyarrow (for parquet I/O)
- tqdm, matplotlib
