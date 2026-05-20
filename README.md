# DL for Win

Deep learning framework for plasma physics structure classification. A BiAutoencoder (CNN + Bi-LSTM) identifies coherent structures — shocks, vortex chains, solitons, holes, and other patterns — from 4-channel magnetic field time series.

## Project structure

```
├── src/
│   ├── train_20240101-0130.ipynb       # Model training pipeline
│   ├── prediction_for_arbitrary_df.ipynb  # Inference on arbitrary data
│   ├── clean_sample.ipynb               # Sample data preprocessing
│   └── temp.py                          # Core module (model, loss, training loop)
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

```python
# Load model and classify new data
model = BiAutoencoder(input_size=4, cnn_channels=16, hidden_size=128, num_layers=2, latent_dim=64)
model.load_state_dict(torch.load('bi_model.pth'))
model.to(device)

predictions, proto_embeddings = test_clustering(model, test_data, prototypes, device)
```

## Requirements

- Python 3.8+
- PyTorch
- pandas, numpy
- pyarrow (for parquet I/O)
- tqdm, matplotlib
