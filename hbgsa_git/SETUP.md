# Setup Instructions

## Data Preparation

Due to file size limitations, the data files are not included in this repository. Please prepare your data in the following structure:

```
github/
├── data/
│   ├── affinity_data.csv
│   ├── training_smi.csv
│   ├── validation_smi.csv
│   ├── test_smi.csv
│   ├── training/
│   │   ├── global/
│   │   └── pocket/
│   ├── validation/
│   │   ├── global/
│   │   └── pocket/
│   ├── test/
│   │   ├── global/
│   │   └── pocket/
│   └── Bond/
│       └── hbond_3d_flattened.csv
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Verify PyTorch installation with CUDA (if using GPU):
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Configuration

Edit `config.py` to adjust:
- Training hyperparameters
- Model architecture parameters
- Data paths (if different from default)

## Running

Train the model:
```bash
python main.py
```

Results will be saved in `./runs/seed{seed}-HBond-GNN-Enhanced-SEQ-PKT-SMI/`

## Notes

- The model requires protein sequence features, pocket features, SMILES features, and hydrogen bond data
- Training on GPU is recommended for faster performance
- Adjust `BATCH_SIZE` in `config.py` based on your GPU memory
