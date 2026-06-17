# HBGSA - Hydrogen Bond Graph with Self-Attention

A deep learning model for drug-target binding affinity prediction using multi-modal features including protein sequences, binding pockets, molecular SMILES, and hydrogen bond interactions.

## Features

- **Multi-modal Architecture**: Integrates sequence, pocket, SMILES, and hydrogen bond features
- **Graph Neural Network**: Models hydrogen bond interactions using GNN
- **Self-Attention Mechanism**: Enhanced feature extraction with 1D self-attention
- **Dilated Residual Blocks**: Captures multi-scale patterns in sequences and molecules

## Project Structure

```
github/
├── config.py          # Configuration parameters
├── dataset.py         # Dataset class and data loading
├── model.py           # Model architecture
├── metrics.py         # Evaluation metrics
├── trainer.py         # Training and evaluation functions
├── main.py            # Main entry point for training
├── demo/
│   ├── retest.py      # Reproduction script with pre-trained weights
│   └── interleaved_weights.pt  # Pre-trained model weights (not in repo)
├── data/              # Data directory (not in repo, see SETUP.md)
├── .gitignore         # Git ignore rules
├── requirements.txt   # Python dependencies
├── LICENSE            # MIT License
├── SETUP.md           # Setup instructions
└── README.md          # This file
```

## File Descriptions

### config.py
Contains all configuration parameters:
- Training hyperparameters (learning rate, batch size, epochs, etc.)
- Model architecture parameters (embedding sizes, channels, etc.)
- Data parameters (max lengths, paths, etc.)
- SMILES character set

### dataset.py
- `label_smiles()`: Convert SMILES string to numerical sequence
- `HBGSADataset`: PyTorch Dataset class for loading data

### model.py
Model components:
- `PearsonCorrelationLoss`: Pearson correlation loss function
- `CDilated`: Dilated convolution layer
- `DilatedParllelResidualBlockA/B`: Parallel dilated residual blocks
- `HBondGNNEncoder`: Hydrogen bond graph neural network encoder
- `SelfAttention1D`: 1D self-attention layer
- `HBGSA`: Main model class

### metrics.py
Evaluation metrics:
- `c_index()`: Concordance Index
- `RMSE()`: Root Mean Squared Error
- `MAE()`: Mean Absolute Error
- `CORR()`: Pearson Correlation Coefficient
- `SD()`: Standard Deviation

### trainer.py
Training utilities:
- `evaluate()`: Evaluate model on a dataset
- `train_model()`: Main training loop
- `save_top_models()`: Save top-K best models
- `load_best_model()`: Load the best model
- `save_results()`: Save evaluation results

### main.py
Main entry point that orchestrates:
1. Setup (seed, device, paths)
2. Data loading
3. Model initialization
4. Training
5. Model selection
6. Final evaluation

## Usage

### Training from Scratch

Train a new model:
```bash
python main.py
```

Results will be saved in `./runs/seed{seed}-HBond-GNN-Enhanced-SEQ-PKT-SMI/`

### Reproducing Paper Results

Use the pre-trained model weights to reproduce results:

```bash
cd demo
python retest.py
```

The reproduction script will prompt you:
- Enter `y` to evaluate all datasets (training, validation, test, CSAR)
- Enter `n` to evaluate only test and CSAR datasets (faster)

**Note**: The pre-trained weights file (`interleaved_weights.pt`) contains two models:
- Model 0: For Training/Validation/Test datasets
- Model 1: For CSAR dataset (fine-tuned model)

Due to file size, weights are not included in the repository. Please obtain them separately or train your own model.

## Requirements

- PyTorch
- NumPy
- Pandas
- scikit-learn
- scipy

## Model Features

- **Sequence Features**: Global protein sequence (SEQ)
- **Pocket Features**: Binding pocket (PKT)
- **SMILES Features**: Molecular structure (SMI)
- **Hydrogen Bond Features**: Graph neural network (GNN)
- **Enhanced Features**: Self-attention layers

## Output

### Training Output
Results are saved in `./runs/seed{seed}-HBond-GNN-Enhanced-SEQ-PKT-SMI/`:
- `model_epoch_{epoch}.pt`: Saved model checkpoints
- `top_models/`: Top-K best models
- `results.txt`: Final evaluation metrics

### Reproduction Script Output
The demo script prints evaluation metrics for each dataset:
- **c_index**: Concordance Index
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **SD**: Standard Deviation
- **CORR**: Pearson Correlation Coefficient

## Installation & Setup

See [SETUP.md](SETUP.md) for detailed installation and data preparation instructions.

## Citation

If you use this code in your research, please cite our paper.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
