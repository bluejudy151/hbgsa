import numpy as np
import torch
from torch.utils.data import DataLoader

from config import *
from dataset import HBGSADataset
from model import HBGSA
from trainer import train_model, save_top_models, load_best_model, save_results


def main():
    seed = RUN_SEED
    run_name = f'seed{seed}-HBond-GNN-Enhanced-SEQ-PKT-SMI'
    save_path = RUNS_PATH / run_name
    save_path.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    data_loaders = {
        'training': DataLoader(
            HBGSADataset(DATA_PATH, 'training', MAX_SEQ_LEN, MAX_PKT_LEN, MAX_SMI_LEN, MAX_HBONDS, HBOND_CSV_PATH),
            batch_size=BATCH_SIZE, pin_memory=True, num_workers=NUM_WORKERS, shuffle=True
        ),
        'validation': DataLoader(
            HBGSADataset(DATA_PATH, 'validation', MAX_SEQ_LEN, MAX_PKT_LEN, MAX_SMI_LEN, MAX_HBONDS, HBOND_CSV_PATH),
            batch_size=BATCH_SIZE, pin_memory=True, num_workers=NUM_WORKERS, shuffle=False
        ),
        'test': DataLoader(
            HBGSADataset(DATA_PATH, 'test', MAX_SEQ_LEN, MAX_PKT_LEN, MAX_SMI_LEN, MAX_HBONDS, HBOND_CSV_PATH),
            batch_size=BATCH_SIZE, pin_memory=True, num_workers=NUM_WORKERS, shuffle=False
        )
    }
    
    model = HBGSA(
        MAX_HBONDS, SMI_EMBED_SIZE, SEQ_EMBED_SIZE, SEQ_OUT_CHANNELS, 
        PKT_OUT_CHANNELS, SMI_OUT_CHANNELS, HBOND_OUT_DIM, 
        DROPOUT_RATE, CLASSIFIER_DROPOUT
    ).to(device)
    
    saved_models, loss_function = train_model(model, data_loaders, device, save_path)
    
    best_model_filename = save_top_models(saved_models, save_path)
    
    model = load_best_model(model, save_path, best_model_filename)
    
    save_results(model, data_loaders, loss_function, device, save_path)


if __name__ == '__main__':
    main()
