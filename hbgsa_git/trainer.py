import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from pathlib import Path
import shutil

from config import *
from dataset import HBGSADataset
from model import HBGSA, PearsonCorrelationLoss
from metrics import c_index, RMSE, MAE, SD, CORR


def evaluate(model, data_loader, loss_function, device):
    model.eval()
    test_loss = 0
    outputs = []
    targets = []
    
    with torch.no_grad():
        for data in data_loader:
            seq, pkt, smi, hbond_coords, y = data
            seq = seq.to(device)
            pkt = pkt.to(device)
            smi = smi.long().to(device)
            hbond_coords = hbond_coords.to(device)
            y = y.to(device)
            
            y_hat = model(seq, pkt, smi, hbond_coords)
            test_loss += loss_function(y_hat.view(-1), y.view(-1)).item()
            outputs.append(y_hat.cpu().numpy().reshape(-1))
            targets.append(y.cpu().numpy().reshape(-1))
    
    targets = np.concatenate(targets).reshape(-1)
    outputs = np.concatenate(outputs).reshape(-1)
    test_loss /= len(data_loader.dataset)
    
    return {
        'loss': test_loss,
        'c_index': c_index(targets, outputs),
        'RMSE': RMSE(targets, outputs),
        'MAE': MAE(targets, outputs),
        'SD': SD(targets, outputs),
        'CORR': CORR(targets, outputs),
    }, targets, outputs


def train_model(model, data_loaders, device, save_path):
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LEARNING_RATE, 
        epochs=N_EPOCH, steps_per_epoch=len(data_loaders['training'])
    )
    
    loss_reg_fn = nn.SmoothL1Loss(reduction='sum')
    loss_pearson_fn = PearsonCorrelationLoss()
    
    save_start_epoch = int(N_EPOCH * (1 - SAVE_BEST_RATIO)) + 1
    best_val_loss = float('inf')
    saved_models = []
    
    for epoch in range(1, N_EPOCH + 1):
        model.train()
        for data in data_loaders['training']:
            seq, pkt, smi, hbond_coords, y = data
            seq = seq.to(device)
            pkt = pkt.to(device)
            smi = smi.long().to(device)
            hbond_coords = hbond_coords.to(device)
            y = y.to(device)
            
            output = model(seq, pkt, smi, hbond_coords)
            loss_reg = loss_reg_fn(output.view(-1), y.view(-1))
            loss_pearson = loss_pearson_fn(output, y)
            loss = loss_reg + (loss_pearson * 100.0)
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
        
        val_performance, _, _ = evaluate(model, data_loaders['validation'], loss_reg_fn, device)
        
        if epoch >= save_start_epoch and val_performance['loss'] < best_val_loss:
            best_val_loss = val_performance['loss']
            model_filename = f'model_epoch_{epoch}.pt'
            torch.save({'model_state_dict': model.state_dict()}, save_path / model_filename)
            saved_models.append({
                'epoch': epoch, 
                'val_loss': val_performance['loss'], 
                'val_rmse': val_performance['RMSE'], 
                'filename': model_filename
            })
    
    return saved_models, loss_reg_fn


def save_top_models(saved_models, save_path):
    if not saved_models:
        return None
    
    saved_models_sorted = sorted(saved_models, key=lambda x: x['val_loss'])
    top_k_models = saved_models_sorted[:min(TOP_K_MODELS, len(saved_models_sorted))]
    
    top_models_dir = save_path / 'top_models'
    top_models_dir.mkdir(exist_ok=True)
    
    for rank, model_info in enumerate(top_k_models, 1):
        src_file = save_path / model_info['filename']
        dst_file = top_models_dir / f'rank{rank}_epoch{model_info["epoch"]}.pt'
        shutil.copy2(src_file, dst_file)
    
    return top_k_models[0]['filename']


def load_best_model(model, save_path, best_model_filename):
    if best_model_filename:
        best_model_path = save_path / best_model_filename
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    return model


def save_results(model, data_loaders, loss_function, device, save_path):
    with open(save_path / 'results.txt', 'w') as f:
        for phase in ['training', 'validation', 'test']:
            performance, _, _ = evaluate(model, data_loaders[phase], loss_function, device)
            f.write(f'{phase}:\n')
            for k, v in performance.items():
                f.write(f'{k}: {v}\n')
            f.write('\n')
