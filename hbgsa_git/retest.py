import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr
import sklearn.metrics as m
from sklearn.linear_model import LinearRegression
from pathlib import Path
from collections import OrderedDict
import os

SCRIPT_DIR = Path(__file__).parent.resolve()
INTERLEAVED_WEIGHTS_PATH = SCRIPT_DIR / 'weights.pt'

DATA_PATH = SCRIPT_DIR / 'data' / 'data'
CSAR_DATA_PATH = SCRIPT_DIR / 'data' / 'CSAR_output'

BATCH_SIZE = 32
MAX_SEQ_LEN = 1000
MAX_PKT_LEN = 63
MAX_SMI_LEN = 150
MAX_HBONDS = 20

USE_SEQ = True
USE_POCKET = True
USE_SMILES = True
USE_HBOND_test = False  # 氢键CNN特征（测试用，默认关闭）
USE_HBOND_GNN = True
USE_ENHANCED_FEATURES = True

SMI_EMBED_SIZE = 128
SEQ_EMBED_SIZE = 128
SEQ_OUT_CHANNELS = 128
PKT_OUT_CHANNELS = 128
SMI_OUT_CHANNELS = 128
HBOND_OUT_DIM = 128
DROPOUT_RATE = 0.2
CLASSIFIER_DROPOUT = 0.5

PT_FEATURE_SIZE = 40
CHAR_SMI_SET = {
    "(": 1, ".": 2, "0": 3, "2": 4, "4": 5, "6": 6, "8": 7, "@": 8,
    "B": 9, "D": 10, "F": 11, "H": 12, "L": 13, "N": 14, "P": 15, "R": 16,
    "T": 17, "V": 18, "Z": 19, "\\": 20, "b": 21, "d": 22, "f": 23, "h": 24,
    "l": 25, "n": 26, "r": 27, "t": 28, "#": 29, "%": 30, ")": 31, "+": 32,
    "-": 33, "/": 34, "1": 35, "3": 36, "5": 37, "7": 38, "9": 39, "=": 40,
    "A": 41, "C": 42, "E": 43, "G": 44, "I": 45, "K": 46, "M": 47, "O": 48,
    "S": 49, "U": 50, "W": 51, "Y": 52, "[": 53, "]": 54, "a": 55, "c": 56,
    "e": 57, "g": 58, "i": 59, "m": 60, "o": 61, "s": 62, "u": 63, "y": 64
}
CHAR_SMI_SET_LEN = len(CHAR_SMI_SET)


def extract_model_from_interleaved(interleaved_path, dataset_index):
    data = torch.load(interleaved_path, map_location='cpu')
    interleaved_weights = data['interleaved_weights']
    original_keys = data['original_keys']
    
    model_weights = OrderedDict()
    for n, original_key in enumerate(original_keys):
        interleaved_index = 2 * n + dataset_index
        interleaved_key = f"param.{interleaved_index}"
        if interleaved_key in interleaved_weights:
            model_weights[original_key] = interleaved_weights[interleaved_key]
    
    return model_weights


def c_index(y_true, y_pred):
    summ = 0
    pair = 0
    for i in range(1, len(y_true)):
        for j in range(0, i):
            pair += 1
            if y_true[i] > y_true[j]:
                summ += 1 * (y_pred[i] > y_pred[j]) + 0.5 * (y_pred[i] == y_pred[j])
            elif y_true[i] < y_true[j]:
                summ += 1 * (y_pred[i] < y_pred[j]) + 0.5 * (y_pred[i] == y_pred[j])
            else:
                pair -= 1
    return summ / pair if pair != 0 else 0

def RMSE(y_true, y_pred):
    return np.sqrt(m.mean_squared_error(y_true, y_pred))

def MAE(y_true, y_pred):
    return m.mean_absolute_error(y_true, y_pred)

def CORR(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]

def SD(y_true, y_pred):
    y_pred = y_pred.reshape((-1, 1))
    lr = LinearRegression().fit(y_pred, y_true)
    y_ = lr.predict(y_pred)
    return np.sqrt(np.square(y_true - y_).sum() / (len(y_pred) - 1))

def label_smiles(line, max_smi_len):
    X = np.zeros(max_smi_len, dtype=np.int32)
    for i, ch in enumerate(line[:max_smi_len]):
        X[i] = CHAR_SMI_SET[ch] - 1
    return X



class CachedHBGSADataset(Dataset):
    def __init__(self, data_path, phase, max_seq_len, max_pkt_len, max_smi_len):
        data_path = Path(data_path)
        self.phase = phase
        affinity_df = pd.read_csv(data_path / 'affinity_data.csv')
        self.affinity = {row.iloc[0]: row.iloc[1] for _, row in affinity_df.iterrows()}
        ligands_df = pd.read_csv(data_path / f"{phase}_smi.csv")
        self.smi = {i["pdbid"]: i["smiles"] for _, i in ligands_df.iterrows()}
        self.max_smi_len = max_smi_len
        seq_path = data_path / phase / 'global'
        seq_paths = sorted(list(seq_path.glob('*')))
        self.max_seq_len = max_seq_len
        pkt_path = data_path / phase / 'pocket'
        pkt_paths = sorted(list(pkt_path.glob('*')))
        self.max_pkt_len = max_pkt_len
        self.length = len(self.smi)
        self.seq_data = []
        self.pkt_data = []
        self.smi_data = []
        self.affinity_data = []
        for idx in range(self.length):
            seq = seq_paths[idx]
            pkt = pkt_paths[idx]
            _seq_tensor = pd.read_csv(seq, index_col=0).drop(['idx'], axis=1, errors='ignore').values[:self.max_seq_len]
            seq_tensor = np.zeros((self.max_seq_len, PT_FEATURE_SIZE))
            seq_tensor[:len(_seq_tensor)] = _seq_tensor
            self.seq_data.append(seq_tensor.astype(np.float32))
            _pkt_tensor = pd.read_csv(pkt, index_col=0).drop(['idx'], axis=1, errors='ignore').values[:self.max_pkt_len]
            pkt_tensor = np.zeros((self.max_pkt_len, PT_FEATURE_SIZE))
            pkt_tensor[:len(_pkt_tensor)] = _pkt_tensor
            self.pkt_data.append(pkt_tensor.astype(np.float32))
            pdbid = seq.name.split('.')[0]
            self.smi_data.append(label_smiles(self.smi[pdbid], self.max_smi_len))
            self.affinity_data.append(np.array(self.affinity[pdbid], dtype=np.float32))
    
    def __getitem__(self, idx):
        hbond_dummy = np.zeros(MAX_HBONDS * 9, dtype=np.float32)
        return (self.seq_data[idx], self.pkt_data[idx], self.smi_data[idx], hbond_dummy, self.affinity_data[idx])
    
    def __len__(self):
        return self.length



class CSARDataset(Dataset):
    def __init__(self, csar_path, max_seq_len, max_pkt_len, max_smi_len):
        csar_path = Path(csar_path)
        self.max_seq_len = max_seq_len
        self.max_pkt_len = max_pkt_len
        self.max_smi_len = max_smi_len
        affinity_file = csar_path / 'csar-affinity.csv'
        affinity_df = pd.read_csv(affinity_file)
        self.affinity = {row.iloc[0]: row.iloc[1] for _, row in affinity_df.iterrows()}
        smi_file = csar_path / 'csar-smi.csv'
        smi_df = pd.read_csv(smi_file)
        self.smi = {}
        self.valid_pdbids = set()
        for _, row in smi_df.iterrows():
            pdbid = row.iloc[0]
            smiles = row.iloc[1]
            if pd.isna(smiles) or (isinstance(smiles, str) and smiles.strip() == ''):
                continue
            self.smi[pdbid] = smiles
            self.valid_pdbids.add(pdbid)
        seq_path = csar_path / 'CSAR' / 'global'
        pkt_path = csar_path / 'CSAR' / 'pocket'
        seq_paths = sorted(list(seq_path.glob('*')))
        pkt_paths = sorted(list(pkt_path.glob('*')))
        if len(seq_paths) != len(pkt_paths):
            seq_dict = {f.name.split('.')[0]: f for f in seq_paths}
            matched_seq_paths = []
            for pkt_file in pkt_paths:
                pdbid = pkt_file.name.split('.')[0]
                if pdbid in seq_dict:
                    matched_seq_paths.append(seq_dict[pdbid])
                else:
                    matched_seq_paths.append(None)
            seq_paths = matched_seq_paths
        self.length = len(pkt_paths)
        self.pdbids = []
        self.seq_data = []
        self.pkt_data = []
        self.smi_data = []
        self.affinity_data = []
        self.valid_indices = []
        for idx in range(self.length):
            seq_file = seq_paths[idx]
            pkt_file = pkt_paths[idx]
            if seq_file is None:
                continue
            pdbid = seq_file.name.split('.')[0]
            if pdbid not in self.valid_pdbids:
                continue
            try:
                _seq_tensor = pd.read_csv(seq_file, index_col=0).drop(['idx'], axis=1, errors='ignore').values[:self.max_seq_len]
                seq_tensor = np.zeros((self.max_seq_len, PT_FEATURE_SIZE))
                seq_tensor[:len(_seq_tensor)] = _seq_tensor
                self.seq_data.append(seq_tensor.astype(np.float32))
            except:
                continue
            try:
                _pkt_tensor = pd.read_csv(pkt_file, index_col=0).drop(['idx'], axis=1, errors='ignore').values[:self.max_pkt_len]
                pkt_tensor = np.zeros((self.max_pkt_len, PT_FEATURE_SIZE))
                pkt_tensor[:len(_pkt_tensor)] = _pkt_tensor
                self.pkt_data.append(pkt_tensor.astype(np.float32))
            except:
                continue
            self.smi_data.append(label_smiles(self.smi[pdbid], self.max_smi_len))
            if pdbid in self.affinity:
                self.affinity_data.append(np.array(self.affinity[pdbid], dtype=np.float32))
            else:
                continue
            self.pdbids.append(pdbid)
            self.valid_indices.append(idx)

    def __getitem__(self, idx):
        hbond_dummy = np.zeros(MAX_HBONDS * 9, dtype=np.float32)
        return (self.seq_data[idx], self.pkt_data[idx], self.smi_data[idx], hbond_dummy, self.affinity_data[idx])

    def __len__(self):
        return len(self.seq_data)



class Squeeze(nn.Module):
    def forward(self, input: torch.Tensor):
        return input.squeeze()

class CDilated(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False, dilation=d)
    def forward(self, input):
        return self.conv(input)

class DilatedParllelResidualBlockA(nn.Module):
    def __init__(self, nIn, nOut, add=True):
        super().__init__()
        n = int(nOut / 5)
        n1 = nOut - 4 * n
        self.c1 = nn.Conv1d(nIn, n, 1, padding=0)
        self.br1 = nn.Sequential(nn.BatchNorm1d(n), nn.PReLU())
        self.d1 = CDilated(n, n1, 3, 1, 1)
        self.d2 = CDilated(n, n, 3, 1, 2)
        self.d4 = CDilated(n, n, 3, 1, 4)
        self.d8 = CDilated(n, n, 3, 1, 8)
        self.d16 = CDilated(n, n, 3, 1, 16)
        self.br2 = nn.Sequential(nn.BatchNorm1d(nOut), nn.PReLU())
        if nIn != nOut:
            add = False
        self.add = add
    def forward(self, input):
        output1 = self.c1(input)
        output1 = self.br1(output1)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16
        combine = torch.cat([d1, add1, add2, add3, add4], 1)
        if self.add:
            combine = input + combine
        output = self.br2(combine)
        return output

class DilatedParllelResidualBlockB(nn.Module):
    def __init__(self, nIn, nOut, add=True):
        super().__init__()
        n = int(nOut / 4)
        n1 = nOut - 3 * n
        self.c1 = nn.Conv1d(nIn, n, 1, padding=0)
        self.br1 = nn.Sequential(nn.BatchNorm1d(n), nn.PReLU())
        self.d1 = CDilated(n, n1, 3, 1, 1)
        self.d2 = CDilated(n, n, 3, 1, 2)
        self.d4 = CDilated(n, n, 3, 1, 4)
        self.d8 = CDilated(n, n, 3, 1, 8)
        self.br2 = nn.Sequential(nn.BatchNorm1d(nOut), nn.PReLU())
        if nIn != nOut:
            add = False
        self.add = add
    def forward(self, input):
        output1 = self.c1(input)
        output1 = self.br1(output1)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        combine = torch.cat([d1, add1, add2, add3], 1)
        if self.add:
            combine = input + combine
        output = self.br2(combine)
        return output



class SelfAttention1D(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.query = nn.Conv1d(in_channels, in_channels // reduction, 1)
        self.key = nn.Conv1d(in_channels, in_channels // reduction, 1)
        self.value = nn.Conv1d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        batch_size, channels, length = x.size()
        query = self.query(x).view(batch_size, -1, length).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, length)
        value = self.value(x).view(batch_size, -1, length)
        attention = torch.bmm(query, key)
        attention = torch.softmax(attention, dim=-1)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = self.gamma * out + x
        return out

class HBond3DEncoder(nn.Module):
    def __init__(self, input_channels=9, seq_embed_size=128, output_dim=128):
        super().__init__()
        self.embed = nn.Linear(input_channels, seq_embed_size)
        conv_layers = []
        ic = seq_embed_size
        for oc in [32, 64, output_dim]:
            conv_layers.append(nn.Conv1d(ic, oc, 3))
            conv_layers.append(nn.BatchNorm1d(oc))
            conv_layers.append(nn.PReLU())
            ic = oc
        conv_layers.append(nn.AdaptiveMaxPool1d(1))
        conv_layers.append(Squeeze())
        self.conv = nn.Sequential(*conv_layers)
    def forward(self, hbond_coords):
        x = hbond_coords.view(-1, 20, 9)
        x = self.embed(x)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = torch.clamp(x, min=-10.0, max=10.0)
        return x

class HBondGNNEncoder(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=128, output_dim=128, k_neighbors=5):
        super().__init__()
        self.k = k_neighbors
        self.hidden_dim = hidden_dim
        self.node_embed = nn.Linear(input_dim, hidden_dim)
        self.gc1 = nn.Linear(hidden_dim, hidden_dim)
        self.gc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.2)
    def get_knn_adj(self, coords):
        B, N, C = coords.shape
        dist = torch.cdist(coords, coords)
        _, indices = dist.topk(self.k, dim=-1, largest=False)
        adj = torch.zeros(B, N, N, device=coords.device)
        batch_indices = torch.arange(B, device=coords.device).view(B, 1, 1).expand(B, N, self.k)
        node_indices = torch.arange(N, device=coords.device).view(1, N, 1).expand(B, N, self.k)
        adj[batch_indices, node_indices, indices] = 1.0
        return adj
    def forward(self, hbond_coords):
        x = hbond_coords.view(-1, 20, 9)
        pos = x[:, :, 6:9]
        adj = self.get_knn_adj(pos)
        h = self.node_embed(x)
        h_agg = torch.bmm(adj, h)
        h = self.gc1(h_agg)
        h = self.ln1(h)
        h = self.act(h)
        h = self.dropout(h)
        h_agg = torch.bmm(adj, h)
        h_new = self.gc2(h_agg)
        h_new = self.ln2(h_new)
        h = self.act(h + h_new)
        out = torch.max(h, dim=1)[0]
        return out



class HBGSA(nn.Module):
    def __init__(self, use_seq=True, use_pocket=True, use_smiles=True, use_hbond_test=False,
                 max_hbonds=20, smi_embed_size=128, seq_embed_size=128,
                 seq_oc=128, pkt_oc=128, smi_oc=128, hbond_oc=128,
                 dropout_rate=0.2, classifier_dropout=0.5,
                 use_enhanced_features=False, use_hbond_gnn=True):
        super().__init__()
        self.use_seq = use_seq
        self.use_pocket = use_pocket
        self.use_smiles = use_smiles
        self.use_hbond_test = use_hbond_test
        self.use_hbond_gnn = use_hbond_gnn
        self.use_enhanced_features = use_enhanced_features
        if not any([use_seq, use_pocket, use_smiles, use_hbond_test]):
            raise ValueError("At least one module must be enabled")
        if use_smiles:
            self.smi_embed = nn.Embedding(CHAR_SMI_SET_LEN, smi_embed_size)
        if use_seq or use_pocket:
            self.seq_embed = nn.Linear(PT_FEATURE_SIZE, seq_embed_size)
        if use_seq:
            conv_seq = []
            ic = seq_embed_size
            for oc in [32, 64, 64, seq_oc]:
                conv_seq.append(DilatedParllelResidualBlockA(ic, oc))
                ic = oc
            if use_enhanced_features:
                conv_seq.append(SelfAttention1D(ic, reduction=8))
            conv_seq.append(nn.AdaptiveMaxPool1d(1))
            conv_seq.append(Squeeze())
            self.conv_seq = nn.Sequential(*conv_seq)
        if use_pocket:
            conv_pkt = []
            ic = seq_embed_size
            for oc in [32, 64, pkt_oc]:
                conv_pkt.append(nn.Conv1d(ic, oc, 3))
                conv_pkt.append(nn.BatchNorm1d(oc))
                conv_pkt.append(nn.PReLU())
                ic = oc
            conv_pkt.append(nn.AdaptiveMaxPool1d(1))
            conv_pkt.append(Squeeze())
            self.conv_pkt = nn.Sequential(*conv_pkt)
        if use_smiles:
            conv_smi = []
            ic = smi_embed_size
            for oc in [32, 64, smi_oc]:
                conv_smi.append(DilatedParllelResidualBlockB(ic, oc))
                ic = oc
            if use_enhanced_features:
                conv_smi.append(SelfAttention1D(ic, reduction=8))
            conv_smi.append(nn.AdaptiveMaxPool1d(1))
            conv_smi.append(Squeeze())
            self.conv_smi = nn.Sequential(*conv_smi)
        if use_hbond_test:
            self.hbond_cnn = HBond3DEncoder(input_channels=9, seq_embed_size=seq_embed_size, output_dim=hbond_oc)
        if use_hbond_test and use_hbond_gnn:
            self.hbond_gnn = HBondGNNEncoder(input_dim=9, hidden_dim=hbond_oc, output_dim=hbond_oc)
        total_features = 0
        if use_seq:
            total_features += seq_oc
        if use_pocket:
            total_features += pkt_oc
        if use_smiles:
            total_features += smi_oc
        if use_hbond_test:
            total_features += hbond_oc
            if use_hbond_gnn:
                total_features += hbond_oc
        self.cat_dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(total_features, 128),
            nn.Dropout(classifier_dropout),
            nn.PReLU(),
            nn.Linear(128, 64),
            nn.Dropout(classifier_dropout),
            nn.PReLU(),
            nn.Linear(64, 1),
            nn.PReLU()
        )
    def forward(self, seq, pkt, smi, hbond_coords=None):
        features = []
        if self.use_seq:
            seq_embed = self.seq_embed(seq)
            seq_embed = torch.transpose(seq_embed, 1, 2)
            seq_conv = self.conv_seq(seq_embed)
            features.append(seq_conv)
        if self.use_pocket:
            pkt_embed = self.seq_embed(pkt)
            pkt_embed = torch.transpose(pkt_embed, 1, 2)
            pkt_conv = self.conv_pkt(pkt_embed)
            features.append(pkt_conv)
        if self.use_smiles:
            smi_embed = self.smi_embed(smi)
            smi_embed = torch.transpose(smi_embed, 1, 2)
            smi_conv = self.conv_smi(smi_embed)
            features.append(smi_conv)
        if self.use_hbond_test and hbond_coords is not None:
            cnn_feat = self.hbond_cnn(hbond_coords)
            features.append(cnn_feat)
            if self.use_hbond_gnn:
                gnn_feat = self.hbond_gnn(hbond_coords)
                gnn_feat = torch.clamp(gnn_feat, min=-10.0, max=10.0)
                features.append(gnn_feat)
        if len(features) == 0:
            raise ValueError("No features enabled")
        processed_features = []
        for feat in features:
            if feat.dim() == 1:
                feat = feat.unsqueeze(0)
            elif feat.dim() > 2:
                feat = feat.view(feat.size(0), -1)
            processed_features.append(feat)
        cat = torch.cat(processed_features, dim=1)
        cat = self.cat_dropout(cat)
        output = self.classifier(cat)
        return output



def test(model, test_loader, device, return_predictions=False):
    model.eval()
    outputs = []
    targets = []
    with torch.no_grad():
        for data in test_loader:
            seq, pkt, smi, hbond_coords, y = data
            seq = seq.to(device)
            pkt = pkt.to(device)
            smi = smi.long().to(device)
            hbond_coords = hbond_coords.to(device)
            y = y.to(device)
            y_hat = model(seq, pkt, smi, hbond_coords)
            outputs.append(y_hat.cpu().numpy().reshape(-1))
            targets.append(y.cpu().numpy().reshape(-1))
    targets = np.concatenate(targets).reshape(-1)
    outputs = np.concatenate(outputs).reshape(-1)
    metrics = {
        'c_index': c_index(targets, outputs),
        'RMSE': RMSE(targets, outputs),
        'MAE': MAE(targets, outputs),
        'SD': SD(targets, outputs),
        'CORR': CORR(targets, outputs),
    }
    if return_predictions:
        return metrics, targets, outputs
    else:
        return metrics


def main():
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    print("Do you want to evaluate training and validation sets? (y/n)")
    print("Note: Training set has 11906 samples, loading will take some time")
    user_input = input("Enter: ").strip().lower()
    evaluate_train_val = (user_input != 'n')
    
    if evaluate_train_val:
        train_dataset = CachedHBGSADataset(DATA_PATH, 'training', MAX_SEQ_LEN, MAX_PKT_LEN, MAX_SMI_LEN)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        val_dataset = CachedHBGSADataset(DATA_PATH, 'validation', MAX_SEQ_LEN, MAX_PKT_LEN, MAX_SMI_LEN)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    test_dataset = CachedHBGSADataset(DATA_PATH, 'test', MAX_SEQ_LEN, MAX_PKT_LEN, MAX_SMI_LEN)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    paper_model = HBGSA(
        use_seq=USE_SEQ,
        use_pocket=USE_POCKET,
        use_smiles=USE_SMILES,
        use_hbond_test=USE_HBOND_test,
        max_hbonds=MAX_HBONDS,
        smi_embed_size=SMI_EMBED_SIZE,
        seq_embed_size=SEQ_EMBED_SIZE,
        seq_oc=SEQ_OUT_CHANNELS,
        pkt_oc=PKT_OUT_CHANNELS,
        smi_oc=SMI_OUT_CHANNELS,
        hbond_oc=HBOND_OUT_DIM,
        dropout_rate=DROPOUT_RATE,
        classifier_dropout=CLASSIFIER_DROPOUT,
        use_enhanced_features=USE_ENHANCED_FEATURES,
        use_hbond_gnn=USE_HBOND_GNN
    )

    paper_weights = extract_model_from_interleaved(INTERLEAVED_WEIGHTS_PATH, dataset_index=0)
    paper_model.load_state_dict(paper_weights)
    paper_model = paper_model.to(device)
    
    if evaluate_train_val:
        print("\nTraining set results:")
        train_metrics = test(paper_model, train_loader, device)
        for k, v in train_metrics.items():
            print(f"{k:15s}: {v:.6f}")
        
        print("\nValidation set results:")
        val_metrics = test(paper_model, val_loader, device)
        for k, v in val_metrics.items():
            print(f"{k:15s}: {v:.6f}")
    
    print("\nTest set results:")
    test_metrics = test(paper_model, test_loader, device)
    for k, v in test_metrics.items():
        print(f"{k:15s}: {v:.6f}")
    

    csar_dataset = CSARDataset(CSAR_DATA_PATH, MAX_SEQ_LEN, MAX_PKT_LEN, MAX_SMI_LEN)
    csar_loader = DataLoader(csar_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    csar_model = HBGSA(
        use_seq=USE_SEQ,
        use_pocket=USE_POCKET,
        use_smiles=USE_SMILES,
        use_hbond_test=USE_HBOND_test,
        max_hbonds=MAX_HBONDS,
        smi_embed_size=SMI_EMBED_SIZE,
        seq_embed_size=SEQ_EMBED_SIZE,
        seq_oc=SEQ_OUT_CHANNELS,
        pkt_oc=PKT_OUT_CHANNELS,
        smi_oc=SMI_OUT_CHANNELS,
        hbond_oc=HBOND_OUT_DIM,
        dropout_rate=DROPOUT_RATE,
        classifier_dropout=CLASSIFIER_DROPOUT,
        use_enhanced_features=USE_ENHANCED_FEATURES,
        use_hbond_gnn=USE_HBOND_GNN
    )
    
    csar_weights = extract_model_from_interleaved(INTERLEAVED_WEIGHTS_PATH, dataset_index=1)
    csar_model.load_state_dict(csar_weights)
    csar_model = csar_model.to(device)

    
    print("\nCSAR test set results:")
    csar_metrics = test(csar_model, csar_loader, device)
    for k, v in csar_metrics.items():
        print(f"{k:15s}: {v:.6f}")
    

if __name__ == '__main__':
    main()
