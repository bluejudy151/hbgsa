import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from config import CHAR_SMI_SET, PT_FEATURE_SIZE


def label_smiles(line, max_smi_len):
    X = np.zeros(max_smi_len, dtype=np.int32)
    for i, ch in enumerate(line[:max_smi_len]):
        X[i] = CHAR_SMI_SET[ch] - 1
    return X


class HBGSADataset(Dataset):
    def __init__(self, data_path, phase, max_seq_len, max_pkt_len, max_smi_len, max_hbonds=20, hbond_csv_path=None):
        data_path = Path(data_path)
        self.phase = phase
        self.max_seq_len = max_seq_len
        self.max_pkt_len = max_pkt_len
        self.max_smi_len = max_smi_len
        self.max_hbonds = max_hbonds

        affinity_df = pd.read_csv(data_path / 'affinity_data.csv')
        self.affinity = {row[0]: row[1] for _, row in affinity_df.iterrows()}

        ligands_df = pd.read_csv(data_path / f"{phase}_smi.csv")
        self.smi = {i["pdbid"]: i["smiles"] for _, i in ligands_df.iterrows()}

        seq_path = data_path / phase / 'global'
        self.seq_paths = sorted(list(seq_path.glob('*')))

        pkt_path = data_path / phase / 'pocket'
        self.pkt_paths = sorted(list(pkt_path.glob('*')))

        self.length = len(self.smi)

        self.hbond_dict = {}
        if hbond_csv_path and Path(hbond_csv_path).exists():
            df = pd.read_csv(hbond_csv_path)
            coord_cols = [col for col in df.columns if col.startswith(('p_', 'l_', 'm_'))]
            for _, row in df.iterrows():
                pdbid = str(row['pdb_id'])
                coords = row[coord_cols].values.astype(np.float32)
                self.hbond_dict[pdbid] = np.clip(coords, -100.0, 100.0)
    
    def __getitem__(self, idx):
        seq_file = self.seq_paths[idx]
        pkt_file = self.pkt_paths[idx]
        
        _seq_tensor = pd.read_csv(seq_file, index_col=0).drop(['idx'], axis=1, errors='ignore').values[:self.max_seq_len]
        seq_tensor = np.zeros((self.max_seq_len, PT_FEATURE_SIZE), dtype=np.float32)
        seq_tensor[:len(_seq_tensor)] = _seq_tensor

        _pkt_tensor = pd.read_csv(pkt_file, index_col=0).drop(['idx'], axis=1, errors='ignore').values[:self.max_pkt_len]
        pkt_tensor = np.zeros((self.max_pkt_len, PT_FEATURE_SIZE), dtype=np.float32)
        pkt_tensor[:len(_pkt_tensor)] = _pkt_tensor

        pdbid = seq_file.name.split('.')[0]
        smi_tensor = label_smiles(self.smi[pdbid], self.max_smi_len)

        if pdbid in self.hbond_dict:
            hbond_coords = self.hbond_dict[pdbid]
        else:
            hbond_coords = np.zeros(self.max_hbonds * 9, dtype=np.float32)

        affinity = np.array(self.affinity[pdbid], dtype=np.float32)

        return seq_tensor, pkt_tensor, smi_tensor, hbond_coords, affinity
    
    def __len__(self):
        return self.length
