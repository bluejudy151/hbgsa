import torch
from torch import nn
from config import CHAR_SMI_SET_LEN, PT_FEATURE_SIZE


class PearsonCorrelationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        preds = preds.view(-1)
        targets = targets.view(-1)
        vx = preds - torch.mean(preds)
        vy = targets - torch.mean(targets)
        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)
        return 1.0 - cost


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


class HBondGNNEncoder(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=128, output_dim=128, k_neighbors=5):
        super().__init__()
        self.k = k_neighbors
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


class HBGSA(nn.Module):
    def __init__(self, max_hbonds=20, smi_embed_size=128, seq_embed_size=128,
                 seq_oc=128, pkt_oc=128, smi_oc=128, hbond_oc=128,
                 dropout_rate=0.2, classifier_dropout=0.5):
        super().__init__()
        
        self.smi_embed = nn.Embedding(CHAR_SMI_SET_LEN, smi_embed_size)
        self.seq_embed = nn.Linear(PT_FEATURE_SIZE, seq_embed_size)
        
        conv_seq = []
        ic = seq_embed_size
        for oc in [32, 64, 64, seq_oc]:
            conv_seq.append(DilatedParllelResidualBlockA(ic, oc))
            ic = oc
        conv_seq.append(SelfAttention1D(ic, reduction=8))
        conv_seq.append(nn.AdaptiveMaxPool1d(1))
        conv_seq.append(Squeeze())
        self.conv_seq = nn.Sequential(*conv_seq)
        
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
        
        conv_smi = []
        ic = smi_embed_size
        for oc in [32, 64, smi_oc]:
            conv_smi.append(DilatedParllelResidualBlockB(ic, oc))
            ic = oc
        conv_smi.append(SelfAttention1D(ic, reduction=8))
        conv_smi.append(nn.AdaptiveMaxPool1d(1))
        conv_smi.append(Squeeze())
        self.conv_smi = nn.Sequential(*conv_smi)
        
        self.hbond_gnn = HBondGNNEncoder(input_dim=9, hidden_dim=hbond_oc, output_dim=hbond_oc)

        total_features = seq_oc + pkt_oc + smi_oc + hbond_oc
        
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
        seq_embed = self.seq_embed(seq)
        seq_embed = torch.transpose(seq_embed, 1, 2)
        seq_conv = self.conv_seq(seq_embed)
        
        pkt_embed = self.seq_embed(pkt)
        pkt_embed = torch.transpose(pkt_embed, 1, 2)
        pkt_conv = self.conv_pkt(pkt_embed)
        
        smi_embed = self.smi_embed(smi)
        smi_embed = torch.transpose(smi_embed, 1, 2)
        smi_conv = self.conv_smi(smi_embed)
        
        gnn_feat = self.hbond_gnn(hbond_coords)
        gnn_feat = torch.clamp(gnn_feat, min=-10.0, max=10.0)
        
        cat = torch.cat([seq_conv, pkt_conv, smi_conv, gnn_feat], dim=1)
        cat = self.cat_dropout(cat)
        output = self.classifier(cat)
        return output
