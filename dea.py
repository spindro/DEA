import torch
import numpy as np
import os.path as osp
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, accuracy_score
from utils import (
    encode_classes,
    get_link_labels,
    prediction_fairness,
)
from os.path import join, dirname, realpath
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges, get_laplacian, to_dense_adj

device = "cuda" if torch.cuda.is_available() else "cpu"

import wandb
wandb.init(project="DEA")
config = wandb.config

class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.layers = torch.nn.ModuleList()
        if config['dataset'] == 'fb' or config['dataset'] =='dblp':
            self.layers.append(GCNConv(in_channels, 128))
            self.layers.append(GCNConv(128, 128))
            self.layers.append(GCNConv(128, 128))
            self.layers.append(GCNConv(128, out_channels))
            self.tr_epochs=201
        else:
            self.layers.append(GCNConv(in_channels, 128))
            self.layers.append(GCNConv(128, out_channels))
            self.tr_epochs=101
        
        self.mlp = nn.Sequential(
            nn.Linear(out_channels * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
          )
  
    def encode(self, x, edge_index, ew = None):
        for conv in self.layers[:-1]:
            x = conv(x, edge_index, ew).relu()
        x = self.layers[-1](x, edge_index, ew)
        return x

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits, edge_index


def gumbel_sigmoid(logits, tau: float = 1):
    gumbels = (
        -torch.empty_like(logits).exponential_().log()
        )
    gumbels = (logits + gumbels) / tau  
    y_soft = gumbels.sigmoid()
    index = torch.nonzero(y_soft>=0.5).flatten()
    y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
    return y_hard - y_soft.detach() + y_soft

def hard_sigmoid(logits):
    index = torch.nonzero(logits.sigmoid()>=0.5).flatten()
    return torch.zeros_like(logits).scatter_(-1, index, 1.0)


def _embeddings(x, edge_idx):
  rows = edge_idx[0]
  cols = edge_idx[1]
  
  row_embeds = x[rows]
  col_embeds = x[cols]
  
  embeds = torch.cat([row_embeds, col_embeds], 1)
  return embeds

acc_auc = []
fairness = []
seeds = [0, 1, 2]
for random_seed in seeds:
        
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    if config['dataset'] in ["citeseer", "cora", "pubmed"]:
        path = osp.join(osp.dirname(osp.realpath('__file__')), "..", "data", config['dataset'])
        dataset = Planetoid(path, config['dataset'], transform=T.NormalizeFeatures())
        data = dataset[0]
        protected_attribute = data.y
        data.train_mask = data.val_mask = data.test_mask = data.y = None
        data = train_test_split_edges(data, val_ratio=0.1, test_ratio=0.2)
        data = data.to(device)
    else:
        
        if config['dataset'] == "dblp":
            dataset_path = join(dirname(realpath("__file__")), "data", "dblp")
            with open(
                join(dataset_path, "author-author.csv"), mode="r", encoding="ISO-8859-1"
                ) as file_name:
                edges = np.genfromtxt(file_name, delimiter=",", dtype=int)

            with open(
                join(dataset_path, "countries.csv"), mode="r", encoding="ISO-8859-1"
                ) as file_name:
                attributes = np.genfromtxt(file_name, delimiter=",", dtype=str)

                protected_attribute = encode_classes(attributes[:, 1])
        
        if config['dataset'] == "fb":
            dataset_path = join(dirname(realpath("__file__")), "data", "facebook")
            with open(
                join(dataset_path, "fb_edges.csv"), mode="r", encoding="ISO-8859-1"
            ) as file_name:
                edges = np.genfromtxt(file_name, delimiter=" ", dtype=int)
            with open(
                join(dataset_path, "fb_features.csv"), mode="r", encoding="ISO-8859-1"
              ) as file_name:
                protected_attribute = np.genfromtxt(file_name, delimiter=" ", dtype=int)[:, 1]

        edge_index=torch.tensor(np.transpose(np.array(edges)), dtype=torch.long)
        L, edge_weight = get_laplacian(edge_index)
        L = to_dense_adj(L).squeeze()

        AV, V = torch.eig(L, eigenvectors=True)
        _, indices= torch.sort(AV[:,0], dim=0, descending=True)
        X = torch.index_select(V, dim=1, index=indices)

        data = Data(x=X, edge_index=edge_index, y=torch.tensor(protected_attribute, dtype=torch.long))
        protected_attribute = data.y
        data.train_mask = data.val_mask = data.test_mask = data.y = None
        data = train_test_split_edges(data, val_ratio=0.1, test_ratio=0.2)
        data = data.to(device)
    
    num_classes = len(np.unique(protected_attribute))
    N = protected_attribute.shape[0]
    Y = torch.LongTensor(protected_attribute).to(device) 
    model = GCN(data.num_features, 128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.007)  

    best_val_perf = test_perf = 0
    for epoch in range(1, model.tr_epochs):        

        model.train()
        optimizer.zero_grad()

        # TRAINING
        neg_edges_tr = negative_sampling(
            edge_index=data.train_pos_edge_index,
            num_nodes=N,
            num_neg_samples=data.train_pos_edge_index.size(1),
        ).to(device)
        
        
        z = model.encode(data.x, data.train_pos_edge_index)
        link_logits, _ = model.decode(                   
            z, data.train_pos_edge_index, neg_edges_tr
        )        
        tr_labels = get_link_labels(
            data.train_pos_edge_index, neg_edges_tr
        ).to(device)
    
        # COMPUTING LOSS
        loss = F.binary_cross_entropy_with_logits(link_logits, tr_labels)
        loss.backward()
        optimizer.step()

        # EVALUATION
        model.eval()
        perfs = []
        for prefix in ["val", "test"]:
            pos_edge_index = data[f"{prefix}_pos_edge_index"]
            neg_edge_index = data[f"{prefix}_neg_edge_index"]
            with torch.no_grad():
                z = model.encode(data.x, data.train_pos_edge_index, None)
                link_logits, edge_idx = model.decode(z, pos_edge_index, neg_edge_index)
            link_probs = link_logits.sigmoid()
            link_labels = get_link_labels(pos_edge_index, neg_edge_index)
            auc = roc_auc_score(link_labels.cpu(), link_probs.cpu())
            perfs.append(auc)

        val_perf, tmp_test_perf = perfs
        if val_perf > best_val_perf:
            best_val_perf = val_perf
            test_perf = tmp_test_perf
        if epoch%10==0:
            log = "Epoch: {:03d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}"
            print(log.format(epoch, loss, best_val_perf, test_perf))

    auc = test_perf            
    cut = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    
    best_acc = 0
    best_cut = cut[0]
    for i in cut:
        acc = accuracy_score(link_labels.cpu(), link_probs.cpu() >= i)
        if acc > best_acc:
            best_acc = acc
            best_cut = i

    optimizer = torch.optim.Adam([
                {'params': model.layers.parameters()},
                {'params': model.mlp.parameters(), 'lr': config['lr_ed']}
            ], lr=config['lr_conv'])

    if config['dyadic'] == 'mixed':
        src_sa = protected_attribute[data.train_pos_edge_index[0]]
        dst_sa = protected_attribute[data.train_pos_edge_index[1]]
        
        sa = torch.zeros(data.train_pos_edge_index.size(1), dtype=torch.float).to(device)

        sa[src_sa != dst_sa] = 1.0  
    
        diff = (sa - sa.mean()) 
    else:
        S = torch.zeros(len(np.unique(protected_attribute)), data.train_pos_edge_index.size(1)).to(device)
        src_sa = protected_attribute[data.train_pos_edge_index[0]]
        dst_sa = protected_attribute[data.train_pos_edge_index[1]]

        for s,i in enumerate(protected_attribute):
            S[i, src_sa == s] = 1.0
            S[i, dst_sa == s] = 1.0
            S[i] -= S[i].mean()        

    
    best_val_perf = test_perf = 0
    temp_schedule = lambda e: 5*((1/5)**(e/config['epochs']))
    for epoch in range(1, 101):
        tau = temp_schedule(epoch)
        with torch.no_grad(): 
            z = model.encode(data.x, data.train_pos_edge_index) 
            embeddings = _embeddings(z, data.train_pos_edge_index)

        model.train()
        optimizer.zero_grad()

        res = model.mlp(embeddings)
        ew = gumbel_sigmoid(res.squeeze(), tau)

        # TRAINING
        z = model.encode(data.x, data.train_pos_edge_index, ew)
        link_logits, _ = model.decode(
            z, data.train_pos_edge_index, neg_edges_tr
        )

        tr_labels = get_link_labels(
            data.train_pos_edge_index, neg_edges_tr
        ).to(device)

        # COMPUTING COVARIANCE
        l_log = link_logits[: data.train_pos_edge_index.size(1)].to(device)
        if config['dyadic'] == 'mixed':
            cov_impact = (diff * (torch.sigmoid(l_log) - best_cut)).mean() 
        else:
            cov_impact = (S * (torch.sigmoid(l_log) - best_cut)).mean(dim=1).sum()

        loss = F.binary_cross_entropy_with_logits(link_logits, tr_labels) + (config['lambda'] * torch.abs(cov_impact))
        loss.backward()
        optimizer.step()

        # EVALUATION
        model.eval()
        perfs = []
        for prefix in ["val", "test"]:
            pos_edge_index = data[f"{prefix}_pos_edge_index"]
            neg_edge_index = data[f"{prefix}_neg_edge_index"]
            with torch.no_grad():
                z = model.encode(data.x, data.train_pos_edge_index, hard_sigmoid(res.squeeze()))
                link_logits, edge_idx = model.decode(z, pos_edge_index, neg_edge_index)
            link_probs = link_logits.sigmoid()
            link_labels = get_link_labels(pos_edge_index, neg_edge_index)
            auc = roc_auc_score(link_labels.cpu(), link_probs.cpu())
            perfs.append(auc)

        val_perf, tmp_test_perf = perfs

        if val_perf > best_val_perf:
            best_val_perf = val_perf
            test_perf = tmp_test_perf
        
        if epoch%10==0:
            log = "Epoch: {:03d}, FTLoss: {:.4f}, FTVal: {:.4f}, FTTest: {:.4f}"
            print(log.format(epoch, loss, best_val_perf, test_perf))

        wandb.log({'FTloss': loss, 'FTval': best_val_perf, 'FTtest': test_perf})

    auc = test_perf
    acc = accuracy_score(link_labels.cpu(), link_probs.cpu() >= best_cut)
    f = prediction_fairness(
        edge_idx.cpu(), link_labels.cpu(), link_probs.cpu() >= best_cut, Y.cpu()
    )
    acc_auc.append([best_acc * 100, auc * 100])
    fairness.append([x * 100 for x in f])

ma = np.mean(np.asarray(acc_auc), axis=0)
mf = np.mean(np.asarray(fairness), axis=0)

wandb.log({'ACC': ma[0], 'AUC': ma[1], 'DP mix': mf[0], 'EoP mix': mf[1], 'DP group': mf[2], 'EoP group': mf[3], 'DP sub': mf[4], 'EoP sub': mf[5]})
