#!/usr/bin/env python3

"""
myeloma_single_script.py

This single script demonstrates:
 1) Loading data (CNV Broad, CNV Focal, Gene Expression, SNV, Fusion, Translocations)
 2) Building a simple multi-modal model (MLP for each modality)
 3) Training with a basic contrastive loss
 4) Evaluating embeddings (t-SNE + a quick cluster approach)

Requirements:
 - The CSV files must be in your "data/" folder with these names (adapt as needed):
   data/COMMPASS_184_cnvBroad.csv
   data/COMMPASS_184_cnvFocal.csv
   data/COMMPASS_184_gene_expression_scaled.csv
   data/COMMPASS_184_snv.csv
   data/COMMPASS_184_fusion_gene.csv
   data/COMMPASS_184_translocations.csv

Usage:
  python myeloma_single_script.py --epochs 5 --batch_size 16
"""

import argparse
import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 1) DATASET DEFINITION
class MyelomaDataset(Dataset):
    def __init__(
        self,
        data_dir="data/",
        cnv_broad_file="COMMPASS_184_cnvBroad.csv",
        cnv_focal_file="COMMPASS_184_cnvFocal.csv",
        expr_file="COMMPASS_184_gene_expression_scaled.csv",
        snv_file="COMMPASS_184_snv.csv",
        fusion_file="COMMPASS_184_fusion_gene.csv",
        trans_file="COMMPASS_184_translocations.csv"
    ):
        """
        Loads each CSV, drops 'Unnamed: 0' if present, then scales or casts the data.
        We'll store:
         self.cnv_broad_arr
         self.cnv_focal_arr
         self.expr_arr
         self.snv_arr
         self.fusion_arr
         self.trans_arr
         self.patient_ids
        """
        # --- Load dataframes ---
        self.cnv_broad_df = pd.read_csv(os.path.join(data_dir, cnv_broad_file))
        self.cnv_focal_df = pd.read_csv(os.path.join(data_dir, cnv_focal_file))
        self.expr_df = pd.read_csv(os.path.join(data_dir, expr_file))
        self.snv_df = pd.read_csv(os.path.join(data_dir, snv_file))
        self.fusion_df = pd.read_csv(os.path.join(data_dir, fusion_file))
        self.trans_df = pd.read_csv(os.path.join(data_dir, trans_file))

        # --- Extract patient IDs from one of them (assuming same order) ---
        if "Unnamed: 0" in self.cnv_broad_df.columns:
            self.patient_ids = self.cnv_broad_df["Unnamed: 0"].values
        else:
            self.patient_ids = np.arange(len(self.cnv_broad_df))

        # --- Drop ID column if present ---
        def drop_id(df):
            return df.drop(columns=["Unnamed: 0"], errors="ignore") if "Unnamed: 0" in df.columns else df

        self.cnv_broad_df = drop_id(self.cnv_broad_df)
        self.cnv_focal_df = drop_id(self.cnv_focal_df)
        self.expr_df      = drop_id(self.expr_df)
        self.snv_df       = drop_id(self.snv_df)
        self.fusion_df    = drop_id(self.fusion_df)
        self.trans_df     = drop_id(self.trans_df)

        # --- Scale continuous data (CNV, Expression) ---
        self.cnv_broad_arr = self._scale(self.cnv_broad_df)
        self.cnv_focal_arr = self._scale(self.cnv_focal_df)
        self.expr_arr      = self._scale(self.expr_df)

        # --- SNV, Fusion, Translocations => cast to float32 (binary) ---
        self.snv_arr    = self.snv_df.astype(np.float32).values
        self.fusion_arr = self.fusion_df.astype(np.float32).values
        self.trans_arr  = self.trans_df.astype(np.float32).values

        # Confirm all have the same # of rows
        expected_n = len(self.patient_ids)
        for name, arr in [
            ("cnv_broad_arr", self.cnv_broad_arr),
            ("cnv_focal_arr", self.cnv_focal_arr),
            ("expr_arr",      self.expr_arr),
            ("snv_arr",       self.snv_arr),
            ("fusion_arr",    self.fusion_arr),
            ("trans_arr",     self.trans_arr)
        ]:
            if arr.shape[0] != expected_n:
                raise ValueError(f"{name} has {arr.shape[0]} rows, expected {expected_n}")

    def _scale(self, df):
        scaler = StandardScaler()
        return scaler.fit_transform(df).astype(np.float32)

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        return {
            "patient_id": self.patient_ids[idx],
            "cnv_broad":      torch.from_numpy(self.cnv_broad_arr[idx]),
            "cnv_focal":      torch.from_numpy(self.cnv_focal_arr[idx]),
            "expression":     torch.from_numpy(self.expr_arr[idx]),
            "snv":            torch.from_numpy(self.snv_arr[idx]),
            "fusion_gene":    torch.from_numpy(self.fusion_arr[idx]),
            "translocations": torch.from_numpy(self.trans_arr[idx]),
        }


# 2) MODEL DEFINITIONS
class MLPEncoder(nn.Module):
    """
    Simple MLP to encode one data modality.
    """
    def __init__(self, input_dim, hidden_dims=[256, 128], dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for hd in hidden_dims:
            layers.append(nn.Linear(prev, hd))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = hd
        self.encoder = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]

    def forward(self, x):
        return self.encoder(x)

class MyelomaMultiModalModel(nn.Module):
    """
    Encodes each modality via a separate MLP, then concatenates everything into a final embedding.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Each encoder has an input_dim that matches the CSV shape
        self.cnv_broad_enc = MLPEncoder(config["cnv_broad_dim"], [256,128], config["dropout"])
        self.cnv_focal_enc = MLPEncoder(config["cnv_focal_dim"], [256,128], config["dropout"])
        self.expr_enc      = MLPEncoder(config["expr_dim"],      [512,256], config["dropout"])
        self.snv_enc       = MLPEncoder(config["snv_dim"],       [128,64],  config["dropout"])
        self.fusion_enc    = MLPEncoder(config["fusion_dim"],    [128,64],  config["dropout"])
        self.trans_enc     = MLPEncoder(config["trans_dim"],     [64,32],   config["dropout"])

        # Calculate the combined dimension after all encoders
        total_dim = (self.cnv_broad_enc.output_dim +
                     self.cnv_focal_enc.output_dim +
                     self.expr_enc.output_dim      +
                     self.snv_enc.output_dim       +
                     self.fusion_enc.output_dim    +
                     self.trans_enc.output_dim)

        # Final unify layer
        self.final_layer = nn.Sequential(
            nn.Linear(total_dim, config["final_emb_dim"] * 2),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["final_emb_dim"] * 2, config["final_emb_dim"])
        )
        self.norm = nn.LayerNorm(config["final_emb_dim"])

    def forward(self, batch):
        cb = self.cnv_broad_enc(batch["cnv_broad"].float())
        cf = self.cnv_focal_enc(batch["cnv_focal"].float())
        ex = self.expr_enc(batch["expression"].float())
        sn = self.snv_enc(batch["snv"].float())
        fu = self.fusion_enc(batch["fusion_gene"].float())
        tr = self.trans_enc(batch["translocations"].float())

        combined = torch.cat([cb, cf, ex, sn, fu, tr], dim=1)
        emb = self.final_layer(combined)
        emb = self.norm(emb)
        return emb


# 3) TRAINING & EVALUATION
def contrastive_loss(embeddings):
    """
    Very simple contrastive-like loss:
     - treat each embedding as the 'positive' for itself
     - all other embeddings in the batch are negatives
    """
    emb = F.normalize(embeddings, dim=-1)
    sim = torch.matmul(emb, emb.T)  # [batch_size, batch_size]
    sim = sim / 0.1  # temperature
    labels = torch.arange(sim.size(0), device=sim.device)
    return F.cross_entropy(sim, labels)

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    epoch_loss = 0.0
    for batch in dataloader:
        # Move tensors to device
        for k in batch:
            if k != "patient_id":
                batch[k] = batch[k].to(device)

        embeddings = model(batch)
        loss = contrastive_loss(embeddings)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

def evaluate_embeddings(model, dataloader, device):
    """
    We'll collect embeddings from the entire dataset, then do a t-SNE + KMeans as a sanity check.
    """
    model.eval()
    all_ids = []
    all_embs = []

    with torch.no_grad():
        for batch in dataloader:
            pid = batch["patient_id"]
            for k in batch:
                if k != "patient_id":
                    batch[k] = batch[k].to(device)
            emb = model(batch)
            all_ids.extend(pid.numpy().tolist() if not isinstance(pid[0], str) else pid)
            all_embs.append(emb.cpu())

    all_embs = torch.cat(all_embs, dim=0).numpy()
    return all_ids, all_embs


def visualize_tsne(ids, embeddings):
    """
    Quick t-SNE to see if there's any structure. Then cluster with KMeans=5.
    """
    if embeddings.shape[0] < 2:
        print("Not enough embeddings to run t-SNE.")
        return

    tsne = TSNE(n_components=2, random_state=42)
    emb_2d = tsne.fit_transform(embeddings)

    # KMeans to color points
    kmeans = KMeans(n_clusters=5, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Plot
    plt.figure(figsize=(10,7))
    plt.scatter(emb_2d[:,0], emb_2d[:,1], c=cluster_labels, cmap='tab10', alpha=0.6)
    for i, pid in enumerate(ids):
        plt.text(emb_2d[i,0], emb_2d[i,1], str(pid), fontsize=8, alpha=0.7)
    plt.colorbar(label="Cluster Label")
    plt.title("t-SNE Visualization of Myeloma Embeddings")
    plt.show()


# 4) MAIN
def main(args):
    # 1. Build dataset & loader
    dataset = MyelomaDataset(data_dir="data/")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 2. Infer input dims from dataset's arrays
    # We'll do a quick check on the first sample
    sample = dataset[0]
    config = {
        "cnv_broad_dim": sample["cnv_broad"].shape[0],
        "cnv_focal_dim": sample["cnv_focal"].shape[0],
        "expr_dim":      sample["expression"].shape[0],
        "snv_dim":       sample["snv"].shape[0],
        "fusion_dim":    sample["fusion_gene"].shape[0],
        "trans_dim":     sample["translocations"].shape[0],
        "final_emb_dim": 128,
        "dropout": 0.2
    }

    print("Detected input dims:", config)

    # 3. Create model & optimizer
    model = MyelomaMultiModalModel(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 4. Train
    for epoch in range(args.epochs):
        loss_val = train_one_epoch(model, dataloader, optimizer, device)
        print(f"Epoch [{epoch+1}/{args.epochs}] Loss: {loss_val:.4f}")

    # 5. Evaluate => get embeddings & do t-SNE
    ids, embs = evaluate_embeddings(model, dataloader, device)
    visualize_tsne(ids, embs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    main(args)


