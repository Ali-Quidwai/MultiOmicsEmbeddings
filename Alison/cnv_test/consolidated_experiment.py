#!/usr/bin/env python3

"""
consolidated_experiment.py

A single script for COMMPASS 655 data:
1) Reads CNV broad/focal, Expression (top-N genes), SNV, Fusion, Translocations.
2) Builds WeightedEncoders + Cross-modal attention with a SimCLR-like loss.
3) Trains for some epochs, logs train/val loss.
4) Clusters the final embeddings with UMAP+HDBSCAN & t-SNE+KMeans.
5) Saves embeddings and cluster labels to CSV, plus silhouette scores.

Usage Example:
python consolidated_experiment.py \
  --data_dir "/Users/alisonpark/Documents/MultiOmicsEmbeddings/Final_Scripts_and_Results_on655/COMMPASS_655_data" \
  --cnv_broad_file COMMPASS_655_cnvBroad.csv \
  --cnv_focal_file COMMPASS_655_cnvFocal.csv \
  --expr_file COMMPASS_655_gene_expression.csv \
  --snv_file COMMPASS_655_SNV.csv \
  --fusion_file COMMPASS_655_fusion_gene.csv \
  --trans_file COMMPASS_655_translocations.csv \
  --top_genes 2000 \
  --epochs 5 \
  --batch_size 4 \
  --modality_subset all \
  --n_clusters 8 \
  --hdbscan_min_cluster_size 5 \
  --output_prefix "test_run_n8_hdb5"
"""

import argparse
import os
#import numpy as np
import pandas as pd
#import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler

# For dimension reduction & clustering
import umap
import hdbscan
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt

import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


###############################################################################
# 1) Dataset
###############################################################################
class MyelomaDataset(Dataset):
    """
    Loads CNV broad/focal, expression, SNV, fusion, trans, etc.
    - top_genes for expression
    - Optionally align with a cytogenetics file
    """
    def __init__(
        self,
        data_dir="data/",
        top_genes=2000,
        cnv_broad_file="COMMPASS_655_cnvBroad.csv",
        cnv_focal_file="COMMPASS_655_cnvFocal.csv",
        expr_file="COMMPASS_655_gene_expression.csv",
        snv_file="COMMPASS_655_SNV.csv",
        fusion_file="COMMPASS_655_fusion_gene.csv",
        trans_file="COMMPASS_655_translocations.csv",
        cytogenetics_file=None
    ):
        super().__init__()

        # --- Load data ---
        self.cnv_broad_df = pd.read_csv(os.path.join(data_dir, cnv_broad_file))
        self.cnv_focal_df = pd.read_csv(os.path.join(data_dir, cnv_focal_file))
        self.expr_df      = pd.read_csv(os.path.join(data_dir, expr_file))
        self.snv_df       = pd.read_csv(os.path.join(data_dir, snv_file))
        self.fusion_df    = pd.read_csv(os.path.join(data_dir, fusion_file))
        self.trans_df     = pd.read_csv(os.path.join(data_dir, trans_file))

        # Identify sample IDs
        if "Unnamed: 0" in self.cnv_broad_df.columns:
            self.patient_ids = self.cnv_broad_df["Unnamed: 0"].values
        else:
            self.patient_ids = np.arange(len(self.cnv_broad_df))

        # Optional cytogenetics
        self.cytogenetics = {}
        if cytogenetics_file and os.path.exists(os.path.join(data_dir, cytogenetics_file)):
            cyto_df = pd.read_csv(os.path.join(data_dir, cytogenetics_file))
            for _, row in cyto_df.iterrows():
                pid = row["patient_id"]
                self.cytogenetics[pid] = dict(row)

        def drop_id(df):
            return df.drop(columns=["Unnamed: 0"], errors="ignore") if "Unnamed: 0" in df.columns else df

        # Drop ID columns if present
        self.cnv_broad_df = drop_id(self.cnv_broad_df)
        self.cnv_focal_df = drop_id(self.cnv_focal_df)
        self.expr_df      = drop_id(self.expr_df)
        self.snv_df       = drop_id(self.snv_df)
        self.fusion_df    = drop_id(self.fusion_df)
        self.trans_df     = drop_id(self.trans_df)

        # Scale CNV broad/focal
        self.cnv_broad_arr = self._scale(self.cnv_broad_df)
        self.cnv_focal_arr = self._scale(self.cnv_focal_df)

        # Expression: top-N variable gene selection
        if top_genes > 0 and top_genes < self.expr_df.shape[1]:
            var_series = self.expr_df.var(axis=0)
            top_idx = var_series.nlargest(top_genes).index
            self.expr_df = self.expr_df[top_idx]
            print(f"Selected top {top_genes} variable genes out of {var_series.shape[0]} total.")
        else:
            print(f"No expression feature selection or top_genes >= #genes => using {self.expr_df.shape[1]} columns.")
            
        gene_list_out = os.path.join(data_dir, f"selected_top{top_genes}_genes.csv")
        if not os.path.exists(gene_list_out):
            pd.Series(top_idx).to_csv(gene_list_out, index=False, header=False)
            print(f"Saved top {top_genes} genes to: {gene_list_out}")
        else:
            print(f"Top {top_genes} genes already saved at: {gene_list_out}")


        self.expr_arr = self._scale(self.expr_df)

        # SNV, fusion, trans => float32
        self.snv_arr    = self.snv_df.astype(np.float32).values
        self.fusion_arr = self.fusion_df.astype(np.float32).values
        self.trans_arr  = self.trans_df.astype(np.float32).values

        # Check shapes
        expected_n = len(self.patient_ids)
        for nm, arr in [
            ("cnv_broad_arr", self.cnv_broad_arr),
            ("cnv_focal_arr", self.cnv_focal_arr),
            ("expr_arr",      self.expr_arr),
            ("snv_arr",       self.snv_arr),
            ("fusion_arr",    self.fusion_arr),
            ("trans_arr",     self.trans_arr)
        ]:
            if arr.shape[0] != expected_n:
                raise ValueError(f"{nm} has {arr.shape[0]} rows, expected {expected_n}")

    def _scale(self, df):
        sc = StandardScaler()
        return sc.fit_transform(df).astype(np.float32)

    def __len__(self):
        return len(self.cnv_broad_df)

    def __getitem__(self, idx):
        return {
            "patient_id": self.patient_ids[idx],  # string or int
            "cnv_broad":      torch.from_numpy(self.cnv_broad_arr[idx]),
            "cnv_focal":      torch.from_numpy(self.cnv_focal_arr[idx]),
            "expression":     torch.from_numpy(self.expr_arr[idx]),
            "snv":            torch.from_numpy(self.snv_arr[idx]),
            "fusion":         torch.from_numpy(self.fusion_arr[idx]),
            "trans":          torch.from_numpy(self.trans_arr[idx])
        }

###############################################################################
# 2) SimCLR-like Augmentation
###############################################################################
def simclr_augment(batch_item, augment_prob=0.5, scale_jitter=0.1):
    """
    Returns (v1, v2) with random scale + noise for each numeric field.
    """
    v1 = {}
    v2 = {}
    for k in batch_item:
        if k == "patient_id":
            # Just copy
            v1[k] = batch_item[k]
            v2[k] = batch_item[k]
        else:
            arr = batch_item[k].clone()
            arr2 = arr.clone()
            # random scale for arr
            if torch.rand(1).item() < augment_prob:
                factor = 1.0 + scale_jitter*(2*torch.rand(1).item()-1)
                arr = arr*factor
            # noise
            noise = scale_jitter*torch.randn_like(arr)
            arr_aug1 = arr + noise

            # random scale for arr2
            if torch.rand(1).item() < augment_prob:
                factor2 = 1.0 + scale_jitter*(2*torch.rand(1).item()-1)
                arr2 = arr2*factor2
            noise2 = scale_jitter*torch.randn_like(arr2)
            arr_aug2 = arr2 + noise2

            v1[k] = arr_aug1
            v2[k] = arr_aug2
    return v1, v2

###############################################################################
# 3) WeightedEncoders + Cross-Modal Attention
###############################################################################
class WeightedEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        if input_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            self.net = None
        # trainable weight
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        if self.input_dim == 0 or self.net is None:
            return None
        out = self.net(x)
        out = out * self.weight
        return out

class CrossModalAttnBlock(nn.Module):
    def __init__(self, dim=128, num_heads=4, dropout=0.2):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x => [M, B, dim]
        attn_out, _ = self.attn(x, x, x)
        out = x + self.drop(attn_out)
        out = self.norm(out)
        return out

class MyelomaMultiModalModel(nn.Module):
    """
    WeightedEncoders => cross-modal attn => flatten => final MLP => [B, emb_dim].
    """
    def __init__(self, config):
        super().__init__()
        hd = config["hidden_dim"]
        dr = config["dropout"]

        self.cnv_broad_enc = WeightedEncoder(config["cnv_broad_dim"], hd, dr)
        self.cnv_focal_enc = WeightedEncoder(config["cnv_focal_dim"], hd, dr)
        self.expr_enc      = WeightedEncoder(config["expression_dim"], hd, dr)
        self.snv_enc       = WeightedEncoder(config["snv_dim"], hd, dr)
        self.fusion_enc    = WeightedEncoder(config["fusion_dim"], hd, dr)
        self.trans_enc     = WeightedEncoder(config["trans_dim"], hd, dr)

        self.num_heads = config["num_heads"]
        self.block1 = CrossModalAttnBlock(dim=hd, num_heads=self.num_heads, dropout=dr)
        self.block2 = CrossModalAttnBlock(dim=hd, num_heads=self.num_heads, dropout=dr)

        emb_dim = config["final_emb_dim"]
        self.out_mlp = nn.Sequential(
            nn.Linear(hd*config["num_modalities_used"], emb_dim),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Linear(emb_dim, emb_dim)
        )
        self.out_norm = nn.LayerNorm(emb_dim)

    def forward(self, batch):
        cb = self.encode_one(self.cnv_broad_enc, batch["cnv_broad"])
        cf = self.encode_one(self.cnv_focal_enc, batch["cnv_focal"])
        ex = self.encode_one(self.expr_enc,      batch["expression"])
        sn = self.encode_one(self.snv_enc,       batch["snv"])
        fu = self.encode_one(self.fusion_enc,    batch["fusion"])
        tr = self.encode_one(self.trans_enc,     batch["trans"])

        mod_list = []
        for m_ in [cb, cf, ex, sn, fu, tr]:
            if m_ is not None:
                mod_list.append(m_)

        if len(mod_list) == 0:
            # no data => return zero
            return batch["cnv_broad"].new_zeros((batch["cnv_broad"].shape[0],1))

        # shape => [M, B, hd]
        X = torch.stack(mod_list, dim=0)
        X = self.block1(X)
        X = self.block2(X)
        # flatten => [B, M*hd]
        X = X.transpose(0,1).reshape(X.size(1), -1)
        X = self.out_mlp(X)
        X = self.out_norm(X)
        return X

    def encode_one(self, encoder, tensor):
        if encoder.input_dim == 0:
            return None
        return encoder(tensor)

###############################################################################
# 4) SimCLR Loss + Training
###############################################################################
def simclr_loss(z1, z2, temperature=0.1):
    """
    Classic SimCLR InfoNCE
    z1,z2 => [B, emb_dim], we produce 2B embeddings
    """
    bsz = z1.size(0)
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    cat = torch.cat([z1, z2], dim=0)  # => [2B, emb_dim]
    sim = torch.matmul(cat, cat.T)    # => [2B, 2B]
    mask = torch.eye(2*bsz, device=z1.device).bool()
    sim = sim / temperature
    sim[mask] = -9999

    pos = torch.cat([
        sim[i, i+bsz].unsqueeze(0) for i in range(bsz)
    ] + [
        sim[i, i-bsz].unsqueeze(0) for i in range(bsz, 2*bsz)
    ], dim=0)

    logsum = torch.logsumexp(sim, dim=1)
    loss = - torch.mean(pos - logsum)
    return loss

def forward_one_sample(model, item_dict, device):
    dd = {}
    for k,v in item_dict.items():
        if isinstance(v, torch.Tensor):
            dd[k] = v.to(device)
        else:
            dd[k] = v
    return model(dd)

def run_epoch_simclr(model, loader, optimizer, device, is_train=True, augment_prob=0.5):
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    for batch_in in loader:
        # batch_in["patient_id"] is a list, numeric fields are Tensors
        bsz = len(batch_in["patient_id"])
        z1_list = []
        z2_list = []

        # per sample
        for i in range(bsz):
            sample_dict = {}
            for k in batch_in:
                if isinstance(batch_in[k], list):
                    sample_dict[k] = batch_in[k][i]
                else:
                    # shape [bsz, dim], index i => shape [dim], unsqueeze => [1, dim]
                    sample_dict[k] = batch_in[k][i].unsqueeze(0)

            # augment
            v1, v2 = simclr_augment(sample_dict, augment_prob=augment_prob)
            z1 = forward_one_sample(model, v1, device)
            z2 = forward_one_sample(model, v2, device)
            z1_list.append(z1)
            z2_list.append(z2)

        z1_batch = torch.cat(z1_list, dim=0)  # => [bsz, emb_dim]
        z2_batch = torch.cat(z2_list, dim=0)  # => [bsz, emb_dim]

        loss = simclr_loss(z1_batch, z2_batch, temperature=0.1)
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)

###############################################################################
# 5) Clustering + Saving
###############################################################################
def cluster_umap_hdbscan(embeddings, min_cluster_size=12):
    n = embeddings.shape[0]
    if n<2:
        return np.array([-1]*n), None, None
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    emb_2d  = reducer.fit_transform(embeddings)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(emb_2d)

    if len(np.unique(labels))<2:
        sil = None
    else:
        sil = silhouette_score(embeddings, labels)
    return labels, sil, emb_2d

def cluster_tsne_kmeans(embeddings, n_clusters=12):
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
    n = embeddings.shape[0]
    if n<2:
        return np.array([-1]*n), None, None
    ts = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    emb_2d = ts.fit_transform(embeddings)

    if n>= n_clusters>1:
        km = KMeans(n_clusters=n_clusters, random_state=42)
        labels = km.fit_predict(embeddings)
        sil = silhouette_score(embeddings, labels)
    else:
        labels = np.zeros(n, dtype=int)
        sil = None
    return labels, sil, emb_2d


###############################################################################
# 6) Main
###############################################################################
def main():
    set_seed(42)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--cnv_broad_file", type=str, default="COMMPASS_655_cnvBroad.csv")
    parser.add_argument("--cnv_focal_file", type=str, default="COMMPASS_655_cnvFocal.csv")
    parser.add_argument("--expr_file",      type=str, default="COMMPASS_655_gene_expression.csv")
    parser.add_argument("--snv_file",       type=str, default="COMMPASS_655_SNV.csv")
    parser.add_argument("--fusion_file",    type=str, default="COMMPASS_655_fusion_gene.csv")
    parser.add_argument("--trans_file",     type=str, default="COMMPASS_655_translocations.csv")
    parser.add_argument("--top_genes", type=int, default=2000)
    parser.add_argument("--epochs",   type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr",       type=float, default=1e-4)
    parser.add_argument("--hidden_dim",   type=int, default=128)
    parser.add_argument("--dropout",  type=float, default=0.2)
    parser.add_argument("--final_emb_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--modality_subset", type=str, default="all")
    parser.add_argument("--output_prefix", type=str, default="myeloma_run",
                        help="Prefix for output CSV filenames.")
    parser.add_argument("--n_clusters", type=int, default=12,
                    help="Number of clusters for t-SNE + KMeans")
    parser.add_argument("--hdbscan_min_cluster_size", type=int, default=12,
                    help="Min cluster size for UMAP + HDBSCAN")

    args = parser.parse_args()

    # Build dataset
    dataset = MyelomaDataset(
        data_dir=args.data_dir,
        top_genes=args.top_genes,
        cnv_broad_file=args.cnv_broad_file,
        cnv_focal_file=args.cnv_focal_file,
        expr_file=args.expr_file,
        snv_file=args.snv_file,
        fusion_file=args.fusion_file,
        trans_file=args.trans_file
    )

    sample = dataset[0]  # first sample
    def get_dim(t):
        return t.shape[0] if t.dim()==1 else t.shape[1]

    # figure out dims
    full_dims = {
        "cnv_broad_dim": get_dim(sample["cnv_broad"]),
        "cnv_focal_dim": get_dim(sample["cnv_focal"]),
        "expression_dim": get_dim(sample["expression"]),
        "snv_dim":       get_dim(sample["snv"]),
        "fusion_dim":    get_dim(sample["fusion"]),
        "trans_dim":     get_dim(sample["trans"])
    }

    # subset
    if args.modality_subset.lower() == "all":
        used_dims = full_dims
        count = 6
    else:
        subset = [m.strip().lower() for m in args.modality_subset.split(",")]
        used_dims = {}
        mod_list = ["cnv_broad","cnv_focal","expression","snv","fusion","trans"]
        count = 0
        for m_ in mod_list:
            key = f"{m_}_dim"
            if m_ in subset:
                used_dims[key] = full_dims[key]
                count +=1
            else:
                used_dims[key] =0

    config = {
        "cnv_broad_dim": used_dims["cnv_broad_dim"],
        "cnv_focal_dim": used_dims["cnv_focal_dim"],
        "expression_dim": used_dims["expression_dim"],
        "snv_dim": used_dims["snv_dim"],
        "fusion_dim": used_dims["fusion_dim"],
        "trans_dim": used_dims["trans_dim"],
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "num_heads": args.num_heads,
        "final_emb_dim": args.final_emb_dim,
        "num_modalities_used": count
    }

    print(f"\n--- Single run => {{'modality_dims': {used_dims}, 'lr': {args.lr}, "
          f"'batch_size': {args.batch_size}, 'epochs': {args.epochs}}}")

    model = MyelomaMultiModalModel(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    from torch.optim import Adam
    optim = Adam(model.parameters(), lr=args.lr)

    # Train/val split
    n_total = len(dataset)
    n_train = int(0.8*n_total)
    n_val   = n_total - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    # training loop
    for ep in range(args.epochs):
        tr_loss = run_epoch_simclr(model, train_loader, optim, device, is_train=True,  augment_prob=0.5)
        val_loss= run_epoch_simclr(model, val_loader,   optim, device, is_train=False, augment_prob=0.5)
        print(f"[single_run] epoch {ep+1}/{args.epochs} => train={tr_loss:.4f} val={val_loss:.4f}")

    # final embeddings
    all_ids = []
    all_embs= []
    model.eval()
    full_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    with torch.no_grad():
        for batch_in in full_loader:
            bsz = len(batch_in["patient_id"])
            Zall=[]
            for i in range(bsz):
                single_item={}
                for k in batch_in:
                    if isinstance(batch_in[k], list):
                        single_item[k] = batch_in[k][i]
                    else:
                        single_item[k] = batch_in[k][i].unsqueeze(0)
                z= forward_one_sample(model, single_item, device)
                Zall.append(z)
            Zcat= torch.cat(Zall, dim=0).cpu().numpy()
            all_embs.append(Zcat)
            for pid_ in batch_in["patient_id"]:
                all_ids.append(pid_)
    embs= np.concatenate(all_embs, axis=0)
    print()

    # clustering
    labels_umap, sil_umap, emb2d_umap= cluster_umap_hdbscan(embs)
    labels_tsne, sil_tsne, emb2d_tsne= cluster_tsne_kmeans(embs)

    print(f"UMAP+HDBSCAN silhouette={sil_umap}, t-SNE+KMeans silhouette={sil_tsne}")

    # save embeddings
    emb_outfile = f"{args.output_prefix}_embeddings.csv"
    df_emb= pd.DataFrame(embs)
    df_emb.insert(0, "patient_id", all_ids)
    df_emb.to_csv(emb_outfile, index=False)
    print(f"Saved embeddings to {emb_outfile}")

    # save cluster assignments
    df_clust= pd.DataFrame({
        "patient_id": all_ids,
        "umap_hdbscan_label": labels_umap,
        "tsne_kmeans_label":  labels_tsne
    })
    clust_outfile= f"{args.output_prefix}_clusters.csv"
    df_clust.to_csv(clust_outfile, index=False)
    print(f"Saved cluster labels to {clust_outfile}")

    # optionally save 2D coords from UMAP/TSNE
    coords_outfile= f"{args.output_prefix}_umap_tsne_coords.csv"
    df_coords= pd.DataFrame({
        "patient_id": all_ids,
        "umap_x": emb2d_umap[:,0],
        "umap_y": emb2d_umap[:,1],
        "tsne_x": emb2d_tsne[:,0],
        "tsne_y": emb2d_tsne[:,1]
    })
    df_coords.to_csv(coords_outfile, index=False)
    print(f"Saved 2D coordinates to {coords_outfile}")

    # ✅ Save silhouette scores as metrics CSV
    metrics_df = pd.DataFrame([{
        "output_prefix": args.output_prefix,
        "silhouette_umap": sil_umap,
        "silhouette_tsne": sil_tsne,
        "num_umap_clusters": len(np.unique(labels_umap)),
        "num_tsne_clusters": len(np.unique(labels_tsne))
    }])
    metrics_df.to_csv(f"{args.output_prefix}_metrics.csv", index=False)
    print(f"Saved silhouette metrics to {args.output_prefix}_metrics.csv")


if __name__ == "__main__":
    main()

