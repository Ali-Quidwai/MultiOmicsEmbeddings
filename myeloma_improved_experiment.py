#!/usr/bin/env python3

"""
myeloma_improved_experiment.py

Key improvements beyond your previous script:
1) Feature selection on expression (top N variable genes).
2) Modality weighting: Each modality's MLP is multiplied by a learnable scalar.
3) Optionally chunk-based approach for expression to handle 16k+ features.

Usage:
  python myeloma_improved_experiment.py --epochs 5 --batch_size 16 --top_genes 2000 --modality_subset all
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


###############################################################################
# 1) DATASET with Expression Feature Selection
###############################################################################
class MyelomaDataset(Dataset):
    """
    Loads CSVs for CNV broad/focal, expression, SNV, etc.
    Optionally does top-N gene selection for expression to reduce dimension.
    """
    def __init__(self,
                 data_dir="data/",
                 top_genes=2000,
                 cnv_broad_file="COMMPASS_184_cnvBroad.csv",
                 cnv_focal_file="COMMPASS_184_cnvFocal.csv",
                 expr_file="COMMPASS_184_gene_expression_scaled.csv",
                 snv_file="COMMPASS_184_snv.csv",
                 fusion_file="COMMPASS_184_fusion_gene.csv",
                 trans_file="COMMPASS_184_translocations.csv"):
        super().__init__()
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

        # Drop ID columns
        def drop_id(df):
            return df.drop(columns=["Unnamed: 0"], errors="ignore") if "Unnamed: 0" in df.columns else df

        self.cnv_broad_df = drop_id(self.cnv_broad_df)
        self.cnv_focal_df = drop_id(self.cnv_focal_df)
        self.expr_df      = drop_id(self.expr_df)
        self.snv_df       = drop_id(self.snv_df)
        self.fusion_df    = drop_id(self.fusion_df)
        self.trans_df     = drop_id(self.trans_df)

        # Scale CNV data
        self.cnv_broad_arr = self._scale(self.cnv_broad_df)
        self.cnv_focal_arr = self._scale(self.cnv_focal_df)

        # Expression: top-N variable gene selection
        if top_genes > 0 and top_genes < self.expr_df.shape[1]:
            # compute variance row-wise or col-wise (depending on how the CSV is arranged).
            # Typically each row is a sample, each column is a gene => we do df.var(axis=0).
            variances = self.expr_df.var(axis=0)
            # pick top genes
            top_indices = variances.nlargest(top_genes).index
            # subselect those columns
            self.expr_df = self.expr_df[top_indices]
            print(f"Selected top {top_genes} variable genes out of {variances.shape[0]} total.")
            # optionally, print variance stats
        else:
            print(f"No gene selection or top_genes >= #genes => using full expression: {self.expr_df.shape[1]} columns")

        # final scale
        self.expr_arr = self._scale(self.expr_df)

        # SNV, fusion, trans => float32
        self.snv_arr    = self.snv_df.astype(np.float32).values
        self.fusion_arr = self.fusion_df.astype(np.float32).values
        self.trans_arr  = self.trans_df.astype(np.float32).values

        # consistency checks
        expected_n = len(self.patient_ids)
        for (name, arr) in [
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
            "patient_id":     self.patient_ids[idx],
            "cnv_broad":      torch.from_numpy(self.cnv_broad_arr[idx]),
            "cnv_focal":      torch.from_numpy(self.cnv_focal_arr[idx]),
            "expression":     torch.from_numpy(self.expr_arr[idx]),
            "snv":            torch.from_numpy(self.snv_arr[idx]),
            "fusion_gene":    torch.from_numpy(self.fusion_arr[idx]),
            "translocations": torch.from_numpy(self.trans_arr[idx]),
        }

###############################################################################
# 2) IMPROVED MODEL
###############################################################################

class WeightedModalityAttentionEncoder(nn.Module):
    """
    Similar to your MLP->Self-Attn approach, but we add a learnable scalar "modality_weight."
    This scalar is multiplied after we get the final embedding from that modality.
    So if the model finds e.g. expression more useful, it can increase that weight.
    """
    def __init__(self, input_dim, projection_dim=256, num_heads=4, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        if input_dim == 0:
            self.output_dim = 0
            # dummy
        else:
            self.initial_proj = nn.Sequential(
                nn.Linear(input_dim, projection_dim),
                nn.LayerNorm(projection_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.attention = nn.MultiheadAttention(embed_dim=projection_dim,
                                                   num_heads=num_heads,
                                                   dropout=dropout)
            self.layer_norm = nn.LayerNorm(projection_dim)
            self.dropout = nn.Dropout(dropout)
            self.output_dim = projection_dim

        # the trainable scalar
        self.modality_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        if self.input_dim == 0:
            # produce zero
            batch_size = x.shape[0]
            return x.new_zeros((batch_size, 0))
        else:
            out = self.initial_proj(x)   # => [B, proj_dim]
            out = out.unsqueeze(0)       # => [1, B, proj_dim]
            attn_out, _ = self.attention(out, out, out)
            out = out + self.dropout(attn_out)
            out = self.layer_norm(out)
            out = out.squeeze(0)         # => [B, proj_dim]
            # multiply by scalar weight
            out = out * self.modality_weight
            return out


class MyelomaImprovedModel(nn.Module):
    """
    Weighted version + cross-modal attention.
    Optionally chunk-based approach for expression if we wanted that.
    For brevity, we skip chunk-based code, but you can adapt it similarly.
    """
    def __init__(self, config):
        super().__init__()
        d_attn = config["attn_dim"]
        n_h = config["num_heads"]
        dr = config["dropout"]

        # encoders
        self.cnv_broad_enc = WeightedModalityAttentionEncoder(config["cnv_broad_dim"], d_attn, n_h, dr)
        self.cnv_focal_enc = WeightedModalityAttentionEncoder(config["cnv_focal_dim"], d_attn, n_h, dr)
        self.expr_enc      = WeightedModalityAttentionEncoder(config["expr_dim"],       d_attn, n_h, dr)
        self.snv_enc       = WeightedModalityAttentionEncoder(config["snv_dim"],        d_attn, n_h, dr)
        self.fusion_enc    = WeightedModalityAttentionEncoder(config["fusion_dim"],     d_attn, n_h, dr)
        self.trans_enc     = WeightedModalityAttentionEncoder(config["trans_dim"],      d_attn, n_h, dr)

        self.cross_attn = nn.MultiheadAttention(d_attn, n_h, dropout=dr)

        # figure out how many modalities actually have >0
        out_dims = [
            self.cnv_broad_enc.output_dim,
            self.cnv_focal_enc.output_dim,
            self.expr_enc.output_dim,
            self.snv_enc.output_dim,
            self.fusion_enc.output_dim,
            self.trans_enc.output_dim
        ]
        self.num_used = sum(d_>0 for d_ in out_dims)
        self.d_attn = d_attn

        if self.num_used==0:
            # dummy
            self.unify = nn.Identity()
            self.norm  = nn.Identity()
        else:
            final_dim = self.num_used * d_attn
            emb_dim = config["final_emb_dim"]
            self.unify = nn.Sequential(
                nn.Linear(final_dim, emb_dim*2),
                nn.ReLU(),
                nn.Dropout(dr),
                nn.Linear(emb_dim*2, emb_dim)
            )
            self.norm = nn.LayerNorm(emb_dim)

    def forward(self, batch):
        cb = self.cnv_broad_enc(batch["cnv_broad"])
        cf = self.cnv_focal_enc(batch["cnv_focal"])
        ex = self.expr_enc(batch["expression"])
        sn = self.snv_enc(batch["snv"])
        fu = self.fusion_enc(batch["fusion_gene"])
        tr = self.trans_enc(batch["translocations"])

        mod_list = []
        for m_ in [cb, cf, ex, sn, fu, tr]:
            if m_.shape[1] > 0:
                mod_list.append(m_)

        if len(mod_list)==0:
            # all zero
            batch_size = cb.shape[0]
            return cb.new_zeros((batch_size, 1))

        stack = torch.stack(mod_list, dim=0)  # => [num_used, B, d_attn]
        # cross attn
        out, _ = self.cross_attn(stack, stack, stack)
        out = out.transpose(0,1).reshape(out.size(1), -1)
        out = self.unify(out)
        out = self.norm(out)
        return out

###############################################################################
# 3) TRAINING / EVAL
###############################################################################
def contrastive_loss(emb, temperature=0.1):
    emb = F.normalize(emb, dim=-1)
    sim = torch.matmul(emb, emb.T)
    sim = sim / temperature
    labels = torch.arange(sim.size(0), device=sim.device)
    return F.cross_entropy(sim, labels)

def run_epoch(model, loader, optimizer, device, train=True):
    if train:
        model.train()
    else:
        model.eval()
    total_loss=0
    with torch.set_grad_enabled(train):
        for batch in loader:
            for k in batch:
                if k!="patient_id":
                    batch[k] = batch[k].to(device)
            emb = model(batch)
            loss = contrastive_loss(emb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss+=loss.item()
    return total_loss/len(loader)

def get_embeddings(model, dataset, device, batch_size=16):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    all_ids=[]
    all_embs=[]
    with torch.no_grad():
        for batch in loader:
            pids = batch["patient_id"]
            for k in batch:
                if k!="patient_id":
                    batch[k] = batch[k].to(device)
            emb = model(batch).cpu().numpy()
            all_embs.append(emb)
            if isinstance(pids[0], str):
                all_ids.extend(pids)
            else:
                all_ids.extend(pids.numpy().tolist())
    all_embs = np.concatenate(all_embs, axis=0)
    return np.array(all_ids), all_embs

def cluster_and_evaluate(ids, embeddings, out_prefix="myeloma", n_clusters=5):
    """
    T-SNE + KMeans + silhouette, no ARI since user said no labels.
    Also saves 'out_prefix_clusters.csv' plus 'out_prefix_tsne.png'
    """
    if embeddings.shape[0]<3:
        print("Not enough data for t-SNE or clustering.")
        return {"silhouette": None}

    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    emb_2d = tsne.fit_transform(embeddings)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)

    sil = None
    if embeddings.shape[0]>=n_clusters:
        sil = silhouette_score(embeddings, cluster_labels)

    df_cl = pd.DataFrame({"patient_id":ids, "cluster":cluster_labels})
    out_csv = f"{out_prefix}_clusters.csv"
    df_cl.to_csv(out_csv, index=False)
    print(f"Saved cluster assignment => {out_csv}")

    # also save embeddings for external analysis
    emb_df = pd.DataFrame(embeddings)
    emb_df["patient_id"] = ids
    emb_df.to_csv(f"{out_prefix}_embeddings.csv", index=False)
    print(f"Saved final embeddings => {out_prefix}_embeddings.csv")

    # plot
    plt.figure(figsize=(10,7))
    sc = plt.scatter(emb_2d[:,0], emb_2d[:,1], c=cluster_labels, cmap='tab10', alpha=0.6)
    for i, pid in enumerate(ids):
        plt.text(emb_2d[i,0], emb_2d[i,1], str(pid), fontsize=6, alpha=0.7)
    plt.colorbar(sc, label="Cluster")
    plt.title("Myeloma t-SNE improved approach")
    out_png = f"{out_prefix}_tsne.png"
    plt.savefig(out_png, dpi=150)
    plt.show()

    return {"silhouette": sil}

###############################################################################
# 4) MAIN
###############################################################################
def run_single_experiment(dataset, param_set, run_name=""):
    # train/val
    n_total = len(dataset)
    n_train = int(0.8*n_total)
    n_val = n_total-n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=param_set["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=param_set["batch_size"], shuffle=False)

    # build config
    config = {
        "cnv_broad_dim": param_set["modality_dims"]["cnv_broad_dim"],
        "cnv_focal_dim": param_set["modality_dims"]["cnv_focal_dim"],
        "expr_dim":      param_set["modality_dims"]["expression_dim"],
        "snv_dim":       param_set["modality_dims"]["snv_dim"],
        "fusion_dim":    param_set["modality_dims"]["fusion_gene_dim"],
        "trans_dim":     param_set["modality_dims"]["translocations_dim"],
        "attn_dim":   param_set["attn_dim"],
        "num_heads":  param_set["num_heads"],
        "dropout":    0.2,
        "final_emb_dim": 128
    }
    model = MyelomaImprovedModel(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=param_set["lr"])

    for epoch in range(param_set["epochs"]):
        tr_loss = run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss= run_epoch(model, val_loader, optimizer, device, train=False)
        print(f"[{run_name}] Epoch [{epoch+1}/{param_set['epochs']}] train={tr_loss:.4f} val={val_loss:.4f}")

    # full
    full_loader = DataLoader(dataset, batch_size=param_set["batch_size"], shuffle=False)
    ids, embs = get_embeddings(model, dataset, device, param_set["batch_size"])
    out_metrics = cluster_and_evaluate(ids, embs, out_prefix=run_name, n_clusters=param_set["n_clusters"])
    return out_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--attn_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--hyperparam_search", type=int, default=0)
    parser.add_argument("--modality_subset", type=str, default="all",
                        help="Comma-separated or 'all'. e.g. 'expression,snv'")
    parser.add_argument("--top_genes", type=int, default=2000,
                        help="top N variable genes in expression. If 0 => all.")
    args=parser.parse_args()

    # 1) load dataset
    dataset = MyelomaDataset(data_dir="data/", top_genes=args.top_genes)

    # figure out dims
    sample = dataset[0]
    full_dims = {
        "cnv_broad_dim": sample["cnv_broad"].shape[0],
        "cnv_focal_dim": sample["cnv_focal"].shape[0],
        "expression_dim": sample["expression"].shape[0],
        "snv_dim": sample["snv"].shape[0],
        "fusion_gene_dim": sample["fusion_gene"].shape[0],
        "translocations_dim": sample["translocations"].shape[0]
    }

    # subset
    if args.modality_subset.lower()=="all":
        used_dims = full_dims
    else:
        used_set = [m.strip().lower() for m in args.modality_subset.split(",")]
        used_dims={}
        for k,v in full_dims.items():
            # e.g. k="cnv_broad_dim" => short = "cnv_broad"
            shortk = k.replace("_dim","")
            if shortk in used_set:
                used_dims[k]=v
            else:
                used_dims[k]=0

    if args.hyperparam_search==1:
        param_grid = [
            {
              "modality_dims": used_dims,
              "attn_dim": args.attn_dim,
              "num_heads": args.num_heads,
              "lr": args.lr,
              "batch_size": args.batch_size,
              "epochs": args.epochs,
              "n_clusters": 5
            },
            {
              "modality_dims": used_dims,
              "attn_dim": args.attn_dim*2,  # bigger dimension
              "num_heads": args.num_heads,
              "lr": args.lr,
              "batch_size": args.batch_size,
              "epochs": args.epochs,
              "n_clusters": 5
            }
        ]
        results=[]
        for i, pset in enumerate(param_grid):
            run_name=f"grid{i+1}"
            print(f"\n--- Running {run_name} => {pset}")
            out_metrics = run_single_experiment(dataset, pset, run_name=run_name)
            results.append((run_name, pset, out_metrics))
        print("\n=== Grid Search Results ===")
        for (rn, pset, om) in results:
            print(f"{rn} => silhouette={om['silhouette']}")

    else:
        param_set = {
            "modality_dims": used_dims,
            "attn_dim": args.attn_dim,
            "num_heads": args.num_heads,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "n_clusters": 5
        }
        print(f"\n--- Single run => {param_set} ---")
        out_metrics = run_single_experiment(dataset, param_set, run_name="single_run")
        print(f"\nFinal => silhouette={out_metrics['silhouette']}")

if __name__=="__main__":
    main()
