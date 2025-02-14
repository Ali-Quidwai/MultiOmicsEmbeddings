#!/usr/bin/env python3

"""
myeloma_modal_subset_experiment.py

A single script that:
1) Loads multi-modal Myeloma data from CSVs in `data/`.
2) Lets you specify which modalities to actually use via --modality_subset:
   'cnv_broad', 'cnv_focal', 'expression', 'snv', 'fusion_gene', 'translocations', or 'all'.
   You can combine multiple by comma, e.g. "expression,fusion_gene".
3) Trains a cross-modal attention model that sets dimension=0 for excluded modalities.
4) Runs a train/val loop with a contrastive-like loss, does t-SNE + KMeans on final embeddings.
5) Optionally does a mini hyperparam search if --hyperparam_search=1.

Usage Examples:

    # Only expression
    python myeloma_modal_subset_experiment.py --modality_subset expression \
        --epochs 5 --batch_size 16

    # Expression + SNV
    python myeloma_modal_subset_experiment.py --modality_subset expression,snv

    # All
    python myeloma_modal_subset_experiment.py --modality_subset all

    # Hyperparam search
    python myeloma_modal_subset_experiment.py --modality_subset all \
        --hyperparam_search 1

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

from sklearn.metrics import silhouette_score, adjusted_rand_score
import matplotlib.pyplot as plt


###############################################################################
# 1) DATASET
###############################################################################
class MyelomaDataset(Dataset):
    """
    Loads all six data modalities from CSV. We'll unify naming so each
    dimension is stored as:
      - cnv_broad_dim
      - cnv_focal_dim
      - expression_dim
      - snv_dim
      - fusion_gene_dim
      - translocations_dim
    The user will specify which of these they want to keep.
    """
    def __init__(
        self,
        data_dir="data/",
        cnv_broad_file="COMMPASS_184_cnvBroad.csv",
        cnv_focal_file="COMMPASS_184_cnvFocal.csv",
        expression_file="COMMPASS_184_gene_expression_scaled.csv",
        snv_file="COMMPASS_184_snv.csv",
        fusion_gene_file="COMMPASS_184_fusion_gene.csv",
        translocations_file="COMMPASS_184_translocations.csv"
    ):
        super().__init__()

        # --- 1. Load data ---
        self.cnv_broad_df = pd.read_csv(os.path.join(data_dir, cnv_broad_file))
        self.cnv_focal_df = pd.read_csv(os.path.join(data_dir, cnv_focal_file))
        self.expression_df = pd.read_csv(os.path.join(data_dir, expression_file))
        self.snv_df        = pd.read_csv(os.path.join(data_dir, snv_file))
        self.fusion_df     = pd.read_csv(os.path.join(data_dir, fusion_gene_file))
        self.transloc_df   = pd.read_csv(os.path.join(data_dir, translocations_file))

        # Identify sample IDs
        if "Unnamed: 0" in self.cnv_broad_df.columns:
            self.patient_ids = self.cnv_broad_df["Unnamed: 0"].values
        else:
            self.patient_ids = np.arange(len(self.cnv_broad_df))

        # Drop ID columns if present
        def drop_id(df):
            return df.drop(columns=["Unnamed: 0"], errors="ignore") if "Unnamed: 0" in df.columns else df

        self.cnv_broad_df = drop_id(self.cnv_broad_df)
        self.cnv_focal_df = drop_id(self.cnv_focal_df)
        self.expression_df= drop_id(self.expression_df)
        self.snv_df       = drop_id(self.snv_df)
        self.fusion_df    = drop_id(self.fusion_df)
        self.transloc_df  = drop_id(self.transloc_df)

        # Scale continuous data
        self.cnv_broad_arr = self._scale(self.cnv_broad_df)
        self.cnv_focal_arr = self._scale(self.cnv_focal_df)
        self.expr_arr       = self._scale(self.expression_df)

        # For binary data => float32
        self.snv_arr     = self.snv_df.astype(np.float32).values
        self.fusion_arr  = self.fusion_df.astype(np.float32).values
        self.trans_arr   = self.transloc_df.astype(np.float32).values

        # Consistency check
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
            "patient_id":      self.patient_ids[idx],
            "cnv_broad":       torch.from_numpy(self.cnv_broad_arr[idx]),
            "cnv_focal":       torch.from_numpy(self.cnv_focal_arr[idx]),
            "expression":      torch.from_numpy(self.expr_arr[idx]),
            "snv":             torch.from_numpy(self.snv_arr[idx]),
            "fusion_gene":     torch.from_numpy(self.fusion_arr[idx]),
            "translocations":  torch.from_numpy(self.trans_arr[idx]),
        }


###############################################################################
# 2) MODEL with "skip" if dimension=0
###############################################################################
class ModalityAttentionEncoder(nn.Module):
    def __init__(self, input_dim, projection_dim=256, num_heads=4, dropout=0.2):
        super().__init__()
        self._input_dim = input_dim
        if input_dim == 0:
            # We'll store dummy layers
            self.output_dim = 0
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

    def forward(self, x):
        # If dimension=0 => return zero
        if self._input_dim == 0:
            batch_size = x.shape[0]
            return x.new_zeros((batch_size, 0))
        else:
            out = self.initial_proj(x)   # => [B, proj_dim]
            out = out.unsqueeze(0)       # => [1, B, proj_dim]
            attn_out, _ = self.attention(out, out, out)
            out = out + self.dropout(attn_out)
            out = self.layer_norm(out)
            out = out.squeeze(0)         # => [B, proj_dim]
            return out

class CrossModalAttention(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.2):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self._dim = dim

    def forward(self, x):
        # x shape => [num_modalities, B, dim], but might be empty if num_modalities=0 or dim=0
        if x.shape[0] == 0 or x.shape[2] == 0:
            # no modalities => return empty
            # let's return shape [0, B, 0]
            batch_size = x.shape[1] if x.shape[1]>0 else 1
            return x.new_zeros((0, batch_size, 0))

        attn_out, _ = self.attn(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.norm(x)
        return x

class MyelomaMultiModalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_attn = config["attn_dim"]
        n_h = config["num_heads"]
        dr  = config["dropout"]

        # Build each encoder. If dimension=0 => it yields empty vectors
        self.enc_cnv_broad = ModalityAttentionEncoder(config["cnv_broad_dim"], d_attn, n_h, dr)
        self.enc_cnv_focal = ModalityAttentionEncoder(config["cnv_focal_dim"], d_attn, n_h, dr)
        self.enc_expr       = ModalityAttentionEncoder(config["expression_dim"], d_attn, n_h, dr)
        self.enc_snv        = ModalityAttentionEncoder(config["snv_dim"], d_attn, n_h, dr)
        self.enc_fusion     = ModalityAttentionEncoder(config["fusion_gene_dim"], d_attn, n_h, dr)
        self.enc_trans      = ModalityAttentionEncoder(config["translocations_dim"], d_attn, n_h, dr)

        self.cross_attn = CrossModalAttention(dim=d_attn, num_heads=n_h, dropout=dr)

        # figure out how many have >0 dimension
        out_dims = [
            self.enc_cnv_broad.output_dim,
            self.enc_cnv_focal.output_dim,
            self.enc_expr.output_dim,
            self.enc_snv.output_dim,
            self.enc_fusion.output_dim,
            self.enc_trans.output_dim
        ]
        self.num_modalities_used = sum([1 for x in out_dims if x>0])

        total_dim = self.num_modalities_used * d_attn
        emb_dim   = config["final_emb_dim"]
        if total_dim>0:
            self.unify = nn.Sequential(
                nn.Linear(total_dim, emb_dim*2),
                nn.ReLU(),
                nn.Dropout(dr),
                nn.Linear(emb_dim*2, emb_dim)
            )
            self.norm = nn.LayerNorm(emb_dim)
        else:
            # if all are zero => we can't do anything
            # We could do Identity() but that yields no grad => runtime error
            self.unify = None
            self.norm  = None

    def forward(self, batch):
        # encode each
        cb = self.enc_cnv_broad(batch["cnv_broad"].float())
        cf = self.enc_cnv_focal(batch["cnv_focal"].float())
        ex = self.enc_expr(batch["expression"].float())
        sn = self.enc_snv(batch["snv"].float())
        fu = self.enc_fusion(batch["fusion_gene"].float())
        tr = self.enc_trans(batch["translocations"].float())

        # collect only non-empty
        mod_list = []
        for v_ in [cb, cf, ex, sn, fu, tr]:
            if v_.shape[1]>0:  # means dimension>0
                mod_list.append(v_)

        if len(mod_list)==0:
            # all zero => no training possible
            raise ValueError(
                "All selected modalities have dimension=0. "
                "No meaningful embeddings can be produced. "
                "Please check --modality_subset or your data shape."
            )

        stack = torch.stack(mod_list, dim=0)  # => [num_used, B, d_attn]
        cross_out = self.cross_attn(stack)    # => [num_used, B, d_attn]

        if cross_out.shape[0]==0 or cross_out.shape[2]==0:
            # This can happen if we messed up. We'll raise an error
            raise ValueError("CrossModalAttention produced an empty tensor. Check your dims/config.")

        # flatten => [B, num_used * d_attn]
        out = cross_out.transpose(0,1).reshape(cross_out.size(1), -1)

        if self.unify is None:
            raise ValueError(
                "All modalities are zero dimension. This can't produce grad. "
                "Double check your subset or data shapes."
            )

        out = self.unify(out)
        out = self.norm(out)
        return out


###############################################################################
# 3) TRAINING & EVAL
###############################################################################
def contrastive_loss(emb, temperature=0.1):
    emb = F.normalize(emb, dim=-1)
    sim = torch.matmul(emb, emb.t())
    sim = sim / temperature
    labels = torch.arange(sim.size(0), device=sim.device)
    return F.cross_entropy(sim, labels)

def run_epoch(model, loader, optimizer, device, train=True):
    if train:
        model.train()
    else:
        model.eval()
    total_loss=0.0
    with torch.set_grad_enabled(train):
        for batch in loader:
            for k in batch:
                if k!="patient_id":
                    batch[k] = batch[k].to(device)

            emb = model(batch)
            loss= contrastive_loss(emb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss+= loss.item()
    return total_loss/len(loader)

def get_embeddings(model, dataset, device, batch_size=16):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    all_ids = []
    all_embs= []
    with torch.no_grad():
        for batch in loader:
            pids = batch["patient_id"]
            for k in batch:
                if k!="patient_id":
                    batch[k] = batch[k].to(device)
            emb = model(batch).cpu().numpy()
            all_embs.append(emb)
            # gather pids
            if isinstance(pids[0], str):
                all_ids.extend(pids)
            else:
                all_ids.extend(pids.numpy().tolist())
    all_embs = np.concatenate(all_embs, axis=0)
    return np.array(all_ids), all_embs

def cluster_and_evaluate(ids, embeddings, out_prefix="results_run", n_clusters=5, label_dict=None):
    if embeddings.shape[0]<3:
        print("Not enough data for t-SNE.")
        return {"ari":None, "silhouette":None}

    # TSNE
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    emb_2d = tsne.fit_transform(embeddings)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    if embeddings.shape[0]>=n_clusters:
        sil = silhouette_score(embeddings, cluster_labels)
    else:
        sil = None

    df_cl = pd.DataFrame({"patient_id":ids, "cluster":cluster_labels})
    out_csv = f"{out_prefix}_clusters.csv"
    df_cl.to_csv(out_csv, index=False)
    print(f"Saved cluster assignment => {out_csv}")

    ari=None
    if label_dict is not None:
        true_labels = []
        for pid in ids:
            if pid in label_dict:
                true_labels.append(label_dict[pid])
            else:
                true_labels.append(-1)
        true_labels = np.array(true_labels)
        mask = (true_labels!=-1)
        if mask.sum()>1 and len(np.unique(true_labels[mask]))>1:
            ari = adjusted_rand_score(true_labels[mask], cluster_labels[mask])
            print(f"Adjusted Rand Index: {ari:.4f}")

    # Plot
    plt.figure(figsize=(10,7))
    sc=plt.scatter(emb_2d[:,0], emb_2d[:,1], c=cluster_labels, cmap='tab10', alpha=0.6)
    for i,pid in enumerate(ids):
        plt.text(emb_2d[i,0], emb_2d[i,1], str(pid), fontsize=6, alpha=0.7)
    plt.colorbar(sc, label="Cluster")
    plt.title("t-SNE Myeloma Embeddings")
    out_png = f"{out_prefix}_tsne.png"
    plt.savefig(out_png, dpi=150)
    plt.show()

    return {"ari":ari, "silhouette":sil}

###############################################################################
# 4) SINGLE EXPERIMENT OR MINI SEARCH
###############################################################################
def run_single_experiment(dataset, param_set, run_name="", label_dict=None):
    n_total = len(dataset)
    n_train= int(0.8*n_total)
    n_val  = n_total - n_train
    train_ds, val_ds = random_split(dataset, [n_train,n_val], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=param_set["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=param_set["batch_size"], shuffle=False)

    # Build model config
    config = {
      "cnv_broad_dim": param_set["modality_dims"]["cnv_broad_dim"],
      "cnv_focal_dim": param_set["modality_dims"]["cnv_focal_dim"],
      "expression_dim": param_set["modality_dims"]["expression_dim"],
      "snv_dim": param_set["modality_dims"]["snv_dim"],
      "fusion_gene_dim": param_set["modality_dims"]["fusion_gene_dim"],
      "translocations_dim": param_set["modality_dims"]["translocations_dim"],
      "num_heads": param_set["num_heads"],
      "attn_dim":  param_set["attn_dim"],
      "dropout":   param_set["dropout"],
      "final_emb_dim": 128
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyelomaMultiModalModel(config).to(device)
    optimizer= torch.optim.Adam(model.parameters(), lr=param_set["lr"])

    for epoch in range(param_set["epochs"]):
        train_loss = run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss   = run_epoch(model, val_loader,   optimizer, device, train=False)
        print(f"[{run_name}] Epoch [{epoch+1}/{param_set['epochs']}] | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

    # embeddings => cluster
    ids, embs = get_embeddings(model, dataset, device, batch_size=param_set["batch_size"])
    out_metrics = cluster_and_evaluate(ids, embs,
        out_prefix=f"{run_name}",
        n_clusters=param_set["n_clusters"],
        label_dict=label_dict
    )
    return out_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--attn_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--hyperparam_search", type=int, default=0)
    parser.add_argument("--label_file", type=str, default=None,
                        help="CSV with columns [patient_id, true_label] for ARI")
    parser.add_argument("--modality_subset", type=str, default="all",
                        help="Comma list of modalities: cnv_broad,cnv_focal,expression,snv,fusion_gene,translocations OR 'all'")
    args = parser.parse_args()

    # 1) Load dataset
    dataset = MyelomaDataset(data_dir="data/")

    # 2) Possibly load labels
    label_dict = None
    if args.label_file and os.path.exists(args.label_file):
        df_lab = pd.read_csv(args.label_file)
        label_dict={}
        for _,row in df_lab.iterrows():
            label_dict[row["patient_id"]] = row["true_label"]
        print(f"Loaded {len(label_dict)} labels from {args.label_file}")

    # 3) Figure out subset
    # e.g. "cnv_broad,expression" or "all"
    # We'll unify the dimension dictionary
    sample = dataset[0]
    full_mod_dims = {
      "cnv_broad_dim": sample["cnv_broad"].shape[0],
      "cnv_focal_dim": sample["cnv_focal"].shape[0],
      "expression_dim": sample["expression"].shape[0],
      "snv_dim":       sample["snv"].shape[0],
      "fusion_gene_dim": sample["fusion_gene"].shape[0],
      "translocations_dim": sample["translocations"].shape[0],
    }

    if args.modality_subset.lower()=="all":
        used_mod_dims = dict(full_mod_dims)
    else:
        # parse comma list
        subset_list = [m.strip() for m in args.modality_subset.split(",")]
        # set dimension=0 if not in subset
        used_mod_dims={}
        # We'll map from user's "cnv_broad" => we keep "cnv_broad_dim", etc.
        # Check each key
        for key in full_mod_dims:
            # key e.g. "cnv_broad_dim" => base = "cnv_broad"
            base = key.replace("_dim","")
            if base in subset_list:
                used_mod_dims[key] = full_mod_dims[key]
            else:
                used_mod_dims[key] = 0

    # If all zero => raise error early
    sum_dims = sum(used_mod_dims.values())
    if sum_dims==0:
        raise ValueError(
          f"Your chosen subset '{args.modality_subset}' leads to all dims=0. "
          "No training can be done. Please pick a valid subset."
        )

    # 4) Run single or mini search
    if args.hyperparam_search==1:
        # Example param grid
        param_grid = [
          {
            "attn_dim": args.attn_dim,
            "num_heads": args.num_heads,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "dropout": 0.2,
            "final_emb_dim":128,
            "n_clusters":5,
            "modality_dims": used_mod_dims
          },
          {
            "attn_dim": args.attn_dim*2,
            "num_heads": args.num_heads,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "dropout": 0.2,
            "final_emb_dim":128,
            "n_clusters":5,
            "modality_dims": used_mod_dims
          }
        ]
        results=[]
        for i, pset in enumerate(param_grid):
            run_name = f"grid{i+1}"
            print(f"\n--- Running {run_name} with param set => {pset} ---")
            metrics= run_single_experiment(dataset, pset, run_name=run_name, label_dict=label_dict)
            results.append((run_name, pset, metrics))

        print("\n==== Grid Search Results ====")
        for run_name, pset, outm in results:
            print(f"{run_name} -> ARI={outm['ari']}, silhouette={outm['silhouette']}")
    else:
        param_set = {
          "attn_dim": args.attn_dim,
          "num_heads": args.num_heads,
          "lr": args.lr,
          "batch_size": args.batch_size,
          "epochs": args.epochs,
          "dropout":0.2,
          "final_emb_dim":128,
          "n_clusters":5,
          "modality_dims": used_mod_dims
        }
        print(f"\n--- Single run with param_set => {param_set} ---")
        metrics = run_single_experiment(dataset, param_set, run_name="single_run", label_dict=label_dict)
        print(f"\nFinal => ARI={metrics['ari']}, silhouette={metrics['silhouette']}")


if __name__=="__main__":
    main()
