#!/usr/bin/env python3

"""
myeloma_full_experiment.py

An "all-in-one" script that:

1) Loads multi-modal Myeloma data from CSVs in `data/`.
2) Optionally runs multiple hyperparameter configurations (a mini grid search):
   - attn_dim
   - num_heads
   - lr
   - etc.
3) For each run:
   - Train a cross-modal attention model with a contrastive-like loss
   - Log train/val loss each epoch
   - After training, do t-SNE + KMeans => cluster embeddings
   - Save cluster assignments in CSV
   - If the user provides a label CSV (with true classes for each patient), compute ARI
4) Summarize results across runs.

Usage Examples:
  - Single run:
    python myeloma_full_experiment.py --epochs 5 --batch_size 16 --lr 1e-4 --attn_dim 128 --num_heads 4

  - Small hyperparam search with multiple combos:
    python myeloma_full_experiment.py --epochs 5 --batch_size 16 --hyperparam_search 1

  - Provide a label CSV to compute ARI:
    python myeloma_full_experiment.py --label_file data/my_patient_labels.csv
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

# Optional for ARI or other metrics if we have ground-truth labels
from sklearn.metrics import adjusted_rand_score, silhouette_score

import matplotlib.pyplot as plt


###############################################################################
# 1) DATASET
###############################################################################
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
        Loads CSVs from data_dir, drops 'Unnamed: 0', scales continuous data,
        and casts binary data to float32.
        Stores Tensors for each modality plus patient_ids.
        """
        # --- Load data ---
        self.cnv_broad_df = pd.read_csv(os.path.join(data_dir, cnv_broad_file))
        self.cnv_focal_df = pd.read_csv(os.path.join(data_dir, cnv_focal_file))
        self.expr_df      = pd.read_csv(os.path.join(data_dir, expr_file))
        self.snv_df       = pd.read_csv(os.path.join(data_dir, snv_file))
        self.fusion_df    = pd.read_csv(os.path.join(data_dir, fusion_file))
        self.trans_df     = pd.read_csv(os.path.join(data_dir, trans_file))

        # --- Identify sample IDs ---
        if "Unnamed: 0" in self.cnv_broad_df.columns:
            self.patient_ids = self.cnv_broad_df["Unnamed: 0"].values
        else:
            self.patient_ids = np.arange(len(self.cnv_broad_df))

        # --- Drop ID columns if present ---
        def drop_id(df):
            return df.drop(columns=["Unnamed: 0"], errors="ignore") if "Unnamed: 0" in df.columns else df
        self.cnv_broad_df = drop_id(self.cnv_broad_df)
        self.cnv_focal_df = drop_id(self.cnv_focal_df)
        self.expr_df      = drop_id(self.expr_df)
        self.snv_df       = drop_id(self.snv_df)
        self.fusion_df    = drop_id(self.fusion_df)
        self.trans_df     = drop_id(self.trans_df)

        # --- Scale continuous data (CNV broad/focal, expression) ---
        self.cnv_broad_arr = self._scale(self.cnv_broad_df)
        self.cnv_focal_arr = self._scale(self.cnv_focal_df)
        self.expr_arr      = self._scale(self.expr_df)

        # --- Convert binary data to float32 (snv, fusion, translocations) ---
        self.snv_arr    = self.snv_df.astype(np.float32).values
        self.fusion_arr = self.fusion_df.astype(np.float32).values
        self.trans_arr  = self.trans_df.astype(np.float32).values

        # Check shapes match the # of samples
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


###############################################################################
# 2) MODEL WITH CROSS-MODAL ATTENTION
###############################################################################

class ModalityAttentionEncoder(nn.Module):
    """
    Example approach for each modality: MLP -> multihead attention -> single vector
    For large #features (e.g. 16k genes), you'd ideally do chunking or dimension reduction first.
    """
    def __init__(self, input_dim, projection_dim=256, num_heads=4, dropout=0.2):
        super().__init__()
        # MLP to reduce dimension from input_dim -> projection_dim
        self.initial_proj = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # Self-attention
        self.attention = nn.MultiheadAttention(embed_dim=projection_dim,
                                               num_heads=num_heads,
                                               dropout=dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = projection_dim

    def forward(self, x):
        # x: [batch_size, input_dim]
        out = self.initial_proj(x)           # => [B, proj_dim]
        # We'll do a toy approach to apply multi-head across batch dimension
        out = out.unsqueeze(0)               # => [1, B, proj_dim]
        attn_out, _ = self.attention(out, out, out)
        out = out + self.dropout(attn_out)
        out = self.layer_norm(out)
        out = out.squeeze(0)                 # => [B, proj_dim]
        return out

class CrossModalAttention(nn.Module):
    """
    Cross-modal attention on top of the 6 embeddings:
      shape => [6, batch, dim]
    """
    def __init__(self, dim, num_heads=4, dropout=0.2):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout)
        self.layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [num_modalities, batch_size, dim]
        """
        attn_out, _ = self.attn(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.layer_norm(x)
        return x

class MyelomaMultiModalModel(nn.Module):
    """
    Encodes each of the 6 data modalities with a ModalityAttentionEncoder,
    then applies a cross-modal attention, then a final unify layer -> embedding
    """
    def __init__(self, config):
        super().__init__()
        d = config["attn_dim"]
        n_h = config["num_heads"]
        dr = config["dropout"]

        self.cnv_broad_enc = ModalityAttentionEncoder(config["cnv_broad_dim"], d, n_h, dr)
        self.cnv_focal_enc = ModalityAttentionEncoder(config["cnv_focal_dim"], d, n_h, dr)
        self.expr_enc      = ModalityAttentionEncoder(config["expr_dim"],       d, n_h, dr)
        self.snv_enc       = ModalityAttentionEncoder(config["snv_dim"],        d, n_h, dr)
        self.fusion_enc    = ModalityAttentionEncoder(config["fusion_dim"],     d, n_h, dr)
        self.trans_enc     = ModalityAttentionEncoder(config["trans_dim"],      d, n_h, dr)

        self.cross_attn = CrossModalAttention(dim=d, num_heads=n_h, dropout=dr)

        total_dim = config["num_modalities"] * d
        emb_dim = config["final_emb_dim"]
        self.unify = nn.Sequential(
            nn.Linear(total_dim, emb_dim * 2),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Linear(emb_dim * 2, emb_dim)
        )
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, batch):
        cb = self.cnv_broad_enc(batch["cnv_broad"].float())
        cf = self.cnv_focal_enc(batch["cnv_focal"].float())
        ex = self.expr_enc(batch["expression"].float())
        sn = self.snv_enc(batch["snv"].float())
        fu = self.fusion_enc(batch["fusion_gene"].float())
        tr = self.trans_enc(batch["translocations"].float())

        # stack => [6, B, d]
        modalities = torch.stack([cb, cf, ex, sn, fu, tr], dim=0)
        cross_out = self.cross_attn(modalities)  # => same shape [6, B, d]
        out = cross_out.transpose(0,1).reshape(cross_out.size(1), -1)
        out = self.unify(out)
        out = self.norm(out)
        return out


###############################################################################
# 3) TRAINING & EVALUATION
###############################################################################
def contrastive_loss(embeddings, temperature=0.1):
    emb = F.normalize(embeddings, dim=-1)
    sim = torch.matmul(emb, emb.T)  # [batch, batch]
    sim = sim / temperature
    labels = torch.arange(sim.size(0), device=sim.device)
    return F.cross_entropy(sim, labels)

def run_epoch(model, dataloader, optimizer, device, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    with torch.set_grad_enabled(train):
        for batch in dataloader:
            for k in batch:
                if k != "patient_id":
                    batch[k] = batch[k].to(device)

            emb = model(batch)
            loss = contrastive_loss(emb)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

    return total_loss / len(dataloader)

def get_embeddings(model, dataloader, device):
    """
    Return (patient_ids, embeddings) for the entire loader
    """
    model.eval()
    all_ids = []
    all_embs = []
    with torch.no_grad():
        for batch in dataloader:
            pids = batch["patient_id"]
            for k in batch:
                if k != "patient_id":
                    batch[k] = batch[k].to(device)
            emb = model(batch).cpu()
            all_embs.append(emb)
            # handle pids
            if isinstance(pids[0], str):
                all_ids.extend(pids)
            else:
                all_ids.extend(pids.numpy().tolist())

    all_embs = torch.cat(all_embs, dim=0).numpy()
    return np.array(all_ids), all_embs


def run_tsne_kmeans(ids, embeddings, out_prefix="myeloma", n_clusters=5, label_dict=None):
    """
    1) t-SNE
    2) KMeans
    3) Save cluster assignments to CSV
    4) If label_dict is given (patient_id -> label), compute ARI
    5) Optionally compute silhouette score (just for reference if you want an unsupervised metric)
    """
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
    import pandas as pd

    if embeddings.shape[0] < 3:
        print("Not enough data for t-SNE.")
        return None

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    emb_2d = tsne.fit_transform(embeddings)

    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Save cluster assignments
    df_clusters = pd.DataFrame({
        "patient_id": ids,
        "cluster": cluster_labels
    })

    # If label_dict is given, compute ARI
    ari = None
    if label_dict is not None:
        # Map each patient to label
        true_labels = []
        for pid in ids:
            if pid in label_dict:
                true_labels.append(label_dict[pid])
            else:
                true_labels.append(-1)  # or some missing label
        true_labels = np.array(true_labels)
        # Only evaluate ARI on those who have a valid label
        mask = (true_labels != -1)
        if mask.sum() > 1 and len(np.unique(true_labels[mask])) > 1:
            ari = adjusted_rand_score(true_labels[mask], cluster_labels[mask])
        else:
            print("Not enough labeled data to compute ARI.")
            ari = None

    # Possibly silhouette
    if embeddings.shape[0] > n_clusters:
        silhouette = silhouette_score(embeddings, cluster_labels)
    else:
        silhouette = None

    # Save to CSV
    out_csv = f"{out_prefix}_clusters.csv"
    df_clusters.to_csv(out_csv, index=False)
    print(f"Cluster assignments saved to {out_csv}")
    if ari is not None:
        print(f"Adjusted Rand Index (vs. provided labels): {ari:.4f}")
    if silhouette is not None:
        print(f"Silhouette Score: {silhouette:.4f}")

    # Plot
    plt.figure(figsize=(10,7))
    sc = plt.scatter(emb_2d[:,0], emb_2d[:,1], c=cluster_labels, cmap='tab10', alpha=0.6)
    for i, pid in enumerate(ids):
        plt.text(emb_2d[i,0], emb_2d[i,1], str(pid), fontsize=6, alpha=0.7)
    plt.colorbar(sc, label="Cluster")
    plt.title("t-SNE Visualization of Myeloma Embeddings (Cross-Modal Attention)")
    plt.savefig(f"{out_prefix}_tsne_plot.png", dpi=150)
    plt.show()

    return {"ari": ari, "silhouette": silhouette}


###############################################################################
# 4) MAIN: Optionally do a Grid Search
###############################################################################
def run_single_experiment(args, dataset, param_set, run_name="", label_dict=None):
    """
    Trains for param_set, does train/val loop, logs final ARI if label_dict provided.
    param_set = { "attn_dim": ..., "num_heads": ..., "lr": ..., "epochs": ..., etc. }
    run_name => prefix for saving cluster CSV, e.g. "run1"
    """

    # 1) Train/val split
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=param_set["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=param_set["batch_size"], shuffle=False)

    # 2) Model config
    sample = dataset[0]
    config = {
        "cnv_broad_dim": sample["cnv_broad"].shape[0],
        "cnv_focal_dim": sample["cnv_focal"].shape[0],
        "expr_dim":      sample["expression"].shape[0],
        "snv_dim":       sample["snv"].shape[0],
        "fusion_dim":    sample["fusion_gene"].shape[0],
        "trans_dim":     sample["translocations"].shape[0],
        "num_modalities": 6,
        "attn_dim":   param_set["attn_dim"],
        "num_heads":  param_set["num_heads"],
        "dropout":    param_set["dropout"],
        "final_emb_dim": param_set["final_emb_dim"]
    }

    # 3) Create model
    model = MyelomaMultiModalModel(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 4) Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=param_set["lr"])

    # 5) Train
    for epoch in range(param_set["epochs"]):
        train_loss = run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss   = run_epoch(model, val_loader,   optimizer, device, train=False)
        print(f"[{run_name}] Epoch [{epoch+1}/{param_set['epochs']}] | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # 6) Evaluate => get embeddings on full dataset
    full_loader = DataLoader(dataset, batch_size=param_set["batch_size"], shuffle=False)
    ids, embs = get_embeddings(model, full_loader, device)

    # 7) T-SNE + KMeans => cluster, measure ARI/silhouette if label_dict
    out_metrics = run_tsne_kmeans(
        ids,
        embs,
        out_prefix=f"results_{run_name}",
        n_clusters=param_set["n_clusters"],
        label_dict=label_dict
    )

    return out_metrics


def main(args):
    # 1) Create dataset
    dataset = MyelomaDataset(data_dir="data/")  # adapt if needed

    # 2) Possibly load label file for ARI
    label_dict = None
    if args.label_file is not None and os.path.exists(args.label_file):
        df_labels = pd.read_csv(args.label_file)
        # expect columns: ["patient_id", "true_label"]
        label_dict = {}
        for _, row in df_labels.iterrows():
            label_dict[row["patient_id"]] = row["true_label"]
        print(f"Loaded {len(label_dict)} labels from {args.label_file}")

    # 3) If hyperparam_search = 1 => run multiple combos
    #    else => single run with CLI params
    if args.hyperparam_search == 1:
        # Just an example small grid
        param_grid = [
          {
            "attn_dim": 128,
            "num_heads": 4,
            "lr": 1e-4,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "dropout": 0.2,
            "final_emb_dim": 128,
            "n_clusters": 5
          },
          {
            "attn_dim": 256,
            "num_heads": 4,
            "lr": 1e-4,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "dropout": 0.2,
            "final_emb_dim": 128,
            "n_clusters": 5
          }
          # ... add more combos if you like ...
        ]

        results = []
        for i, pset in enumerate(param_grid):
            run_name = f"grid{i+1}"
            print(f"\n--- Running {run_name} with param set: {pset} ---")
            out_metrics = run_single_experiment(args, dataset, pset, run_name=run_name, label_dict=label_dict)
            results.append((run_name, pset, out_metrics))

        # Print summary
        print("\n==== Grid Search Results ====")
        for run_name, pset, out_metrics in results:
            ari = out_metrics["ari"] if out_metrics else None
            sil = out_metrics["silhouette"] if out_metrics else None
            print(f"{run_name} -> ARI={ari}, silhouette={sil}, config={pset}")

    else:
        # Single run from CLI
        param_set = {
            "attn_dim":   args.attn_dim,
            "num_heads":  args.num_heads,
            "lr":         args.lr,
            "batch_size": args.batch_size,
            "epochs":     args.epochs,
            "dropout":    0.2,
            "final_emb_dim": 128,
            "n_clusters": 5
        }
        # We do a single run
        print(f"\n--- Single run with param set: {param_set} ---")
        run_single_experiment(args, dataset, param_set, run_name="single_run", label_dict=label_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--attn_dim", type=int, default=128,
                        help="Dimension for the ModalityAttentionEncoder output")
    parser.add_argument("--num_heads", type=int, default=4,
                        help="Heads for the multi-head attention")

    parser.add_argument("--hyperparam_search", type=int, default=0,
                        help="Set to 1 to run multiple param combos in a mini grid")

    parser.add_argument("--label_file", type=str, default=None,
                        help="CSV with columns [patient_id, true_label] to compute ARI")

    args = parser.parse_args()

    main(args)
