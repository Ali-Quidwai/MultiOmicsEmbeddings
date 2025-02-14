#!/usr/bin/env python3

"""
myeloma_advanced_experiment.py

A unified script that:
1) Loads multi-omics Myeloma data (CNV, SNV, Expression, etc.)
2) Selects top-N variable genes for expression
3) Uses WeightedEncoder for each modality (including expression)
4) Stacks multiple cross-modal attention layers
5) Trains via SimCLR-like contrastive approach on multi-sample batches
6) Clusters embeddings with UMAP+HDBSCAN & t-SNE+KMeans, logs silhouette
7) Optional post-hoc enrichment if a cytogenetics CSV is provided

Usage Examples:
  python myeloma_advanced_experiment.py --modality_subset all --top_genes 2000 --epochs 5 --batch_size 4
  python myeloma_advanced_experiment.py --modality_subset expression,snv --top_genes 500 --epochs 5
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

# For dimension reduction & clustering
import umap
import hdbscan
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt

###############################################################################
# 1) Dataset with top-N gene selection for expression
###############################################################################

class MyelomaDataset(Dataset):
    """
    Loads CNV broad/focal, expression, SNV, fusion, trans, etc.
    - top_genes for expression
    - If a cytogenetics_file is provided, we store that info in self.cytogenetics for post-hoc checks
    """
    def __init__(
        self,
        data_dir="data/",
        top_genes=2000,
        cnv_broad_file="COMMPASS_184_cnvBroad.csv",
        cnv_focal_file="COMMPASS_184_cnvFocal.csv",
        expr_file="COMMPASS_184_gene_expression_scaled.csv",
        snv_file="COMMPASS_184_snv.csv",
        fusion_file="COMMPASS_184_fusion_gene.csv",
        trans_file="COMMPASS_184_translocations.csv",
        cytogenetics_file=None
    ):
        super().__init__()
        # Load the CSVs
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

        # Optionally load cytogenetics
        self.cytogenetics = {}
        if cytogenetics_file and os.path.exists(cytogenetics_file):
            cyto_df = pd.read_csv(cytogenetics_file)
            for _,row in cyto_df.iterrows():
                pid = row["patient_id"]
                self.cytogenetics[pid] = dict(row)

        # Drop ID columns if present
        def drop_id(df):
            return df.drop(columns=["Unnamed: 0"], errors="ignore") if "Unnamed: 0" in df.columns else df
        self.cnv_broad_df = drop_id(self.cnv_broad_df)
        self.cnv_focal_df = drop_id(self.cnv_focal_df)
        self.expr_df      = drop_id(self.expr_df)
        self.snv_df       = drop_id(self.snv_df)
        self.fusion_df    = drop_id(self.fusion_df)
        self.trans_df     = drop_id(self.trans_df)

        # Scale CNV broad/focal
        self.cnv_broad_arr = self._scale(self.cnv_broad_df)
        self.cnv_focal_arr = self._scale(self.cnv_focal_df)

        # Expression top-gene selection
        if top_genes>0 and top_genes< self.expr_df.shape[1]:
            var_series = self.expr_df.var(axis=0)
            top_idx = var_series.nlargest(top_genes).index
            self.expr_df = self.expr_df[top_idx]
            print(f"Selected top {top_genes} variable genes out of {var_series.shape[0]} total.")
        else:
            print(f"No expression feature selection or top_genes >= #genes => using {self.expr_df.shape[1]} columns.")
        # Scale expression
        self.expr_arr = self._scale(self.expr_df)

        # SNV/fusion/trans => float32
        self.snv_arr = self.snv_df.astype(np.float32).values
        self.fusion_arr = self.fusion_df.astype(np.float32).values
        self.trans_arr = self.trans_df.astype(np.float32).values

        # check shapes
        expected_n = len(self.patient_ids)
        for nm, arr in [
            ("cnv_broad_arr", self.cnv_broad_arr),
            ("cnv_focal_arr", self.cnv_focal_arr),
            ("expr_arr", self.expr_arr),
            ("snv_arr", self.snv_arr),
            ("fusion_arr", self.fusion_arr),
            ("trans_arr", self.trans_arr)
        ]:
            if arr.shape[0] != expected_n:
                raise ValueError(f"{nm} has {arr.shape[0]} rows, expected {expected_n}")

    def _scale(self, df):
        sc = StandardScaler()
        return sc.fit_transform(df).astype(np.float32)

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        sample = {
            "patient_id": self.patient_ids[idx],
            "cnv_broad": torch.from_numpy(self.cnv_broad_arr[idx]),
            "cnv_focal": torch.from_numpy(self.cnv_focal_arr[idx]),
            "expression": torch.from_numpy(self.expr_arr[idx]),
            "snv": torch.from_numpy(self.snv_arr[idx]),
            "fusion": torch.from_numpy(self.fusion_arr[idx]),
            "trans": torch.from_numpy(self.trans_arr[idx])
        }
        pid = sample["patient_id"]
        if pid in self.cytogenetics:
            sample["cyto"] = self.cytogenetics[pid]
        else:
            sample["cyto"] = {}
        return sample


###############################################################################
# 2) SimCLR-like data augmentation for numeric data
###############################################################################

def simclr_augment(batch_item, augment_prob=0.5, scale_jitter=0.1):
    """
    For each numeric field (cnv_broad, cnv_focal, expression, snv, fusion, trans),
    we'll produce two "augmented" versions:
     - with probability augment_prob, scale the vector by (1+random factor)
     - add random Gaussian noise
    This yields (v1, v2) for each sample in the batch.
    """
    # We'll produce two dicts: v1, v2
    v1={}
    v2={}
    # items like "cyto" or "patient_id" we just copy
    for k in batch_item:
        if k in ["patient_id","cyto"]:
            v1[k]= batch_item[k]
            v2[k]= batch_item[k]
        else:
            arr = batch_item[k].float().clone() # shape [dim]
            arr2= arr.clone()
            # random scale
            if torch.rand(1).item()<augment_prob:
                factor = 1.0 + scale_jitter*(2*torch.rand(1).item()-1)
                arr = arr*factor
            # noise
            noise = scale_jitter*torch.randn_like(arr)
            arr_aug1 = arr+noise

            if torch.rand(1).item()<augment_prob:
                factor2= 1.0 + scale_jitter*(2*torch.rand(1).item()-1)
                arr2= arr2*factor2
            noise2= scale_jitter*torch.randn_like(arr2)
            arr_aug2= arr2+noise2

            v1[k]= arr_aug1
            v2[k]= arr_aug2
    return v1, v2

###############################################################################
# 3) Weighted Encoders + multi-layer cross-modal attention
###############################################################################

class WeightedEncoder(nn.Module):
    """
    Weighted MLP for numeric data (including expression).
    input_dim => hidden_dim
    """
    def __init__(self, input_dim, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.input_dim= input_dim
        if input_dim>0:
            self.net=nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            self.net=None
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        if self.input_dim==0 or self.net is None:
            return None
        out= self.net(x)
        out= out*self.weight
        return out

class CrossModalAttnBlock(nn.Module):
    """
    Cross-modal attention across modalities.
    Suppose we have M modalities => we stack => [M, B, hidden_dim].
    We'll do multihead => shape => [M, B, hidden_dim], residual + layernorm.
    """
    def __init__(self, dim=128, num_heads=4, dropout=0.2):
        super().__init__()
        self.attn= nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.norm= nn.LayerNorm(dim)
        self.drop= nn.Dropout(dropout)

    def forward(self, x):
        # x => [M, B, dim]
        attn_out, _= self.attn(x,x,x)
        out= x + self.drop(attn_out)
        out= self.norm(out)
        return out

class MyelomaMultiModalModel(nn.Module):
    """
    For each modality => WeightedEncoder => shape [B, hidden_dim].
    Then we gather => [M, B, hidden_dim] => pass 2 layers cross-modal attn => flatten => final MLP => [B, emb_dim].
    """
    def __init__(self, config):
        super().__init__()
        hd= config["hidden_dim"]
        dr= config["dropout"]
        self.cnv_broad_enc= WeightedEncoder(config["cnv_broad_dim"],hd,dr)
        self.cnv_focal_enc= WeightedEncoder(config["cnv_focal_dim"],hd,dr)
        self.expr_enc     = WeightedEncoder(config["expression_dim"],hd,dr)
        self.snv_enc      = WeightedEncoder(config["snv_dim"],hd,dr)
        self.fusion_enc   = WeightedEncoder(config["fusion_dim"],hd,dr)
        self.trans_enc    = WeightedEncoder(config["trans_dim"],hd,dr)

        self.num_heads= config["num_heads"]
        self.block1= CrossModalAttnBlock(dim=hd, num_heads=self.num_heads, dropout=dr)
        self.block2= CrossModalAttnBlock(dim=hd, num_heads=self.num_heads, dropout=dr)

        emb_dim= config["final_emb_dim"]
        self.out_mlp= nn.Sequential(
            nn.Linear(hd* config["num_modalities_used"], emb_dim),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Linear(emb_dim, emb_dim)
        )
        self.out_norm= nn.LayerNorm(emb_dim)

    def forward(self, batch):
        # batch => dict with "cnv_broad", ...
        # shape => [B, dim]
        cb= self.encode_one(self.cnv_broad_enc, batch["cnv_broad"])
        cf= self.encode_one(self.cnv_focal_enc, batch["cnv_focal"])
        ex= self.encode_one(self.expr_enc,      batch["expression"])
        sn= self.encode_one(self.snv_enc,       batch["snv"])
        fu= self.encode_one(self.fusion_enc,    batch["fusion"])
        tr= self.encode_one(self.trans_enc,     batch["trans"])

        # gather them => list
        mod_list= []
        for m_ in [cb, cf, ex, sn, fu, tr]:
            if m_ is not None:
                mod_list.append(m_)
        # shape => [B, hd] for each => we want => [M, B, hd]
        if len(mod_list)==0:
            # no data => return zero
            bsz= batch["cnv_broad"].size(0)
            return batch["cnv_broad"].new_zeros((bsz,1))

        X= torch.stack(mod_list, dim=0) # => [M, B, hd]
        # cross modal
        X= self.block1(X) # => [M, B, hd]
        X= self.block2(X) # => [M, B, hd]
        # flatten => [B, M*hd]
        X= X.transpose(0,1).reshape(X.size(1), -1)
        # final mlp
        X= self.out_mlp(X)
        X= self.out_norm(X)
        return X

    def encode_one(self, encoder, tensor):
        if encoder.input_dim==0:
            return None
        out= encoder(tensor)
        return out  # shape [B, hidden_dim]


###############################################################################
# 4) SimCLR-style training loop
###############################################################################

def simclr_loss(z1, z2, temperature=0.1):
    """
    Classic SimCLR InfoNCE for in-batch negatives:
    z1, z2 => [B, emb_dim]
    We'll produce 2B total embeddings, each sample i in z1 matches i in z2
    """
    bsz= z1.size(0)
    z1= F.normalize(z1, dim=-1)
    z2= F.normalize(z2, dim=-1)
    cat= torch.cat([z1, z2], dim=0) # => [2B, emb_dim]
    sim= torch.matmul(cat, cat.T) # => [2B, 2B]
    # remove diagonal
    mask= torch.eye(2*bsz, device=z1.device).bool()
    sim= sim/ temperature
    sim[mask]= -9999

    # positives:
    # For i in [0..B-1], pos= i+B
    # For i in [B..2B-1], pos= i-B
    # We'll do the approach from many simCLR code references:
    pos= torch.cat([
        sim[i, i+ bsz].unsqueeze(0) for i in range(bsz)
    ] + [
        sim[i, i- bsz].unsqueeze(0) for i in range(bsz, 2*bsz)
    ], dim=0)
    logsum= torch.logsumexp(sim, dim=1)
    loss= - torch.mean(pos- logsum)
    return loss

def run_epoch_simclr(model, loader, optimizer, device, train=True, augment_prob=0.5):
    if train:
        model.train()
    else:
        model.eval()

    total_loss=0
    for batch_in in loader:
        bsz= batch_in["patient_id"].size(0)
        # We'll produce v1, v2 for each sample in the batch => feed model => gather => simclr_loss
        # shape => z1: [B, emb_dim], z2: [B, emb_dim]
        z1_list= []
        z2_list= []

        # move batch to device
        # but we'll do the augmentation sample-by-sample
        for i in range(bsz):
            single_item={}
            for k in batch_in:
                # shape => [B, dim]
                single_item[k]= batch_in[k][i].unsqueeze(0)  # => shape [1, dim]

            v1, v2= simclr_augment(single_item, augment_prob=augment_prob)
            # forward => z1, z2
            z1= model_forward_one(model, v1, device)
            z2= model_forward_one(model, v2, device)
            z1_list.append(z1)
            z2_list.append(z2)

        z1_batch= torch.cat(z1_list, dim=0) # => [B, emb_dim]
        z2_batch= torch.cat(z2_list, dim=0) # => [B, emb_dim]

        loss= simclr_loss(z1_batch, z2_batch, temperature=0.1)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss+= loss.item()

    return total_loss/ len(loader)


def model_forward_one(model, item_dict, device):
    # item_dict => each field has shape [1, dim]
    # move to device
    dd={}
    for k,v in item_dict.items():
        dd[k]= v.to(device)
    out= model(dd) # => shape [1, emb_dim]
    return out


###############################################################################
# 5) Clustering & post-hoc
###############################################################################
def cluster_umap_hdbscan(embeddings, min_cluster_size=10):
    """
    returns (labels, silhouette, emb_2d)
    If fewer than 2 points => return dummy
    """
    n= embeddings.shape[0]
    if n<2:
        return np.array([-1]*n), None, None
    reducer= umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    emb_2d= reducer.fit_transform(embeddings)
    clusterer= hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels= clusterer.fit_predict(emb_2d)
    unique_labels= np.unique(labels)
    if len(unique_labels)<2:
        sil= None
    else:
        sil= silhouette_score(embeddings, labels)
    return labels, sil, emb_2d

def cluster_tsne_kmeans(embeddings, n_clusters=5):
    n= embeddings.shape[0]
    if n<2:
        return np.array([-1]*n), None, None
    ts= TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    emb_2d= ts.fit_transform(embeddings)
    if n_clusters>1 and n>= n_clusters:
        km= KMeans(n_clusters=n_clusters, random_state=42)
        labels= km.fit_predict(embeddings)
        sil= silhouette_score(embeddings, labels)
    else:
        labels= np.array([0]*n)
        sil= None
    return labels, sil, emb_2d

def posthoc_enrichment(ids, labels, cyto_dict):
    """
    cluster -> means
    """
    data_map={}
    for pid,lab in zip(ids, labels):
        if lab<0:
            continue
        if pid in cyto_dict:
            feats= cyto_dict[pid]
            if lab not in data_map:
                data_map[lab]={}
            for fkey in feats:
                if fkey=="patient_id": continue
                if fkey not in data_map[lab]:
                    data_map[lab][fkey]= []
                data_map[lab][fkey].append(feats[fkey])

    rows=[]
    for clus in sorted(data_map.keys()):
        row={"cluster": clus}
        for fkey, arr in data_map[clus].items():
            # attempt float
            vals=[]
            for x in arr:
                try:
                    vals.append(float(x))
                except:
                    pass
            if len(vals)>0:
                row[f"{fkey}_mean"]= np.mean(vals)
        rows.append(row)
    df= pd.DataFrame(rows)
    return df


###############################################################################
# 6) Main
###############################################################################
def main():
    parser= argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--top_genes", type=int, default=2000)
    parser.add_argument("--modality_subset", type=str, default="all",
                        help="Comma-separated or 'all'. E.g. 'cnv_broad,expression'")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--final_emb_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--hyperparam_search", type=int, default=0)
    parser.add_argument("--cytogenetics_file", type=str, default=None,
        help="Optional CSV with columns [patient_id, 1q_gain, t414, etc.] for post-hoc enrichment")

    args= parser.parse_args()

    # 1) create dataset
    dataset= MyelomaDataset(
        data_dir=args.data_dir,
        top_genes=args.top_genes,
        cytogenetics_file=args.cytogenetics_file
    )
    # figure out dims from sample[0]
    sample= dataset[0]
    full_dims={
      "cnv_broad_dim": sample["cnv_broad"].shape[0],
      "cnv_focal_dim": sample["cnv_focal"].shape[0],
      "expression_dim": sample["expression"].shape[0],
      "snv_dim": sample["snv"].shape[0],
      "fusion_dim": sample["fusion"].shape[0],
      "trans_dim": sample["trans"].shape[0]
    }

    # subset
    if args.modality_subset.lower()=="all":
        used_dims= full_dims
        count=6
    else:
        subset= [m.strip().lower() for m in args.modality_subset.split(",")]
        used_dims={}
        mod_list=["cnv_broad","cnv_focal","expression","snv","fusion","trans"]
        count=0
        for m_ in mod_list:
            key= f"{m_}_dim"
            if m_ in subset:
                used_dims[key]= full_dims[key]
                count+=1
            else:
                used_dims[key]=0

    # We define model config
    def build_config(dim_map):
        c={
          "cnv_broad_dim": dim_map["cnv_broad_dim"],
          "cnv_focal_dim": dim_map["cnv_focal_dim"],
          "expression_dim": dim_map["expression_dim"],
          "snv_dim":       dim_map["snv_dim"],
          "fusion_dim":    dim_map["fusion_dim"],
          "trans_dim":     dim_map["trans_dim"],
          "hidden_dim": args.hidden_dim,
          "dropout": args.dropout,
          "num_heads": args.num_heads,
          "final_emb_dim": args.final_emb_dim,
          "num_modalities_used": count
        }
        return c

    from torch.optim import Adam

    def train_and_eval(param_cfg, run_name="run"):
        # build model
        mcfg= build_config(param_cfg["modality_dims"])
        model= MyelomaMultiModalModel(mcfg)
        device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        optim= Adam(model.parameters(), lr= param_cfg["lr"])

        # split train/val
        n_total= len(dataset)
        n_train= int(0.8* n_total)
        n_val= n_total- n_train
        train_ds, val_ds= random_split(dataset,[n_train,n_val], generator=torch.Generator().manual_seed(42))
        train_loader= DataLoader(train_ds, batch_size= param_cfg["batch_size"], shuffle=True)
        val_loader= DataLoader(val_ds,   batch_size= param_cfg["batch_size"], shuffle=False)

        for ep in range(param_cfg["epochs"]):
            tr_loss= run_epoch_simclr(model, train_loader, optim, device, train=True, augment_prob=0.5)
            val_loss= run_epoch_simclr(model, val_loader, optim, device, train=False, augment_prob=0.5)
            print(f"[{run_name}] epoch {ep+1}/{param_cfg['epochs']} => train={tr_loss:.4f} val={val_loss:.4f}")

        # final embeddings
        all_ids=[]
        all_embs=[]
        model.eval()
        loader= DataLoader(dataset, batch_size= param_cfg["batch_size"], shuffle=False)
        with torch.no_grad():
            for b_in in loader:
                bsz= b_in["patient_id"].size(0)
                Zall=[]
                for i in range(bsz):
                    single_item={}
                    for k in b_in:
                        single_item[k]= b_in[k][i].unsqueeze(0) # shape [1, dim]
                    z= model_forward_one(model, single_item, device)
                    Zall.append(z)
                # shape => [bsz, emb_dim]
                Zcat= torch.cat(Zall, dim=0).cpu().numpy()
                all_embs.append(Zcat)
                pids= b_in["patient_id"].cpu().numpy().tolist()
                all_ids.extend(pids)
        embs= np.concatenate(all_embs, axis=0)
        return model, np.array(all_ids), embs

    if args.hyperparam_search==1:
        param_grid=[
          {"modality_dims": used_dims, "lr": args.lr, "batch_size": args.batch_size, "epochs": args.epochs},
          {"modality_dims": used_dims, "lr": args.lr, "batch_size": args.batch_size, "epochs": args.epochs+5}
        ]
        results=[]
        for i,pg in enumerate(param_grid):
            run_name= f"grid{i+1}"
            print(f"\n=== Running {run_name} => {pg}")
            model, ids, emb= train_and_eval(pg, run_name=run_name)
            # cluster
            labels1, sil1, emb2d1= cluster_umap_hdbscan(emb)
            labels2, sil2, emb2d2= cluster_tsne_kmeans(emb)
            results.append({"run": run_name, "sil_umap_hdbscan": sil1, "sil_tsne_kmeans": sil2})
        print("\n=== Search Results ===")
        for r_ in results:
            print(r_)
    else:
        # single run
        param_set={
          "modality_dims": used_dims,
          "lr": args.lr,
          "batch_size": args.batch_size,
          "epochs": args.epochs
        }
        print(f"\n--- Single run => {param_set}")
        model, ids, embs= train_and_eval(param_set, run_name="single_run")
        labels1, sil1, emb2d1= cluster_umap_hdbscan(embs)
        labels2, sil2, emb2d2= cluster_tsne_kmeans(embs)
        print(f"\nUMAP+HDBSCAN silhouette={sil1}, t-SNE+KMeans silhouette={sil2}")

        # Optionally post-hoc
        d_cyto= dataset.cytogenetics
        if len(d_cyto)>0:
            df_enr1= posthoc_enrichment(ids, labels1, d_cyto)
            df_enr2= posthoc_enrichment(ids, labels2, d_cyto)
            df_enr1.to_csv("single_run_umap_hdbscan_enrichment.csv", index=False)
            df_enr2.to_csv("single_run_tsne_kmeans_enrichment.csv", index=False)
            print("Saved post-hoc enrichment results.")


if __name__=="__main__":
    main()
