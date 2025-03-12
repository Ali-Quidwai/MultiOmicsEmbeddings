# Multiple Myeloma Multi-Omics: A SimCLR-Based Contrastive Learning Approach

This repository demonstrates how to create patient embeddings from multi-modal Myeloma data (CNV broad/focal, SNV, gene expression, fusions, translocations) using **SimCLR-like** contrastive learning combined with WeightedEncoders and Cross-Modal Attention.

## Table of Contents
1. [Overview](#overview)
2. [Data & Preprocessing](#data--preprocessing)
3. [Scripts](#scripts)
   - [Consolidated Script](#consolidated-script)
   - [run_grid_search.py](#run_grid_searchpy)
4. [Hyperparameter Grid Search](#hyperparameter-grid-search)
5. [Results & Interpretation](#results--interpretation)
6. [Next Steps](#next-steps)
7. [References & Acknowledgments](#references--acknowledgments)

---

## Overview
We aim to:
- Integrate **multiple** genomic modalities (CNV, SNV, expression, fusion, etc.).
- Train a **SimCLR-like** model that encourages embeddings of augmented views of the same sample to be close, and embeddings of different samples to be apart.
- Produce a final **patient embedding** that we can cluster using **UMAP+HDBSCAN** or **t-SNE+KMeans**, measuring cluster quality via **silhouette** score.

**Key Points**:
- **WeightedEncoders** let each modality be scaled by a trainable weight.
- **Cross-Modal Attention** merges embeddings across modalities.
- **SimCLR** data augmentation: random scale and noise per sample.
- Evaluate cluster separation with **silhouette** on the final embedding.

---

## Data & Preprocessing

We primarily used **COMMPASS_655_data** containing:
- `COMMPASS_655_cnvBroad.csv`
- `COMMPASS_655_cnvFocal.csv`
- `COMMPASS_655_gene_expression.csv`
- `COMMPASS_655_SNV.csv`
- `COMMPASS_655_fusion_gene.csv`
- `COMMPASS_655_translocations.csv`

**Requirements**:
- **Same number of rows** in each file (655) with **aligned order** of samples.
- We scale continuous features (CNV, expression) with `StandardScaler`.
- We cast SNV/fusion/translocation data to `float32`.
- We optionally select **top N** variable genes from the expression matrix (e.g., top 500 or top 2000).

---

## Scripts

### Consolidated Script
We provide [`consolidated_myeloma_experiment.py`](./consolidated_myeloma_experiment.py) which:

1. **Loads** data from the CSV files (with optional top-geness filtering).
2. **Builds** WeightedEncoders and cross-modal attention for a multi-modal model.
3. **Trains** using a SimCLR-like loss function across mini-batches.
4. **Clusters** final embeddings with UMAP+HDBSCAN and t-SNE+KMeans.
5. **Saves**:
   - `*_embeddings.csv`: final embeddings (one row per sample).
   - `*_clusters.csv`: cluster labels for each sample.
   - `*_umap_tsne_coords.csv`: 2D coordinates from UMAP and t-SNE.

**Example Command**:
```bash
python consolidated_myeloma_experiment.py \
  --data_dir "/Users/ali/Desktop/Alison Project /COMMPASS_655_data" \
  --cnv_broad_file COMMPASS_655_cnvBroad.csv \
  --cnv_focal_file COMMPASS_655_cnvFocal.csv \
  --expr_file COMMPASS_655_gene_expression.csv \
  --snv_file COMMPASS_655_SNV.csv \
  --fusion_file COMMPASS_655_fusion_gene.csv \
  --trans_file COMMPASS_655_translocations.csv \
  --top_genes 2000 \
  --epochs 5 \
  --batch_size 4 \
  --output_prefix "my_655_run"
```

### run_grid_search.py
We also have [`run_grid_search.py`](./run_grid_search.py) which **loops** over:
- different `top_genes` values (e.g. 500 vs. 2000),
- `batch_size` (8 vs. 16),
- `epochs` (10 vs. 20),
- `modality_subset` (all vs. expression,cnv_broad),

and calls the **consolidated script** for each combination. It captures the final silhouette scores from the console output so you can compare them side by side.

---

## Hyperparameter Grid Search

We ran a grid over:

- **`top_genes`** ∈ {500, 2000}
- **`batch_size`** ∈ {8, 16}
- **`epochs`** ∈ {10, 20}
- **`modality_subset`** ∈ {all, expression,cnv_broad}

For each combination, the script reports **UMAP+HDBSCAN** and **t-SNE+KMeans** silhouette scores.

Below is a summary of 16 runs:

| **Run #** | **top_genes** | **subset**            | **batch** | **epochs** | **UMAP/HDBSCAN** | **t-SNE/KMeans** |
|-----------|--------------:|-----------------------|----------:|----------:|-----------------:|-----------------:|
| **1**     | 500           | all                   | 8         | 10        | 0.1374           | 0.1485           |
| **2**     | 500           | expr+cnv_broad        | 8         | 10        | **0.1563**       | 0.1276           |
| **3**     | 500           | all                   | 8         | 20        | 0.1062           | 0.1329           |
| **4**     | 500           | expr+cnv_broad        | 8         | 20        | 0.0206           | 0.1357           |
| **5**     | 500           | all                   | 16        | 10        | 0.0707           | 0.1487           |
| **6**     | 500           | expr+cnv_broad        | 16        | 10        | 0.1401           | 0.1208           |
| **7**     | 500           | all                   | 16        | 20        | 0.0792           | 0.1224           |
| **8**     | 500           | expr+cnv_broad        | 16        | 20        | 0.0635           | 0.0968           |
| **9**     | 2000          | all                   | 8         | 10        | 0.0396           | 0.1444           |
| **10**    | 2000          | expr+cnv_broad        | 8         | 10        | 0.0627           | 0.1107           |
| **11**    | 2000          | all                   | 8         | 20        | 0.0266           | 0.1121           |
| **12**    | 2000          | expr+cnv_broad        | 8         | 20        | 0.0518           | 0.0952           |
| **13**    | 2000          | all                   | 16        | 10        | 0.1336           | **0.1606**       |
| **14**    | 2000          | expr+cnv_broad        | 16        | 10        | 0.0765           | 0.1009           |
| **15**    | 2000          | all                   | 16        | 20        | 0.0819           | 0.1102           |
| **16**    | 2000          | expr+cnv_broad        | 16        | 20        | 0.0270           | 0.1021           |

**Highlights**:
- **Highest UMAP+HDBSCAN** silhouette: **Run #2** (`top_genes=500`, `subset=expr+cnv_broad`, `batch=8`, `epochs=10`) → **0.1563**.
- **Highest t-SNE+KMeans** silhouette: **Run #13** (`top_genes=2000`, `subset=all`, `batch=16`, `epochs=10`) → **0.1606**.

---

## Results & Interpretation

1. **Silhouette Scores** around ~0.15–0.16 reflect modest cluster separation. This can be typical in complex diseases like multiple myeloma, where subtypes overlap.
2. **Longer training** (20 epochs) did **not** always improve silhouette—some runs ended up with lower values, possibly due to overfitting or the need for a different learning rate/temperature.
3. **Subset vs. All**:
   - Using **all** modalities can be noisier, but sometimes yields better t-SNE silhouette (e.g. run #13).
   - Restricting to `expression,cnv_broad` can help UMAP silhouette (e.g. run #2).

## Future work 
1. If you have **clinical labels** or known subtypes, check which clusters align with biologically meaningful groups. Silhouette alone might not tell the full story -> We have availability of Sherry subgroup labels(which will help in future works)

---
