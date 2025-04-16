# making a script to try only one cnv omic in the run.
# source PSN_AI_env/bin/activate

import subprocess
import pandas as pd
import os

top_genes_list = [500, 2000]
batch_sizes    = [8, 16]
epochs_list    = [10, 20]
cnv_types      = ["cnv_broad", "cnv_focal"]
modality_sets = [
    "all",
    "expression,cnv_broad",
    "expression,cnv_focal",
    "cnv_broad,expression,snv,fusion,trans",   # all omics + cnv_broad
    "cnv_focal,expression,snv,fusion,trans"    # all omics + cnv_focal
]


base_cmd = [
    "python", "consolidated_experiment.py",
    "--data_dir", "/Users/alisonpark/Documents/MultiOmicsEmbeddings/Final_Scripts_and_Results_on655/COMMPASS_655_data",
    "--expr_file", "COMMPASS_655_gene_expression.csv",
    "--snv_file", "COMMPASS_655_SNV.csv",
    "--fusion_file", "COMMPASS_655_fusion_gene.csv",
    "--trans_file", "COMMPASS_655_translocations.csv"
]

summary_rows = []
run_counter = 1

n_clusters = 12
hdbscan_min = 12


seen_configs = set()

for tg in top_genes_list:
    for bs in batch_sizes:
        for ep in epochs_list:
            for subset in modality_sets:
                
                # Sanitize subset for filename
                modality_tag = subset.replace(",", "_")
                
                # Avoid duplicate runs for the same combo
                config_id = (subset, tg, bs, ep)
                if config_id in seen_configs:
                    continue
                seen_configs.add(config_id)

                # Determine CNV file(s)
                if subset == "all":
                    # All files already provided in base_cmd
                    cnv_flag = []  # No extra --cnv_*_file needed
                elif "cnv_broad" in subset:
                    cnv_flag = ["--cnv_broad_file", "COMMPASS_655_cnvBroad.csv"]
                elif "cnv_focal" in subset:
                    cnv_flag = ["--cnv_focal_file", "COMMPASS_655_cnvFocal.csv"]
                else:
                    cnv_flag = []  # No CNV in this subset

                out_prefix = f"grid_run_{run_counter}_{modality_tag}_{tg}g_{bs}bs_{ep}ep_k{n_clusters}_hdb{hdbscan_min}"

                cmd = base_cmd + cnv_flag + [
                    "--top_genes", str(tg),
                    "--batch_size", str(bs),
                    "--epochs", str(ep),
                    "--modality_subset", subset,
                    "--output_prefix", out_prefix,
                    "--n_clusters", str(n_clusters),
                    "--hdbscan_min_cluster_size", str(hdbscan_min)
                ]

                print(f"\n=== Running: top_genes={tg}, bs={bs}, epochs={ep}, subset={subset} ===")
                subprocess.run(cmd)

                # Read metrics
                metrics_file = f"{out_prefix}_metrics.csv"
                sil_umap = sil_tsne = num_umap = num_tsne = None
                if os.path.exists(metrics_file):
                    try:
                        metrics_df = pd.read_csv(metrics_file)
                        sil_umap = metrics_df.at[0, "silhouette_umap"]
                        sil_tsne = metrics_df.at[0, "silhouette_tsne"]
                        num_umap = metrics_df.at[0, "num_umap_clusters"]
                        num_tsne = metrics_df.at[0, "num_tsne_clusters"]
                    except Exception as e:
                        print(f"Warning: Could not read {metrics_file}: {e}")

                summary_rows.append({
                    "run_id": run_counter,
                    "output_prefix": out_prefix,
                    "top_genes": tg,
                    "batch_size": bs,
                    "epochs": ep,
                    "modality_subset": subset,
                    "silhouette_umap": sil_umap,
                    "num_umap_clusters": num_umap,
                    "silhouette_tsne": sil_tsne,
                    "num_tsne_clusters": num_tsne
                })

                run_counter += 1

# Save final summary CSV
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("grid_summary.csv", index=False)
print("\n✅ Saved all grid search results to grid_summary.csv")

