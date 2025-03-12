#!/usr/bin/env python3

import subprocess

top_genes_list = [500, 2000]
batch_sizes    = [8, 16]
epochs_list    = [10, 20]
modality_sets  = ["all", "expression,cnv_broad"]

base_cmd = [
    "python", "consolidated_myeloma_experiment.py",
    "--data_dir", "/Users/ali/Desktop/Alison Project /COMMPASS_655_data", 
    "--cnv_broad_file", "COMMPASS_655_cnvBroad.csv",
    "--cnv_focal_file", "COMMPASS_655_cnvFocal.csv",
    "--expr_file",      "COMMPASS_655_gene_expression.csv",
    "--snv_file",       "COMMPASS_655_SNV.csv",
    "--fusion_file",    "COMMPASS_655_fusion_gene.csv",
    "--trans_file",     "COMMPASS_655_translocations.csv"
]

run_counter=1

for tg in top_genes_list:
    for bs in batch_sizes:
        for ep in epochs_list:
            for subset in modality_sets:
                out_prefix = f"grid_run_{run_counter}"
                cmd = base_cmd + [
                    "--top_genes", str(tg),
                    "--batch_size", str(bs),
                    "--epochs", str(ep),
                    "--modality_subset", subset,
                    "--output_prefix", out_prefix
                ]

                print(f"\n=== Running: top_genes={tg}, bs={bs}, epochs={ep}, subset={subset} ===")
                subprocess.run(cmd)
                run_counter +=1
