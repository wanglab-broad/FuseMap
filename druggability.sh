#!/bin/bash
#SBATCH --job-name=druggability
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=druggability_%j.log

cd /ibex/user/wuj0c/Projects/LLM/FuseMap-revision
source .venv/bin/activate

echo "=== Druggability: 13months-disease ==="
python scripts/druggability_analysis.py \
    --dataset 13months-disease-replicate_1.h5ad \
    --top_n_genes 10

echo ""
echo "=== Druggability: adata_ad_cosmx ==="
python scripts/druggability_analysis.py \
    --dataset adata_ad_cosmx.h5ad \
    --top_n_genes 10

echo "=== Done ==="
