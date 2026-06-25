#!/bin/bash
#SBATCH --job-name=niche_id
#SBATCH --time=1:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=niche_id_%j.log

cd /ibex/user/wuj0c/Projects/LLM/FuseMap-revision
source .venv/bin/activate

echo "=== Running niche identification: 13months-disease ==="
python scripts/identify_altered_niches.py \
    --dataset 13months-disease-replicate_1.h5ad

echo ""
echo "=== Running niche identification: adata_ad_cosmx ==="
python scripts/identify_altered_niches.py \
    --dataset adata_ad_cosmx.h5ad

echo "=== Done ==="
