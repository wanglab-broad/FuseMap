#!/bin/bash
#SBATCH --job-name=baseline_test
#SBATCH --time=4:00:00
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --output=baseline_%j.log

set -e  # exit on error

cd /ibex/user/wuj0c/Projects/LLM/FuseMap-revision
source .venv/bin/activate

# The pre-DiscGAL commit (before entropy gating was added)
BASELINE_COMMIT="c6f5dfd"
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

echo "=== Step 1: Checkout baseline code (pre-DiscGAL) ==="
git checkout ${BASELINE_COMMIT} -- fusemap/loss.py fusemap/train_model.py

echo "=== Step 2: Run baseline training ==="
python main.py \
  --input_data_folder_path ./example_data/application_data/disease/ \
  --output_save_dir ./output/baseline_disease_test/ \
  --mode map \
  --pretrain_model_path ./molCCF/

echo "=== Step 3: Restore DiscGAL code ==="
git checkout ${CURRENT_BRANCH} -- fusemap/loss.py fusemap/train_model.py

echo "=== Step 4: Run comparison visualization ==="
python scripts/compare_discgal_baseline.py

echo "=== Done ==="
