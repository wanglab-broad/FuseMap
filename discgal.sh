#!/bin/bash
#SBATCH --job-name=discgal_test
#SBATCH --time=4:00:00
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --output=discgal_%j.log

# 激活环境 (根据你实际的 conda 环境名调整)
# source activate fusemap  # 或 conda activate fusemap
cd /ibex/user/wuj0c/Projects/LLM/FuseMap-revision

source .venv/bin/activate



python main.py \
  --input_data_folder_path ./example_data/application_data/disease/ \
  --output_save_dir ./output/discgal_disease_test/ \
  --mode map \
  --pretrain_model_path ./molCCF/
