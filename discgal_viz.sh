#!/bin/bash
#SBATCH --job-name=discgal_viz
#SBATCH --time=0:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=discgal_viz_%j.log

cd /ibex/user/wuj0c/Projects/LLM/FuseMap-revision
source .venv/bin/activate

python scripts/visualize_discgal.py
