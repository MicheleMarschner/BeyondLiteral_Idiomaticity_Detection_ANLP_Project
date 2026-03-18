#!/bin/bash -eux
#SBATCH --job-name=anlp-array
#SBATCH --account=sci-demelo-computer-vision
#SBATCH --partition=gpu-batch
#SBATCH --constraint=ARCH:X86
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --array=0-71%6
#SBATCH --output=/sc/home/michele.marschner/project/beyondliteral_idiomaticity_detection_anlp_project/logs/array_%A_%a.out
#SBATCH --error=/sc/home/michele.marschner/project/beyondliteral_idiomaticity_detection_anlp_project/logs/array_%A_%a.err

source /sc/home/michele.marschner/conda3/etc/profile.d/conda.sh
conda activate anlp

cd /sc/home/michele.marschner/project/beyondliteral_idiomaticity_detection_anlp_project

mkdir -p logs

nvidia-smi || true
python -c "import torch; print('cuda available:', torch.cuda.is_available()); print('device count:', torch.cuda.device_count())"

python src/experiments/run_experiments.py \
  --experiments_path src/experiments/experiments_template.py \
  --slurm