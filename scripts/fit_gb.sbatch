#!/bin/bash
#SBATCH --job-name=gbfit
#SBATCH --output=/home/akumar/gb-agent/logs/gbfit_%x_%j.log
#SBATCH --error=/home/akumar/gb-agent/logs/gbfit_%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

source /apps/compilers/anaconda3-24.2/etc/profile.d/conda.sh
conda activate llm_env
cd /home/akumar/gb-agent
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "python: $(which python)"
python scripts/fit_gb.py --dataset ${DATASET} --theta ${THETA:-0.95} ${EXTRA}
