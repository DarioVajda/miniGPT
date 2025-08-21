#!/bin/bash
#SBATCH --job-name=download_images
#SBATCH --output=download_images_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00

echo "Starting job: $SLURM_JOB_NAME"
module load Anaconda3
source activate vit_env
echo "Activated conda environment: vit_env"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

srun python download_images.py $1 $2 $3 $4