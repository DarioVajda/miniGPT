#!/bin/bash

# run the download.sbatch script with parameter from set {0, 50000, 100000,... 2500000}

mkdir -p logs
mkdir -p train
mkdir -p validation

for i in $(seq 0 5000 2500000); do
    # set output file to logs/download_images_{i}.log
    sbatch --output=logs/download_images_${i}.log download.sbatch GCC-training.tsv train $i 5000
    sleep 3
done

# sbatch --output=logs/download_images_validation.log download.sbatch GCC-1.1.0-Validation.tsv validation 0