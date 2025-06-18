#!/bin/bash
#SBATCH -t 2:00:00
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:1
#SBATCH -J "Test Fused Parallel Scan"
#SBATCH --output=benchmarking_GRU_vs_minGRU_%j.txt

srun python3 benckmarking_minGRU_vs_GRU.py
