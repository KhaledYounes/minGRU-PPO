#!/bin/bash
#SBATCH -t 2:00:00
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH -J "Train minigrid GRU-PPO vs minGRU-PPO"
#SBATCH --output=logs/GRU-PPO_vs_minGRU-PPO_%j.txt

export OMP_NUM_THREADS=16

srun python3 train.py
