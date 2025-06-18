#!/bin/bash
#SBATCH -t 2:00:00
#SBATCH -q express
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:1
#SBATCH -J "Train minigrid GRU-PPO vs minGRU-PPO"
#SBATCH --output=logs/GRU-PPO_vs_minGRU-PPO_T_MAZE_%j.txt

srun python3 train.py
