#!/bin/bash
#SBATCH -t 2:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH -J "Test Min-GRU on Text"
#SBATCH --output=GRU_VS_MINGRU_ON_TEXT_%j.txt

export PYTHONPATH=..

srun python3 testing_gru.py
srun python3 testing_mingru.py
