#!/bin/bash -l

#SBATCH --job-name="train_ae_gpuxl"
#SBATCH --output=logs/train_ae_gpuxl_%j.log
#
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpuxl

source ~/.venvs/lpdm/bin/activate

array=(
    "ae=64x64x4_small_attention optim=adamw_1e-4_0_constant_1e0"
    "ae=64x64x4_medium_attention optim=adamw_1e-4_0_constant_1e0"
    "ae=64x64x4_medium_spectral optim=adamw_1e-4_0_constant_1e0"
    "ae=64x64x4_large_attention optim=adamw_1e-4_0_constant_1e0"
)

for i in ${!array[@]}; do
    srun --output=logs/train_ae_gpuxl_%j_$i.log --ntasks=1 --cpus-per-task=16 --gpus-per-task=1 --exact python train_autoencoder.py --gpuxl ${array[$i]} &
done

wait
