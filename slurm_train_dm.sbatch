#!/bin/bash
#SBATCH -A m3443
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 04:00:00
#SBATCH -C gpu
#SBATCH -J dm_latents
#SBATCH -o logs/slurm/%x_%j.out
#SBATCH -e logs/slurm/%x_%j.err
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH -c 8

set -euo pipefail

# --- Perlmutter modules + your Poetry env ---
source /opt/cray/pe/cpe/24.07/restore_lmod_system_defaults.sh >/dev/null 2>&1 || true
module load PrgEnv-gnu
module load cudatoolkit/12.4
module load cudnn/9.5.0
module load cray-mpich/8.1.30
module load nccl


export MPICH_GPU_SUPPORT_ENABLED=1
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5
export NCCL_SOCKET_IFNAME=hsn,ib
export NCCL_NET_GDR_LEVEL=2

mkdir -p logs/slurm

source ~/.bashrc || true
poetry env info --path >/dev/null 2>&1 || true
LD_LIBRARY_PATH_SAVE="$LD_LIBRARY_PATH"
eval "$(poetry env activate)"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH_SAVE}:${LD_LIBRARY_PATH:-}"
unset LD_LIBRARY_PATH_SAVE

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

export WANDB_ENTITY=${WANDB_ENTITY:-kathlynnsimotas}
export WANDB_PROJECT=${WANDB_PROJECT:-lola-dm}


# Torchrun: 1 node x 4 procs (matches --ntasks-per-node)
srun torchrun \
  --nnodes 1 \
  --nproc-per-node 4 \
  --standalone \
  /pscratch/sd/k/ksimotas/lola/experiments/train_diffusion.py \
  dataset.path=/global/cfs/cdirs/m3443/usr/xju/Fundra/data/vqvae/v0.6.0/predictions/train \
  dataset.valid_path=/global/cfs/cdirs/m3443/usr/xju/Fundra/data/vqvae/v0.6.0/predictions/val \
  dataset.latent_key=z_q \
  dataset.item_index=0 \
  dataset.stats_file=/pscratch/sd/k/ksimotas/lola/latent_channel_stats.pt \
  dataset.channels=512