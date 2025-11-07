# activate_lola.sh
# Usage: source activate_lola.sh
# (must be sourced so the activation persists in your current shell)

# 1) Make sure 'module' is available
if ! type module &>/dev/null; then
  # Perlmutter: load environment modules
  source /etc/profile.d/modules.sh
fi

# Helper: try-load a module, fall back to load
try_load () {
  module try-load "$1" 2>/dev/null || module load "$1"
}

# 2) Load the CUDA/NCCL/cuDNN stack you need
try_load cudatoolkit/12.4
try_load cudnn/9
try_load nccl
# Perlmutter GPU architecture helper (no-op elsewhere)
module try-load craype-accel-nvidia80 >/dev/null 2>&1 || true

# Optional perf knobs (tweak to taste)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

# 3) Activate Poetry environment
if ! type poetry &>/dev/null; then
  echo "[activate_lola] Poetry not found on PATH. Load it or install it first."
  return 1
fi

# Prefer the user's request: eval $(poetry env activate)
if poetry help env 2>/dev/null | grep -q 'activate'; then
  eval "$(poetry env activate)"
else
  # Fallback for Poetry versions without 'env activate'
  VENV_PATH="$(poetry env info --path 2>/dev/null || true)"
  if [[ -z "$VENV_PATH" || ! -d "$VENV_PATH" ]]; then
    echo "[activate_lola] No venv yet; running 'poetry install'..."
    poetry install || { echo "[activate_lola] poetry install failed"; return 1; }
    VENV_PATH="$(poetry env info --path)"
  fi
  # shellcheck disable=SC1090
  source "$VENV_PATH/bin/activate"
fi

# 4) (Optional) quick sanity print
echo "[activate_lola] Activated: $(python -c 'import sys,platform;print(platform.python_version())')  |  Poetry venv: ${VIRTUAL_ENV:-unknown}"
