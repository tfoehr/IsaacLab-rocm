# Source this file (bash/zsh) to use the IsaacLab-rocm environment.
#   source scripts/env.sh
_ilr_root="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
source "$_ilr_root/.venv/bin/activate"
# Use the ROCm SDK that ships with the PyTorch wheels (not /opt/rocm) for HIP headers and tools.
export ROCM_PATH="$("$_ilr_root/.venv/bin/rocm-sdk" path --root)"
export HIP_PATH="$ROCM_PATH"
export PATH="$ROCM_PATH/bin:$ROCM_PATH/lib/llvm/bin:$PATH"
export ISAACLAB_ROCM_ROOT="$_ilr_root"
unset _ilr_root
