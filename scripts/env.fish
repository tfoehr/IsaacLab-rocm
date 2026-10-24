# Source this file (fish) to use the IsaacLab-rocm environment.
#   source scripts/env.fish
set -l _ilr_root (realpath (dirname (status filename))/..)
source $_ilr_root/.venv/bin/activate.fish
# Use the ROCm SDK that ships with the PyTorch wheels (not /opt/rocm) for HIP headers and tools.
set -gx ROCM_PATH ($_ilr_root/.venv/bin/rocm-sdk path --root)
set -gx HIP_PATH $ROCM_PATH
fish_add_path -g --prepend $ROCM_PATH/bin $ROCM_PATH/lib/llvm/bin
set -gx ISAACLAB_ROCM_ROOT $_ilr_root
