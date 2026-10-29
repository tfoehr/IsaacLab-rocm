# Source this file (fish) to use the IsaacLab-rocm environment.
#   source scripts/env.fish
set -l _ilr_root (realpath (dirname (status filename))/..)
source $_ilr_root/.venv/bin/activate.fish
# Use the ROCm SDK that ships with the PyTorch wheels (not /opt/rocm) for HIP headers and tools.
set -gx ROCM_PATH ($_ilr_root/.venv/bin/rocm-sdk path --root)
set -gx HIP_PATH $ROCM_PATH
fish_add_path -g --prepend $ROCM_PATH/bin $ROCM_PATH/lib/llvm/bin
set -gx ISAACLAB_ROCM_ROOT $_ilr_root
# ROCm 7.13's VM-heap memory pool can zero-fill live allocations after pool memory is reused;
# use the classic pool. Must be set before any library initializes HIP.
set -gx DEBUG_HIP_MEM_POOL_VMHEAP 0
# PyTorch's Kineto registers a rocprofiler-sdk tool whose atexit handler calls _exit(0), so every
# Python process that used the GPU exits with status 0, even after an exception or sys.exit(1).
# Disabling rocprofiler tool registration keeps exit codes intact.
set -gx ROCPROFILER_REGISTER_ENABLED 0
