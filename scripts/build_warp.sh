#!/usr/bin/env bash
# Build the ROCm Warp port for the GPUs on this machine and install it into the venv.
#   scripts/build_warp.sh [extra build_lib.py args, e.g. --mode=debug or --hip-arch=gfx1151]
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
source "$root/scripts/env.sh"
cd "$root/third_party/warp-rocm"
python build_lib.py --rocm-path="$ROCM_PATH" --jobs="$(nproc)" "$@"
uv pip install --python "$root/.venv/bin/python" --no-deps -e .
