#!/usr/bin/env bash
# Install Isaac Lab (kitless, Newton backend) on ROCm into the IsaacLab-rocm venv.
# Prerequisite: scripts/build_warp.sh has built and installed the ROCm Warp port.
#   scripts/install.sh [extra uv pip args]
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
source "$root/scripts/env.sh"
py="$root/.venv/bin/python"
il="$root/isaaclab"

# Isaac Lab packages that work without Isaac Sim / Kit. isaaclab_physx and isaaclab_ov stay
# installed because task configs reference their (lazily imported) config classes.
members=(isaaclab isaaclab_assets isaaclab_contrib isaaclab_experimental isaaclab_newton isaaclab_ov
         isaaclab_physx isaaclab_ppisp isaaclab_rl isaaclab_tasks isaaclab_tasks_experimental isaaclab_visualizers)
editables=(-e "$root/third_party/warp-rocm" -e "$root/third_party/mujoco_warp" -e "$root/third_party/newton[sim]" -e "$il")
for m in "${members[@]}"; do editables+=(-e "$il/source/$m"); done
# Browser viewer (--viz viser), Newton recordings and remeshing, and the test runners
# (Newton/MuJoCo-Warp tests, scripts/healthcheck.sh).
extras=("viser>=1.1.1" "cbor2>=5.7.0" "pyfqmr>=0.5.0" pytest pytest-xdist pytest-timeout)

# --no-sources: ignore Isaac Lab's CUDA torch index and git sources; the overrides pin the ROCm stack.
uv pip install --python "$py" --no-sources \
    --index-url https://pypi.org/simple \
    --extra-index-url https://rocm.nightlies.amd.com/v2/gfx1151/ \
    --index-strategy unsafe-best-match --prerelease allow \
    --override "$root/requirements/rocm-overrides.txt" \
    "${editables[@]}" "${extras[@]}" "$@"

# The path overrides above install the local checkouts as regular packages; switch them to
# editable installs so changes in third_party/ take effect without reinstalling.
uv pip install --python "$py" --no-deps --no-sources \
    -e "$root/third_party/warp-rocm" -e "$root/third_party/mujoco_warp" -e "$root/third_party/newton"
