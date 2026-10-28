# IsaacLab-rocm

Isaac Lab on AMD GPUs with ROCm, built and tested on a Ryzen AI MAX+ 395 (Strix Halo,
Radeon 8060S, `gfx1151`). It runs Isaac Lab's kit-less workflow (Newton physics with the
MuJoCo-Warp solver, headless RL training) on the GPU through a HIP port of NVIDIA Warp.

## What runs, and what cannot

| Works on ROCm | Cannot work on ROCm |
|---|---|
| Newton physics with the MuJoCo-Warp solver on the GPU | Isaac Sim / Omniverse Kit, the RTX renderer and RTX cameras |
| Manager-based and direct RL tasks on the Newton backend | PhysX GPU physics (`isaaclab_physx`) and OvPhysX/OVRTX |
| RSL-RL, skrl (PyTorch), SB3, rl_games training | XR teleoperation, Isaac Lab Mimic |
| Ray casters, contact sensors, terrains (Warp mesh/BVH) | JAX on CUDA (`skrl-jax`) |

The right-hand column consists of closed-source NVIDIA binaries; there is nothing to port.

Validated end to end (`rsl_rl`, 1024 envs): `Isaac-Cartpole`, `Isaac-Ant`,
`Isaac-Velocity-Flat-G1`, `Isaac-Velocity-Rough-G1`. A full `Isaac-Velocity-Flat-G1` training
(4096 envs, 1500 iterations, 83 min) reaches a mean reward of 28.9 with every episode running
to the 1000-step limit.

Newton's Kamino solver does not work on ROCm yet (see known issues).

## Layout

- `third_party/warp-rocm`: Warp 1.17 with the HIP backend (branch `rocm-1.17-gfx1151`)
- `third_party/newton`, `third_party/mujoco_warp`: small ROCm fixes (branch `rocm-hip`)
- `isaaclab`: Isaac Lab (branch `rocm`), currently unmodified
- `components.toml`: upstream, base revision and branch of each component
- `scripts/`: environment, build and install scripts; `tools/run_warp_tests.py`: crash-isolated Warp test runner

## Setup

Requirements: Linux, an AMD GPU supported by the TheRock ROCm nightlies, `uv`, GCC.
The ROCm toolchain comes from Python packages in the venv; a system ROCm is not needed
and is deliberately not used.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
    --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match --prerelease allow \
    "torch==2.12.0a0+rocm7.13.0a20260411" "torchvision==0.27.0a0+rocm7.13.0a20260411" \
    "rocm[devel]==7.13.0a20260411"
.venv/bin/rocm-sdk init
scripts/build_warp.sh      # builds libwarp for the GPUs in this machine
scripts/install.sh         # Isaac Lab + Newton + MuJoCo-Warp, with the ROCm overrides
```

For other GPUs, use the matching TheRock index (e.g. `.../v2/gfx110X-all/`) and pin the
torch/ROCm versions in `requirements/rocm-overrides.txt` accordingly.

## Usage

```bash
source scripts/env.sh        # or: source scripts/env.fish
cd isaaclab
python scripts/reinforcement_learning/train.py --rl_library rsl_rl \
    --task Isaac-Velocity-Rough-G1 --num_envs 4096 --device cuda:0
```

ROCm devices keep the `cuda:N` names in both PyTorch and Warp, so Isaac Lab configs and
scripts need no changes.

Warp tests: `python tools/run_warp_tests.py -j 6` (one process per test module; results in
`test_results/warp/`).

## How the port works

**Warp on HIP.** AMD's ROCm Warp fork (Warp 1.13, MI300-class `gfx942`) was merged up to
Warp 1.17 and extended for RDNA:

- Wavefront size is derived per compile target (wave32 on RDNA, wave64 on CDNA) instead of
  being fixed at 64; warp-level reductions, ballots and shuffles use matching lane masks.
- Kernels are compiled at runtime with hipRTC for the detected `gfx` target; the native
  library is built for the GPUs found by `amdgpu-arch`.
- CUDA-only features are disabled on HIP: thread block clusters, `.wrp` graph export
  (`capture_save`), MathDx/cuBLASDx. The rocWMMA `tile_matmul` fast path is limited to CDNA.

**One HIP runtime.** PyTorch's ROCm wheels ship their own ROCm (`rocm-sdk` packages).
Warp is built against the same SDK, preloads its HIP runtime before opening `warp.so`, and
takes hipRTC headers from it, so Warp and PyTorch share one `libamdhip64` and interoperate
zero-copy, regardless of import order or a system ROCm in the loader path.

**Conditional graphs.** HIP has no conditional graph nodes, which Newton and MuJoCo-Warp use
for early-exit solver loops (`wp.capture_while`, `wp.capture_if`). Warp emulates them on HIP
(`warp.config.hip_conditional_graph_emulation`): the capture is split into graph segments,
a captured kernel publishes each condition to pinned host memory, and `wp.capture_launch`
replays the segments as a host-driven program. The overhead is about 6 us per evaluated
condition on Strix Halo; solvers still exit as soon as they converge.

**Graph-captured allocations.** HIP graph memory nodes can be invalidated when another graph
is captured (CUDA guarantees they stay valid for the graph's lifetime). On HIP, arrays
allocated during a Warp capture are regular device allocations owned by the captured graph
instead (`warp.config.hip_graph_retained_allocations`).

**ROCm memory pool bug.** ROCm 7.13's "VM heap" stream-ordered memory pool can zero-fill a live
allocation after freed pool memory is reused and a host copy of ~20 KB or more touches the reused
range (`tools/repro_hip_vmheap_pool.py` reproduces it with plain HIP calls). Warp sets
`DEBUG_HIP_MEM_POOL_VMHEAP=0` before loading the HIP runtime, and `scripts/env.sh` exports it so it
also applies when PyTorch initializes HIP first.

## Physics validation

MuJoCo-Warp's test suite (compared against MuJoCo C 3.15.0) and Newton's test suite were run on
gfx1151. Everything Isaac Lab's locomotion tasks use passes: the MuJoCo-Warp solver, articulations
and kinematics, joint limits, drives and actuators, contacts and friction, USD/URDF/MJCF import,
and contact, ray-cast and IMU sensors.

## Status and known issues

- Performance tuning is in progress (see `logs/`); measured with 1024 envs while the GPU was
  shared with other work: Cartpole ~138k, Ant ~65k, Flat G1 ~20k, Rough G1 ~15k env steps/s.
- The OpenGL viewer is untested on ROCm; train headless, or watch training and trained policies
  in the browser with the viser visualizer (`--viz viser`, serves `http://localhost:8080`).
- AMD GPUs have no independent thread scheduling, so GPU code where one lane waits on another
  (spin locks) deadlocks, and on gfx1151 the driver often cannot recover from the hang (the
  desktop freezes until a reboot). Newton's viewer picking used such a lock; it is now lock-free.
- Newton Kamino (shared memory limits, 32-bit shuffle masks, GPU-only friction drift) does not
  work on HIP yet; a port is in progress.
- An intermittent GPU hang occurs in Newton's `test_collision_pipeline` when run after other tests.
- Warp tests need the LFS assets: install `git-lfs` and run `git lfs pull` in `third_party/warp-rocm`.
- The component branches exist in local checkouts only; push them to your own forks to share them.
