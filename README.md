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
`Isaac-Velocity-Flat-G1`, `Isaac-Velocity-Rough-G1`. Full trainings with 4096 envs:
`Isaac-Velocity-Flat-G1` (1500 iterations, 83 min) reaches a mean reward of 28.9 and
`Isaac-Velocity-Rough-G1` (5000 iterations, 6.5 h) reaches 28.7, with episodes running to the
1000-step limit.

Newton's Kamino solver also runs on the GPU: its 681 unit tests and 29 examples pass, and the
G1, H1 and ANYmal examples match the CPU.

## Layout

- `third_party/warp-rocm`: Warp 1.17 with the HIP backend ([tfoehr/warp](https://github.com/tfoehr/warp), branch `rocm-1.17-gfx1151`)
- `third_party/newton`: ROCm fixes and the Kamino GPU port ([tfoehr/newton](https://github.com/tfoehr/newton), branch `rocm-hip`)
- `third_party/mujoco_warp`: small ROCm fixes ([tfoehr/mujoco_warp](https://github.com/tfoehr/mujoco_warp), branch `rocm-hip`)
- `isaaclab`: upstream Isaac Lab at commit `b03345bbf`, unmodified
- `components.toml`: upstream, base revision and branch of each component
- `scripts/`: environment, build and install scripts; `tools/run_warp_tests.py`: crash-isolated Warp test runner

## Setup

Requirements: Linux, an AMD GPU supported by the TheRock ROCm nightlies, `uv`, GCC.
The ROCm toolchain comes from Python packages in the venv; a system ROCm is not needed
and is deliberately not used.

```bash
git clone https://github.com/tfoehr/IsaacLab-rocm.git && cd IsaacLab-rocm
GIT_LFS_SKIP_SMUDGE=1 git clone -b stable-2026-10-30 https://github.com/tfoehr/warp.git third_party/warp-rocm
git clone -b stable-2026-10-30 https://github.com/tfoehr/newton.git third_party/newton
git clone -b stable-2026-10-30 https://github.com/tfoehr/mujoco_warp.git third_party/mujoco_warp
git clone https://github.com/isaac-sim/IsaacLab.git isaaclab && git -C isaaclab checkout b03345bbf
```

The `stable-2026-10-30` tags mark the state validated above; the branches carry newer work.

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

Then check the install with `scripts/healthcheck.sh` (see Usage).

### Reproducing the validated environment

`requirements/lock-2026-10-30.txt` is the `uv pip freeze` of the validated venv (component
commits in its header, which match the `stable-2026-10-30` tags). The local checkouts
(`third_party/*`, `isaaclab/*`) appear there only as comments because they are editable
installs of local paths; `build_warp.sh` and `install.sh` install them. To pin every other
package to exactly the validated versions, pass the lock file as a constraints file to the
last step of the setup above:

```bash
scripts/install.sh --constraint requirements/lock-2026-10-30.txt
```

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
`test_results/warp/`; `--modules test_codegen interop.test_torch ...` runs only those modules).

### Health check

```bash
scripts/healthcheck.sh       # ~5 min (~7 min on a fresh install: cold kernel cache); from any directory, no need to source env.sh first
```

It runs the GPU jobs one after another and prints a PASS/FAIL line per check, then
`RESULT: PASS` (exit code 0) or `RESULT: FAIL` (nonzero). Logs go to `logs/healthcheck/<timestamp>/`.
It refuses to start while another `train.py`/`play.py`/test process is running. The checks:

- `device`: PyTorch and Warp see the `gfx1151` GPU and load a single `libamdhip64`;
  a Warp kernel writes into a PyTorch tensor zero-copy; a process that used the GPU
  reports a nonzero exit status (see "Exit status" under known issues)
- `warp-tests`: 16 core Warp test modules (codegen, launches, atomics, graphs and HIP
  conditional-graph emulation, torch/DLPack interop, BVH/mesh, tiles, sorts and scans)
- `mujoco-warp`: MuJoCo-Warp's kinematics and constraint-solver tests (vs. MuJoCo C)
- `newton-g1`: Newton's G1 example on the MuJoCo-Warp solver, 200 frames in test mode
- `train-cartpole`, `train-g1-flat`: `Isaac-Cartpole` (256 envs, 20 iterations) and
  `Isaac-Velocity-Flat-G1` (512 envs, 5 iterations) with RSL-RL; the run must finish and every
  logged mean reward must be finite
- `kernel-log`: no amdgpu ring timeout, GPU reset or page fault in `journalctl -k` since the
  check started

**Rule: do not upgrade ROCm, PyTorch, the kernel or linux-firmware without running the health
check before and after.** If it fails afterwards, go back to the previous versions (see
"Reproducing the validated environment" under Setup).

## Robots for experiments

`extensions/isaaclab_rocm_robots` (installed editable by `scripts/install.sh`) adds tasks for
small real robots. It registers its gym ids through Isaac Lab's `isaaclab.tasks` entry point, so
`train.py`/`play.py` find them without changes to Isaac Lab.

| Task id | Robot | What it learns |
|---|---|---|
| `ROCm-Reach-SO101` | TheRobotStudio SO-101 arm | move the fingertip to points 0.15-0.30 m ahead, ±0.15 m sideways, 0.05-0.25 m high; joint position targets on the 5 arm joints |
| `ROCm-Lift-Cube-SO101` | SO-101 | pick a 25 mm, 30 g cube off the floor and hold it at a goal point; arm joint targets plus open/close gripper |
| `IsaacContrib-Velocity-Flat-MicroDuck`, `IsaacContrib-Velocity-Rough-MicroDuck` | Pollen Robotics MicroDuck biped | velocity-command walking (Isaac Lab's own tasks, used unchanged) |
| `ROCm-HeadPose-ReachyMini` (experimental) | Pollen Robotics Reachy Mini | turn the head (Stewart platform plus body yaw) to roll/pitch ±0.4 rad, yaw ±1.0 rad |

Every `ROCm-*` id has a `-Play` variant (smaller scene, no observation noise). The SO-101 tasks use
the SysID servo model from NVIDIA's hosted SO-101 USD (STS3215, 3.35 N·m); Reachy Mini uses
XL330-like servos (0.5 N·m) with passive ball joints and five loop closures solved as MuJoCo
equality constraints.

```bash
source scripts/env.sh
python tools/convert_assets.py        # once, for Reachy Mini (CPU only, ~10 s); see below
cd isaaclab
python scripts/reinforcement_learning/train.py --rl_library rsl_rl --task ROCm-Reach-SO101 \
    --num_envs 4096 --max_iterations 500 --device cuda:0
# watch the latest checkpoint in the browser (http://localhost:8080)
python scripts/reinforcement_learning/play.py --rl_library rsl_rl --task ROCm-Reach-SO101 \
    --checkpoint latest --num_envs 16 --viz viser --device cuda:0
```

`play.py` applies the play settings to any task id; `--checkpoint latest` looks up runs by task
id, so play the id you trained (or give a `-Play` id an explicit `--checkpoint <path to model_N.pt>`).
Runs are logged under `isaaclab/logs/rsl_rl/<experiment>/` (`rocm_so101_reach`,
`rocm_so101_lift`, `rocm_reachy_mini_head`, `microduck_velocity_flat`). The MicroDuck agent config
asks for 50000 iterations; pass `--max_iterations`.

Results on the Radeon 8060S with 4096 environments (RSL-RL PPO):

| Task | Iterations | Wall time | Env steps/s | Mean reward (start → end) | Episode length | Outcome |
|---|---|---|---|---|---|---|
| `ROCm-Reach-SO101` | 500 | 6 min | ~135k | −0.00 → 9.8 | 360 (full) | fingertip within ~1 cm on average |
| `ROCm-Lift-Cube-SO101` | 1500 | 35 min | ~71k | 0.03 → 182 | 180 (full) | grasps and lifts in 64/64 test envs, cube held 96% of the episode, ~4 cm from the goal (below) |
| `IsaacContrib-Velocity-Flat-MicroDuck` | 300 | 12 min | ~42k | −0.2 → 106 | 24 → 956 of 1000 | walks; 23% of episodes still end in a fall |
| `ROCm-HeadPose-ReachyMini` | 400 | 14 min | ~46k | −0.2 → 5.1 | 400 (full) | head orientation error 0.54 → 0.03 rad, head centre held within ~1 cm |

**Assets.** The SO-101 and MicroDuck USDs are NVIDIA-hosted and download on first use (no Kit
needed). Reachy Mini is converted locally: `tools/convert_assets.py` fetches the official MJCF from
[pollen-robotics/reachy_mini](https://github.com/pollen-robotics/reachy_mini) at a pinned commit
(needs `git-lfs`), converts it with Isaac Lab's kit-less MJCF importer into `assets_generated/`,
and checks it with Newton and MuJoCo on the CPU. The importers (`requirements/importers.txt`,
from pypi.nvidia.com) are installed into the main venv by `scripts/install.sh`; they change no
package of the validated environment. `uv pip check` lists their exact pins on MuJoCo,
usd-exchange and usd-optimize, which Isaac Lab overrides as it does for torch.

**Licenses.** The Reachy Mini hardware files (meshes) are CC BY-NC-SA (the reachy_mini code is
Apache-2.0); [microduck_rl](https://github.com/pollen-robotics/microduck_rl) now licenses its
MicroDuck meshes CC BY-NC-SA 4.0 as well (`LICENSE-MODELS.md`), while the `ATTRIBUTION.txt` next to
NVIDIA's hosted MicroDuck USD cites an older Apache-2.0 revision. Treat both robots' geometry as
non-commercial and do not redistribute converted files: this repository commits no robot meshes or
USDs, and `assets_generated/` is gitignored. The SO-101 task rewards follow
[MuammerBay/isaac_so_arm101](https://github.com/MuammerBay/isaac_so_arm101) (BSD-3-Clause).

**Known limitations.**

- SO-101: the USD has no tool frame, so rewards use fixed offsets from the `gripper` link (fingertip
  for Reach, between the jaws for Lift). The command markers and the `Metrics/*/position_error`
  values refer to the `gripper` link origin, about 0.1 m behind the fingertip. Orientation is not
  commanded (5-DOF arm).
- Lift: the policy grasps the cube between the jaws and carries it towards the goal, but goal
  tracking is coarse (fine-grained term 2.5 of 5). Per-term episode rewards, iteration 0 → 1499:
  reaching 0.003 → 0.67 (max 1), lifting 0 → 14.1 (max 15), goal tracking 0 → 13.5 (max 16), fine
  goal tracking 0 → 2.5 (max 5). Headless evaluation of `model_1499.pt` (64 envs × 3 episodes,
  `extensions/isaaclab_rocm_robots/scripts/so101_eval_lift.py`): the cube is lifted in 100% of
  episodes and above the 25 mm threshold for 96% of the steps, still held at the end in 99.5%; in
  the last second the cube is 38 mm from the goal on average (the goal is resampled at 5 s of the
  6 s episode). The jaw closes to ~0.05 rad on the cube, its centre 13 mm from the grasp point.
  What made it learn (the earlier 600/1000-iteration runs never lifted): the arm now starts, and
  zero actions hold it, with the jaws pointing down 0.1 m above the cube (before, the gripper started
  horizontal 0.24 m up and the policy had to turn the wrist by 1.4 rad first); the grasp point used
  by the reaching reward moved from 7 cm to 8 cm below the `gripper` link, where a held cube actually
  sits (at 7 cm the reward peak was below the floor for the fingertips); the lift threshold dropped
  from 40 mm to 25 mm (12.5 mm off the floor; a cube resting on an edge reaches 17.7 mm); the
  action-rate/joint-velocity penalties ramp to −1e-2 instead of −1e-1, after ~830 iterations.
  Physics was checked first with a scripted grasp (`so101_scripted_lift.py`: IK to above the cube,
  descend, close, lift to 0.1 m): ~80% of 64 envs lift the cube, identically on CPU and GPU, with the
  SysID servo gains unchanged (the jaw servo saturates at ~2.5 N·m and holds the 30 g cube); misses
  are the single moving jaw pushing the cube away while closing. MuJoCo's elliptic friction cone
  with `impratio=10` (as in Isaac Lab's SO-101 stack task) made the scripted grasp fail in 8/8 envs
  (the open jaw pushed the cube into the floor), so the task keeps the pyramidal cone of the core
  lift preset. Training prints ~20 `linesearch iterations limit reached` warnings over 1500
  iterations, without visible effect.
- Reachy Mini is experimental: the head-pose command does not account for the real robot's
  workspace limits (some sampled poses are unreachable), the servo model is a plain PD with a
  0.5 N·m limit, and the passive joints have no friction.
- Not yet validated: `play.py` with `--viz viser` and the `-Play` ids (headless `play.py` runs the
  Lift checkpoint), the Rough MicroDuck task, and longer trainings than the runs above.

### Training speed options

Measured with `Isaac-Velocity-Flat-G1`, 4096 envs, RSL-RL (2026-10-30):

- **bfloat16 policy update** (recommended for longer runs): add `agent.algorithm.use_mixed_precision=True`
  to the `train.py` command line (an Isaac Lab/RSL-RL option, off by default). RDNA 3.5 has bfloat16
  matrix instructions but no TF32, so the PPO update runs at about twice the speed: learning time
  0.45 s -> 0.22 s per iteration, 41.0k -> 45.5k env steps/s. Over 300 iterations (seed 42) it learned
  as well as float32 (mean reward of the last 20 iterations 12.1 vs 10.3, episode length 996 vs 994,
  wall time 722 s vs 796 s). It changes floating-point results, so runs are not comparable bit for bit
  with float32 runs.
- Without effect, not adopted: `agent.torch_compile_mode=default` (same speed, 8 s more startup),
  `HIP_FORCE_DEV_KERNARG=1`, `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`,
  `TORCH_BLAS_PREFER_HIPBLASLT` (hipBLASLt is slower than the default rocBLAS here).
- PyTorch TunableOp (`PYTORCH_TUNABLEOP_ENABLED=1`) finds faster float32 GEMMs for the PPO MLPs (13.0 vs
  17.8 ms per update step in isolation), but Isaac Lab's `train.py` enables TF32 matmuls and TunableOp
  has no TF32 candidates on ROCm, so it changes nothing as shipped. With TF32 disabled (requires
  editing `isaaclab_rl/entrypoints/backends/train_rsl_rl.py`, which this repository leaves unmodified)
  it would reduce the learning time to 0.33 s per iteration (+5%); bfloat16 is faster still.

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
replays the segments as a host-driven program. Solvers still exit as soon as they converge. Each
evaluated condition costs one host round trip: a loop iteration with a trivial body takes about
8 us on Strix Halo, of which ~6 us are the graph launch and the host noticing the published value
(the GPU idles meanwhile). Measured on the workloads below, these round trips are 0.6-0.8% of the
time of MuJoCo-Warp and G1 training and ~8% of Kamino's DR Legs test.

Two alternatives were measured and not adopted. Predicated unrolling (capturing several loop
iterations whose kernels return early once the condition is false, so the host checks only every
K iterations) needs a guard argument in every generated kernel and guarded replacements for
memsets and copies, and saves at most the round trips themselves; running a loop to its maximum
iteration count without any host check would let the host run ahead, but Isaac Lab's MuJoCo-Warp
solver allows 100 iterations and converges in ~7, so the no-op iterations would cost more than the
round trips.

**Graph-captured allocations.** HIP graph memory nodes can be invalidated when another graph
is captured (CUDA guarantees they stay valid for the graph's lifetime). On HIP, arrays
allocated during a Warp capture are regular device allocations owned by the captured graph
instead (`warp.config.hip_graph_retained_allocations`).

**ROCm memory pool bug.** ROCm 7.13's "VM heap" stream-ordered memory pool can zero-fill a live
allocation after freed pool memory is reused and a host copy of ~20 KB or more touches the reused
range (`tools/repro_hip_vmheap_pool.py` reproduces it with plain HIP calls). Warp sets
`DEBUG_HIP_MEM_POOL_VMHEAP=0` before loading the HIP runtime, and `scripts/env.sh` exports it so it
also applies when PyTorch initializes HIP first.

**ROCm queue wedge.** Relaunching a graph of ~600 or more nodes back to back without host
synchronization wedges the device queue on ROCm 7.13: the command processor stops reading it, no
kernel runs, and the host blocks in `hipGraphLaunch`. Warp waits for earlier graph launches on a
stream once more than `warp.config.hip_graph_max_inflight_nodes` (8192) nodes are queued. It
tracks completion with an event per 1024 queued nodes rather than per launch (an event record costs
~4.4 us of host time on ROCm, which sat on the critical path of emulated loops), and restarts the
count whenever an emulated conditional has read a condition, since the stream has drained then.

**Tile solvers without MathDx.** NVIDIA's cuSolverDx does not exist for AMD GPUs, so tile Cholesky
factorizations and triangular solves (`wp.tile_cholesky`, `wp.tile_cholesky_solve`,
`wp.tile_lower_solve`, ...) use Warp's portable fallback. Its single right-hand-side solve ran on
one thread, and it dominated the physics: MuJoCo-Warp's fused mass-matrix factor-and-solve kernel
took 39% of the GPU time in G1 training, Kamino's sparse Cholesky solve 82% in the DR Legs test. On
HIP the solve now runs on a whole wavefront (each lane owns rows, values are broadcast with
`v_readlane`), and the factorization's inner loops are unrolled so that several shared-memory loads
are in flight (same operations in the same order, bitwise identical factors). Per call: 2.86 ms ->
1.19 ms (MuJoCo-Warp, 4096 worlds) and 208 us -> 59 us (Kamino).

The factor-and-solve kernel was then limited by occupancy, not arithmetic: it gathers the
mass-matrix block with `wp.tile_load_indexed`, which staged the n*n gather indices in shared memory,
doubling the kernel's shared memory (15.2 KB for G1's 43x43 block) and halving the blocks per
compute unit. For 1D gathers each thread only needs the indices of its own elements, so Warp now
keeps a register indices tile in registers there. MuJoCo-Warp also picks the block size of its dense
tile Cholesky kernels from the block size on HIP (one wave32 wavefront up to ~36 dofs, two up to
~60, four beyond; measured per size, explicitly configured `BlockDim` values are kept). G1
factor-and-solve: 1.19 ms -> 0.57 ms per call, with bitwise identical results.

**Graph node count.** Every node of a captured graph costs ~2.4 us of dispatch gap on gfx1151,
independent of its work. ROCm implements `hipMemsetAsync` with up to three fill kernels (head,
16-byte aligned body, tail), so on HIP Warp sets memory with its own kernel in a single dispatch.
Kamino's APGD loop zeroes its operator temporaries in one launch instead of three and resets its
loop conditions inside the preceding kernel instead of with separate memsets. DR Legs: 722k -> 493k
dispatches per 30 steps.

**Graph-capturable sorts.** With many segments, rocPRIM's segmented radix sort reads partition
counts back to the host, which a graph capture rejects; Warp sorts without partitioning while
capturing on HIP.

## Physics validation

MuJoCo-Warp's test suite (compared against MuJoCo C 3.15.0) and Newton's test suite were run on
gfx1151. Everything Isaac Lab's locomotion tasks use passes: the MuJoCo-Warp solver, articulations
and kinematics, joint limits, drives and actuators, contacts and friction, USD/URDF/MJCF import,
and contact, ray-cast and IMU sensors.

## Status and known issues

- Performance: with 4096 envs (RSL-RL, median env steps/s over iterations 2-11), at the
  `stable-2026-10-29` tags -> after the tile-solver and launch-budget changes (2026-10-30) -> after the
  shared-memory, block-size and graph-node changes (2026-10-30, later):

  | Benchmark | stable-2026-10-29 | tile solvers | current |
  |---|---|---|---|
  | `Isaac-Velocity-Flat-G1` | 29.8k | 37.6k | 41.2k |
  | `IsaacContrib-Velocity-Flat-MicroDuck` | 42.9k | 47.7k | 49.9k |
  | `ROCm-Lift-Cube-SO101` | 82.6k | 83.8k | 84.6k |
  | MuJoCo-Warp `testspeed` humanoid, 8192 worlds (steps/s) | 808k | 884k | 927k |
  | Kamino DR Legs dynamics test | 297 s | 143 s | 127 s |

  Older numbers in `logs/` were measured with 1024 envs while the GPU was shared with other work.
  See "Training speed options" below for RSL-RL settings that speed up the policy update.
- The OpenGL viewer is untested on ROCm; train headless, or watch training and trained policies
  in the browser with the viser visualizer (`--viz viser`, serves `http://localhost:8080`).
- AMD GPUs have no independent thread scheduling, so GPU code where one lane waits on another
  (spin locks) deadlocks, and on gfx1151 the driver often cannot recover from the hang (the
  desktop freezes until a reboot). Newton's viewer picking used such a lock; it is now lock-free.
- HIP has no conditional graph nodes, so Warp emulates `wp.capture_while`/`wp.capture_if` with
  host-driven segments. Solvers that iterate inside captured graphs pay for one host round trip
  (~8 us) per condition, and the host waits inside `wp.capture_launch` until the last loop has
  finished, so it cannot queue the next work ahead (see "Conditional graphs").
- Kamino's DR Legs dynamics test (2 min) still spends ~39% of the GPU's time idle between its many
  small graph nodes (~2.4 us each on ROCm) and in condition round trips; its bilateral sparse
  Cholesky solve (54 us per call, latency-bound on one world) is most of the busy time.
- Exit status: PyTorch's Kineto registers a rocprofiler-sdk tool whose `atexit` handler calls
  `_exit(0)`, so with the PyTorch ROCm nightly every Python process that used the GPU exits with
  status 0, even after an exception. `scripts/env.sh` sets `ROCPROFILER_REGISTER_ENABLED=0`, which
  restores exit codes (`torch.profiler` still works, without rocprofiler GPU traces). Run
  training and tests from a shell that sourced `env.sh`, or scripts cannot detect failures.
- Warp's kernel cache key includes a digest of the native headers (`warp/native/*.h`, which kernels
  include when they are compiled at runtime) and of the native libraries, so editing a header such as
  `tile_solve.h` no longer serves stale kernels (upstream Warp only keys the cache by version).
- `scripts/build_warp.sh` clears Warp's kernel cache (`~/.cache/warp`) at the end of the build.
  Do not rebuild while Warp programs are running; set `WARP_CACHE_PATH` to give an install its own cache.
- Warp tests need the LFS assets. The fork does not host them; install `git-lfs` and fetch them
  from AMD's repository:
  `git -C third_party/warp-rocm config lfs.url https://github.com/ROCm/warp.git/info/lfs && git -C third_party/warp-rocm lfs pull`
