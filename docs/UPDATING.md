# Updating and rolling back

How to pull in new ROCm/PyTorch nightlies and upstream changes of the components (Isaac Lab,
Newton, MuJoCo-Warp, Warp), validate the result, mark it as the new known-good state, and get
back to the previous one if something breaks.

The known-good states are the `stable-*` tags (`stable-2026-10-29`, `stable-2026-10-30`, ...),
which exist with the same name in every repository: this one, `third_party/warp-rocm`,
`third_party/newton`, `third_party/mujoco_warp` and (local only, Isaac Lab is not forked)
`isaaclab`. Each has a lock file `requirements/lock-<date>.txt` with the exact package set of its
venv. `scripts/stable.sh` handles the git side:

```bash
scripts/stable.sh status [--fetch]           # per repo: branch, HEAD, nearest stable tag, dirty/unpushed
scripts/stable.sh checkout <tag> [--top]     # detach all components at <tag> (refuses if a repo is dirty)
scripts/stable.sh branches [--top]           # back to the working branches named in components.toml
scripts/stable.sh tag <stable-YYYY-MM-DD>    # lock file + components.toml + README, same tag everywhere
scripts/stable.sh push <tag>                 # push branches and tag to the `tfoehr` remotes only
```

It uses git and `uv pip freeze` only and never touches the GPU. Destructive steps ask for
confirmation (`--yes` skips it).

Rules that apply throughout:

- **One GPU job at a time.** Do not build, test or train while an experiment is running on the GPU;
  `scripts/healthcheck.sh` refuses to start while a `train.py`/`play.py`/test process exists. A GPU
  hang on gfx1151 can take the desktop down: never run new upstream code that waits on another lane
  (spin locks, flags polled in a loop) on the GPU; see "Resolving conflicts" below.
- **Do not rebuild Warp while Warp programs are running.** `scripts/build_warp.sh` clears the
  kernel cache at the end.
- Update one thing at a time (ROCm, then each component), validate, and tag. A tag after each
  successful step gives a short way back.

## 1. Before updating

```bash
cd ~/workspace/IsaacLab-rocm
scripts/stable.sh status --fetch
```

Every repository should be `clean` (untracked files are fine), on its branch, at a `stable-*` tag
(not `stable-... +N`) and without `unpushed` commits. If a component is ahead of its last tag,
validate and tag that state first (section 5), or accept that a rollback goes back to the older
tag. Commit or stash local edits; `checkout` and `tag` refuse to run with uncommitted changes to
tracked files.

Keep a baseline to compare against. If `logs/` has no validation logs for the current tag, run the
validation ladder (section 4) on it now and keep the logs, e.g. in `logs/baseline-<tag>/`. Compare
failure *lists*, not just counts: a few Warp and MuJoCo-Warp tests fail on gfx1151 already.

Enable git's conflict-resolution cache once per repository; it replays resolutions when a merge is
redone:

```bash
for r in . third_party/warp-rocm third_party/newton third_party/mujoco_warp; do git -C $r config rerere.enabled true; done
```

### In place or side by side

- **In place** (simplest): update this checkout and its venv while the GPU is otherwise idle. The
  old venv is kept under another name for the rollback (section 2).
- **Side by side**: build the update in a second checkout made of git worktrees, while experiments
  keep running here. The worktrees share the repositories (branches and tags), so nothing needs to
  be fetched between them. GPU validation still has to wait until the GPU is free.

  ```bash
  next=~/workspace/IsaacLab-rocm-next; b=update/$(date +%F)
  git worktree add -b next "$next" main
  for p in third_party/warp-rocm third_party/newton third_party/mujoco_warp isaaclab; do
      GIT_LFS_SKIP_SMUDGE=1 git -C $p worktree add -b $b "$next/$p" HEAD
  done
  cd "$next"
  export WARP_CACHE_PATH="$next/.warp-cache"   # own kernel cache (build_warp.sh clears only this one); set it in every shell for $next
  # then the Setup steps of README.md from `uv venv` on (own .venv, own warp.so)
  ```

  The update branches are merged into the working branches only after validation (section 5).
  Remove the worktrees afterwards with `git worktree remove` (and `git branch -d` the update branches).

## 2. ROCm and PyTorch nightlies

ROCm comes only from the TheRock Python packages in the venv (`rocm[devel]`, pulled in with the
PyTorch wheels); `/opt/rocm` is not used. Torch, torchvision, triton and `rocm*` must come from the
same nightly (`+rocm7.13.0a20260411` suffix). Find newer builds:

```bash
idx=https://rocm.nightlies.amd.com/v2/gfx1151
curl -s $idx/torch/ | grep -oE 'torch-[^"#<]+-cp312-cp312-linux_x86_64\.whl' \
    | sed -E 's/%2B/+/; s/^.*(a|rc)(20[0-9]{6}).*$/\2 &/' | sort -u | tail -5
curl -s $idx/torchvision/ | grep -oE 'torchvision-[^"#<]+-cp312-cp312-linux_x86_64\.whl' | sed 's/%2B/+/' | sort -u | tail -3
```

Pick a torch build whose `+rocmX.Y.ZaDATE` has matching `torchvision` and `rocm` builds. Isaac Lab
itself targets torch 2.11; this port uses the 2.12 nightly because it is the one built for gfx1151.

In place, keeping the old venv (a venv only works at the path it was created at, so it is parked
under another name and moved back for a rollback):

```bash
old=stable-2026-10-30                                     # current tag
new=7.13.0a2026MMDD; tv=2.12.0a0; vv=0.27.0a0             # versions chosen above
lock=requirements/lock-${old#stable-}.txt
mv .venv .venv-$old
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
    --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match --prerelease allow \
    "torch==$tv+rocm$new" "torchvision==$vv+rocm$new" "rocm[devel]==$new"
.venv/bin/rocm-sdk init
sed -i -E "s/^(torch==).*/\1$tv+rocm$new/; s/^(torchvision==).*/\1$vv+rocm$new/" requirements/rocm-overrides.txt
scripts/build_warp.sh          # rebuild libwarp against the new SDK (it lives in the source tree, not the venv); clears the kernel cache
# every other package as before: the old lock file without the ROCm stack as constraints
mkdir -p logs && grep -vE '^(torch|torchvision|triton|rocm|rocm-sdk-[a-z0-9-]+)==' $lock > logs/constraints-$old.txt
scripts/install.sh --constraint logs/constraints-$old.txt
```

Also update the versions in the `uv pip install` line of README's Setup. Then validate
(section 4) and re-check the ROCm workarounds below. Do not run `uv cache clean`: the uv cache
holds the old wheels, in case a nightly is removed from the index.

### Re-check the ROCm workarounds

Each workaround stays until a new ROCm demonstrably fixes the bug and the full validation ladder
passes without it. Run these one at a time, with nothing else on the GPU, from a shell that sourced
`scripts/env.sh`:

| Workaround | Check on the new ROCm | Fixed when |
|---|---|---|
| `DEBUG_HIP_MEM_POOL_VMHEAP=0` (Warp and `env.sh`): VM-heap pool zero-fills live allocations | `for op in pageable pinned memset d2d d2h; do DEBUG_HIP_MEM_POOL_VMHEAP=1 python tools/repro_hip_vmheap_pool.py 64e6 8192 $op; done` (plain HIP calls) | every variant prints OK instead of CORRUPT; with `=0` it must still print OK |
| `ROCPROFILER_REGISTER_ENABLED=0` (`env.sh`): Kineto's rocprofiler tool turns every exit status into 0 | `env -u ROCPROFILER_REGISTER_ENABLED python -c 'import sys, torch; torch.zeros(1, device="cuda:0"); sys.exit(3)'; echo $?` | prints 3 (the health check's `device` step tests the same with the workaround set) |
| `warp.config.hip_graph_max_inflight_nodes = 8192`: relaunching a graph of ~600+ nodes back to back wedges the queue | Do not test casually: a wedged queue can need a reboot. Only if AMD's release notes mention a fix: save all work, then replay a ~1000-node graph 3000x with the limit set to `0`, under `timeout` | no hang; then keep the limit anyway unless it costs measurable time |
| Capture-safe segmented sort (Warp `sort.cu`): rocPRIM reads partition counts to the host during capture | Warp's `test_utils` (`python tools/run_warp_tests.py --modules test_utils`) covers the captured path; the workaround is cheap | keep it; revisit only if rocPRIM changes its segmented sort |
| Conditional-graph emulation, graph-retained allocations (`warp.config.hip_*`) | HIP has no conditional graph nodes yet; check the HIP release notes for `hipGraphConditionalHandleCreate` / conditional nodes | a ROCm with native conditional nodes would let Warp drop the emulation (a Warp change, not a switch) |

Also re-check the health check's `kernel-log` step: a new ROCm or linux-firmware must not produce
amdgpu ring timeouts or page faults.

## 3. Components

### Order

Isaac Lab decides the versions, so update it first and move the others to what it pins:

1. **Isaac Lab** (`isaaclab/pyproject.toml`): pins `newton[sim] @ git+...newton.git@<rev>` (in
   `dependencies` and `[tool.uv] override-dependencies`), `mujoco~=` / `mujoco-warp~=`, `warp-lang==`,
   `torch==` and the other overrides.
2. **Newton** at exactly the revision Isaac Lab pins. Its `pyproject.toml` requires `mujoco-warp~=`
   and `warp-lang>=`.
3. **MuJoCo-Warp** at the release Newton and Isaac Lab require (tag `vX.Y.Z`).
4. **Warp** only when Isaac Lab or Newton require a newer `warp-lang` (or for a specific fix): it
   is the largest merge.

Look first, then decide:

```bash
git -C isaaclab fetch origin
git -C isaaclab log --oneline HEAD..origin/main | wc -l
il=origin/main      # or the Isaac Lab commit/tag you want
git -C isaaclab diff HEAD $il -- pyproject.toml 'source/*/pyproject.toml' 'source/*/setup.py'
git -C isaaclab show $il:pyproject.toml | grep -nE 'newton.git@|mujoco|warp-lang|"torch'
git -C isaaclab diff --stat HEAD $il -- source | grep -v changelog | tail -3; ls isaaclab/source   # new packages?
```

### Merge, not rebase

The ROCm branches (`rocm-1.17-gfx1151`, `rocm-hip`) are updated by **merging** the new upstream
revision into them, on an update branch that is fast-forwarded onto the working branch after
validation:

- The branches are published (`tfoehr`) and the `stable-*` tags must stay on their history; a
  rebase rewrites every ROCm commit, so old tags would no longer be ancestors of the branch and
  `status`, `describe` and fork pulls would break.
- The history is already built from merges (`Merge Warp v1.17.0 into ROCm Warp ...`, one merge per
  fix branch), and AMD's fork syncs upstream the same way.
- Conflicts are resolved once, in one reviewable commit (`git show --remerge-diff <merge>`), instead
  of once per replayed commit.

To send a fix upstream, cherry-pick it from its fix branch (e.g. `fix/kamino`, `perf/...`) onto
upstream's main instead.

### Isaac Lab

Isaac Lab is used unmodified, so its local `rocm` branch only fast-forwards:

```bash
git -C isaaclab switch rocm && git -C isaaclab merge --ff-only $il
```

Then, in this repository:

- `requirements/rocm-overrides.txt`: mirror Isaac Lab's `override-dependencies` (except torch,
  torchvision and the Newton/MuJoCo-Warp/Warp lines, which point at the local checkouts), e.g.
  `mujoco~=`, `usd-exchange==`, `usd-optimize~=`, `numpy`, `fsspec`, `packaging`, `urllib3`.
- `scripts/install.sh`: add new packages under `isaaclab/source/` to `members` if they work without
  Kit.
- `requirements/importers.txt` if Isaac Lab's `importers` extra changed.
- `components.toml` `base` (or let `stable.sh tag` write it).
- `extensions/isaaclab_rocm_robots` may need changes for Isaac Lab API changes; the robot trainings
  in section 4 show it.

### Newton

```bash
cd third_party/newton
git fetch origin
new=$(git -C ../../isaaclab show HEAD:pyproject.toml | grep -oE 'newton\.git@[0-9a-f]{40}' | head -1 | cut -d@ -f2)
old=71eeba16a717d4dcf0b9dd7056a7b6f863054dc3    # components.toml base
git log --oneline $old..$new | wc -l
git switch -c update/newton-${new:0:8} rocm-hip
git merge --no-ff -m "Merge upstream Newton ${new:0:12} (Isaac Lab's pin)" $new
git show $new:pyproject.toml | grep -nE 'mujoco|warp-lang'   # what MuJoCo-Warp and Warp it needs
```

Where the ROCm changes are (`git diff --stat $old rocm-hip`): the Kamino solver
(`newton/_src/solvers/kamino/...`: tile Cholesky on HIP, APGD loop with fewer graph nodes), the
lock-free ray cast and viewer picking (`newton/_src/geometry/raycast.py`, `newton/_src/viewer/picking.py`),
support for devices without native conditional graph nodes, VBD's `nextafterf`, and test skips.
Set `base` in `components.toml` to `$new`.

### MuJoCo-Warp

```bash
cd third_party/mujoco_warp
git fetch origin --tags
git switch -c update/mjw-vX.Y.Z rocm-hip
git merge --no-ff -m "Merge MuJoCo-Warp vX.Y.Z" vX.Y.Z
```

The ROCm changes are small (`git diff --stat v3.15.0 rocm-hip`): `warp_util.py` (`tile_block_dim`,
per-size HIP block dims), `smooth.py`, `solver.py`, `collision_driver.py`, `collision_flex.py`,
`derivative.py` (loops without graph conditionals, HIP block sizes). Update `base` in
`components.toml` and `mujoco~=` in `requirements/rocm-overrides.txt` to the MuJoCo release it
requires.

### Warp

Remotes in `third_party/warp-rocm`: `upstream` = NVIDIA/warp (release tags `vX.Y.Z`), `origin` =
ROCm/warp (AMD's fork, branch `amd-integration`), `tfoehr` = the published fork. Use a new branch
per Warp minor version:

```bash
cd third_party/warp-rocm
git fetch upstream --tags && git fetch origin
git log --oneline rocm-1.17-gfx1151..origin/amd-integration    # new AMD commits? merge them first, same way
git switch -c rocm-1.18-gfx1151 rocm-1.17-gfx1151
git merge --no-ff -m "Merge Warp v1.18.0 into the ROCm port" v1.18.0
# version: VERSION.md and warp/config.py -> 1.18.0+rocm.1
```

The ROCm delta is large (`git diff --stat v1.17.0 rocm-1.17-gfx1151`: ~150 files). Most conflicts
will be in `warp/_src/context.py` (HIP device setup, graph capture and conditional emulation,
launch budget), `warp/native/warp.cu` and `cuda_util.cpp` (HIP runtime calls, memsets, events),
`hip_util.h`, `bvh.cu`, `sort.cu`, the tile headers (`tile.h`, `tile_reduce.h`, `tile_solve.h`:
wave32/wave64, wavefront solves), `builtin.h`, `warp/_src/build.py`/`build_dll.py` (hipRTC, cache
key) and `build_lib.py`. Set `branch` in `components.toml` to the new branch.

### Resolving conflicts

- Look at both sides separately: `git diff <old base> <new upstream> -- <file>` (upstream's change)
  and `git diff <old base> ORIG_HEAD -- <file>` with `git log --oneline <old base>..ORIG_HEAD -- <file>`
  (the ROCm change and the commit that explains it).
- Take upstream's version of the shared code and re-apply the HIP part on top (`is_hip` branches,
  `__HIP_PLATFORM_AMD__`/`WP_ENABLE_HIP` blocks, wavefront-size handling). Never drop a HIP block
  to make a conflict go away; if upstream rewrote the code, port the fix.
- After the merge, `git diff --stat <new upstream> HEAD` is the ROCm delta: it should be about as large
  as before. `git show --remerge-diff HEAD` shows how the conflicts were resolved.
- Review new upstream GPU code for things that do not work on AMD before running it:
  - one lane or block waiting for another's write (spin locks, polling loops): AMD GPUs have no
    independent thread scheduling, so this deadlocks and can take the desktop with it. Look at it
    on the CPU (`device="cpu"`) and replace it with lock-free atomics before the first GPU run;
  - hard-coded warp size 32 or 32-bit lane masks (CDNA has wave64);
  - new uses of CUDA graph features: conditional nodes outside `wp.capture_while`/`wp.capture_if`,
    graph memory nodes, `hipMemsetAsync` in hot loops (each adds graph nodes);
  - CUDA-only libraries (MathDx/cuBLASDx/cuSolverDx, thread block clusters).

### Rebuild and caches

| Changed | Do |
|---|---|
| Warp (any change, native or Python) | `scripts/build_warp.sh` (rebuilds `warp.so`, clears the kernel cache) |
| ROCm SDK / torch | `scripts/build_warp.sh`, then `scripts/install.sh` |
| Newton, MuJoCo-Warp (Python only, editable installs) | nothing to build; `scripts/install.sh` if their dependencies changed |
| Isaac Lab, overrides, importers | `scripts/install.sh` |

The kernel cache key includes the Warp version, the native headers and libraries, and each module's
source, so stale kernels are not served; clearing after a Warp update is still the safe default.
By hand (with no Warp program running): `rm -rf "${WARP_CACHE_PATH:-$HOME/.cache/warp}"`. The first
runs after clearing are slower (cold cache: the health check takes ~7 instead of ~5 minutes).

## 4. Validation ladder

From a shell that sourced `scripts/env.sh`, one step at a time. Stop at the first step that fails
and compare with the baseline before going on.

| # | Step | Command | Expected (stable-2026-10-30 unless noted) |
|---|---|---|---|
| 0 | Packages (CPU) | `uv pip check --python .venv/bin/python` | the 9 known incompatibilities: `isaaclab-dev` wants torch 2.11/torchvision 0.26; `isaacsim-asset-isolated`/`isaacsim-robot-schema` pin older `usd-exchange`, `usd-optimize`, `mujoco`, `newton-usd-schemas`, converters |
| 1 | Health check | `scripts/healthcheck.sh` | `RESULT: PASS`, ~5 min (~7 min with a cold kernel cache) |
| 2 | Warp suite | `python tools/run_warp_tests.py -j 6 --out test_results/warp-<tag>` | 2026-10-29: 197 modules, 7539 tests, 12 failed, 117 skipped, 2 crashed modules (`cuda.test_tile_rocwmma`, CDNA only; `interop.test_paddle`); no new failures |
| 3 | MuJoCo-Warp | `cd third_party/mujoco_warp && python -m pytest -q -n 4 -p no:cacheprovider mujoco_warp` | same failures as the baseline |
| 4 | Newton | `cd third_party/newton && python -m newton.tests -j 4 --parallel-timeout 3600` (or `-p 'test_mujoco*.py'`, `-p 'test_import_*.py'`, ... per area) | same failures as the baseline; `test_collision_pipeline` has hung intermittently, use the timeout |
| 5 | Kamino | `python -m newton.tests -j 4 -p 'test_kamino*.py'` and `python -m newton.tests -p test_examples.py -k kamino` | 681 tests pass; examples pass (G1, H1, ANYmal match the CPU) |
| 6 | Trainings | below | numbers within ~10% of the table |
| 7 | Speed | below | within ~5% of the table, or explained |

Trainings (`cd isaaclab`, 4096 envs, RSL-RL, `--device cuda:0`):

```bash
python scripts/reinforcement_learning/train.py --rl_library rsl_rl --task ROCm-Reach-SO101 --num_envs 4096 --max_iterations 500 --device cuda:0 2>&1 | tee ../logs/upd-reach.log
python scripts/reinforcement_learning/train.py --rl_library rsl_rl --task IsaacContrib-Velocity-Flat-MicroDuck --num_envs 4096 --max_iterations 300 --device cuda:0 2>&1 | tee ../logs/upd-duck.log
python scripts/reinforcement_learning/train.py --rl_library rsl_rl --task Isaac-Velocity-Flat-G1 --num_envs 4096 --max_iterations 300 --seed 42 --device cuda:0 2>&1 | tee ../logs/upd-g1.log
# optional, longer: ROCm-Lift-Cube-SO101 1500 it (35 min), ROCm-HeadPose-ReachyMini 400 it (14 min),
# Isaac-Velocity-Flat-G1 1500 it (83 min, reward 28.9)
```

| Task | Iterations | Wall time | Env steps/s | Mean reward (start -> end) | Episode length |
|---|---|---|---|---|---|
| `ROCm-Reach-SO101` | 500 | 6 min | ~135k | -0.00 -> 9.8 | 360 (full) |
| `IsaacContrib-Velocity-Flat-MicroDuck` | 300 | 12 min | ~42k | -0.2 -> 106 | 24 -> 956 of 1000 |
| `Isaac-Velocity-Flat-G1` | 300 | ~13 min | ~41k | last 20 iterations ~10 (seed 42) | ~994 |
| `ROCm-Lift-Cube-SO101` | 1500 | 35 min | ~71k | 0.03 -> 182 | 180 (full) |
| `ROCm-HeadPose-ReachyMini` | 400 | 14 min | ~46k | -0.2 -> 5.1 | 400 (full) |

The steps/s of the robot rows were measured before the 2026-10-30 speedups (now ~50k for
MicroDuck, ~85k for Lift); use the speed numbers below for performance. Training is not
deterministic across ROCm/PyTorch versions; judge the reward curve's shape and end
value, not exact numbers. Summarize a log (CPU only):

```bash
python3 - logs/upd-g1.log <<'EOF'
import re, statistics, sys
text = re.sub(r"\x1b\[[0-9;]*m", "", open(sys.argv[1], errors="replace").read())
sps = [int(x) for x in re.findall(r"Steps per second:\s*(\d+)", text)]
rew = re.findall(r"Mean reward:\s*(\S+)", text)
print(f"iterations {len(sps)}, median steps/s (it 2-11) {statistics.median(sps[2:12]):.0f}, mean reward first/last {rew[0]} / {rew[-1]}")
EOF
```

Speed (the README's performance table; median env steps/s over iterations 2-11 with 4096 envs):
`Isaac-Velocity-Flat-G1` 41.2k, `IsaacContrib-Velocity-Flat-MicroDuck` 49.9k, `ROCm-Lift-Cube-SO101`
84.6k; MuJoCo-Warp `cd third_party/mujoco_warp && mjwarp-testspeed benchmarks/humanoid/humanoid.xml --nworld 8192`
927k steps/s; Kamino's DR Legs dynamics test 127 s (time `python -m newton.tests -p 'test_kamino_solvers_dvi*.py' -k dr_legs`
on the baseline and on the update).

## 5. Marking a new stable state

After the ladder passed:

1. Side by side only: promote the update branches in this checkout (with the GPU idle):

   ```bash
   b=update/2026-MM-DD
   git -C third_party/newton switch rocm-hip && git -C third_party/newton merge --ff-only $b   # same for mujoco_warp
   git -C third_party/warp-rocm switch rocm-1.18-gfx1151                                      # a new Warp branch: just switch
   git -C isaaclab switch rocm && git -C isaaclab merge --ff-only $b
   git merge --ff-only next
   ```

   then repeat the venv/build/install steps here (section 2, or `scripts/build_warp.sh` and
   `scripts/install.sh`) and run `scripts/healthcheck.sh`.
2. Commit everything that belongs to the new state (overrides, `components.toml` branch/base,
   README setup versions and validated numbers). `stable.sh tag` needs all repositories clean.
3. Tag:

   ```bash
   scripts/stable.sh tag stable-2026-MM-DD -m "Known-good IsaacLab-rocm state (<what changed>; health check PASS)"
   ```

   It writes `requirements/lock-2026-MM-DD.txt` (`uv pip freeze` with the same header and comment
   format as the earlier lock files; editable checkouts and `py-spy` become comments), sets
   `tag = ...` and Isaac Lab's `base` in `components.toml`, updates the README setup lines
   (`git clone -b <tag>`, `git -C isaaclab checkout <commit>`, lock file name), shows the diff, asks,
   then commits these three files in this repository and creates the annotated tag in all five
   repositories (Isaac Lab's tag is local). Check the README prose (validated numbers, dates) by hand.
4. Publish (to the `tfoehr` remotes only; Isaac Lab is skipped):

   ```bash
   scripts/stable.sh push stable-2026-MM-DD    # or `tag ... --push`
   scripts/stable.sh status --fetch            # no `unpushed`, no `tags-not-on-tfoehr`
   ```

5. Once the new state has proven itself, delete the parked venv (`rm -rf .venv-stable-...`).

## 6. Rollback

**Components only** (venv and ROCm unchanged):

```bash
tag=stable-2026-10-30
scripts/stable.sh checkout $tag                                  # all components detached at $tag
scripts/build_warp.sh                                            # also clears the kernel cache
scripts/install.sh --constraint requirements/lock-${tag#stable-}.txt
scripts/healthcheck.sh
```

Isaac Lab has no `stable-*` tags from before 2026-11; `checkout` then uses the `base` commit in
`components.toml` at `$tag`. To also put this repository's scripts, overrides and robot tasks back,
add `--top` (detaches this repository too; `git checkout main` returns). A single file is enough
in most cases: `git checkout $tag -- requirements/rocm-overrides.txt`.

Back to the working branches when the problem is fixed: `scripts/stable.sh branches`, then rebuild
and reinstall the same way.

**ROCm/PyTorch** (after a nightly update), with the parked venv:

```bash
tag=stable-2026-10-30
rm -rf .venv && mv .venv-$tag .venv
git checkout $tag -- requirements/rocm-overrides.txt              # if it was changed and committed
scripts/build_warp.sh                                             # warp.so was built against the newer SDK
scripts/healthcheck.sh
```

Without a parked venv, rebuild it from the lock file:

```bash
tag=stable-2026-10-30; lock=requirements/lock-${tag#stable-}.txt
rm -rf .venv && uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
    --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match --prerelease allow \
    $(grep -E '^(torch|torchvision)==' $lock) "rocm[devel]==$(sed -n 's/^rocm==//p' $lock)"
.venv/bin/rocm-sdk init
git checkout $tag -- requirements/rocm-overrides.txt
scripts/build_warp.sh
scripts/install.sh --constraint $lock
scripts/healthcheck.sh
```

The lock file pins every package except the local checkouts, so `install.sh --constraint`
reproduces the validated venv exactly (`uv pip freeze` then matches the lock file's
non-comment lines). If the health check fails on a state that passed before, suspect what is
outside the lock file: the kernel and linux-firmware (`journalctl -k`, the `kernel-log` step), a
leftover `WARP_CACHE_PATH`, or another job on the GPU.
