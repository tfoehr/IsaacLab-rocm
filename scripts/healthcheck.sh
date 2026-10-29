#!/usr/bin/env bash
# Quick end-to-end health check of the IsaacLab-rocm install (~5 min on Strix Halo).
#   scripts/healthcheck.sh            # from any directory
# Checks: Warp sees the GPU and shares one HIP runtime with PyTorch, a core set of Warp test
# modules, MuJoCo-Warp and Newton (MuJoCo solver) sanity runs, short Isaac Lab trainings
# (Cartpole, Flat G1) with finite rewards, and the kernel log for amdgpu faults.
# Logs: logs/healthcheck/<timestamp>/. Exit status 0 only if every check passes.
# Run it before and after upgrading ROCm, PyTorch or the kernel. Runs GPU jobs one at a time.
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$root/scripts/env.sh"
py="$root/.venv/bin/python"

stamp="$(date +%Y-%m-%d_%H-%M-%S)"
start_wall="$(date '+%Y-%m-%d %H:%M:%S')"
start_mono="$(cut -d' ' -f1 /proc/uptime)"
out="$root/logs/healthcheck/$stamp"
mkdir -p "$out"
t_all=$SECONDS

names=() results=() times=()
record() {  # name status seconds
    names+=("$1"); results+=("$2"); times+=("$3")
    printf '%-28s %-4s %4ss\n' "$1" "$2" "$3"
}

# run_step NAME TIMEOUT_SEC CMD...   (output to $out/NAME.log, cwd $out)
run_step() {
    local name="$1" tmo="$2"; shift 2
    local t0=$SECONDS rc
    (cd "$out" && timeout --kill-after=30 "$tmo" "$@") >"$out/$name.log" 2>&1
    rc=$?
    [ $rc -eq 124 ] && echo "healthcheck: timed out after ${tmo}s" >>"$out/$name.log"
    echo "healthcheck: exit code $rc" >>"$out/$name.log"
    return $rc
}

# check_train NAME LOG ITERATIONS: rc already checked; require the last iteration and finite rewards.
check_train() {
    local log="$1" iters="$2"
    grep -q "Learning iteration $((iters - 1))/$iters" "$log" || { echo "missing final iteration" >>"$log"; return 1; }
    "$py" - "$log" <<'EOF'
import math, re, sys
text = open(sys.argv[1], errors="replace").read()
vals = re.findall(r"Mean reward:\s*(\S+)", text)
print(f"healthcheck: {len(vals)} mean-reward values, last {vals[-1] if vals else None}")
ok = bool(vals) and all(math.isfinite(float(v)) for v in vals)
sys.exit(0 if ok else 1)
EOF
}

echo "IsaacLab-rocm health check, logs in $out"

# 0. No other GPU training/test job may run at the same time.
others="$(pgrep -a python | grep -E 'train\.py|play\.py|run_warp_tests|pytest|unittest|newton\.examples' || true)"
if [ -n "$others" ]; then
    echo "$others" >"$out/busy.log"
    echo "Other GPU jobs are running (see $out/busy.log); not starting." >&2
    record "gpu-idle" FAIL 0
    echo "RESULT: FAIL"; exit 2
fi

# 1. Device and shared HIP runtime. Warp kernels need a source file, so write the check to one.
cat >"$out/device_check.py" <<'EOF'
import os, torch, warp as wp
wp.init()
assert torch.cuda.is_available() and torch.version.hip, "PyTorch has no ROCm device"
dev = wp.get_device("cuda:0")
print("torch", torch.__version__, "hip", torch.version.hip, "| warp", wp.__version__, dev.name, dev.arch_str)
assert dev.is_hip, "Warp device is not HIP"
assert dev.arch_str.startswith("gfx1151"), f"unexpected arch {dev.arch_str}"
libs = {l.split()[-1] for l in open("/proc/self/maps") if "libamdhip64" in l}
print("libamdhip64 mapped:", sorted(libs))
assert len({os.path.realpath(p) for p in libs}) == 1, "more than one HIP runtime loaded"


@wp.kernel
def axpy(a: float, x: wp.array(dtype=float), y: wp.array(dtype=float)):
    i = wp.tid()
    y[i] = a * x[i] + y[i]


x = torch.arange(1 << 16, device="cuda:0", dtype=torch.float32)
y = torch.ones_like(x)
wp.launch(axpy, dim=x.numel(), inputs=[2.0, wp.from_torch(x), wp.from_torch(y)], device="cuda:0")
wp.synchronize_device("cuda:0")
assert torch.equal(y, 2.0 * x + 1.0), "zero-copy torch/warp interop mismatch"
print("DEVICE-CHECK-OK")
EOF
# The OK marker guards against a nonzero exit getting lost in HIP runtime teardown.
t0=$SECONDS
exit_code_ok() {  # a process that used the GPU must report its exit status (see env.sh)
    "$py" -c 'import sys, torch; torch.zeros(1, device="cuda:0"); sys.exit(3)'
    local rc=$?; echo "exit-code probe: expected 3, got $rc"; [ $rc -eq 3 ]
}
if run_step device 300 "$py" "$out/device_check.py" && grep -qx 'DEVICE-CHECK-OK' "$out/device.log" \
   && exit_code_ok >>"$out/device.log" 2>&1
then record device PASS $((SECONDS - t0)); else record device FAIL $((SECONDS - t0)); fi

# 2. Core Warp test modules (all passed in the 2026-10-29 full suite), one process at a time.
warp_modules=(test_codegen test_func test_atomic test_launch test_graph test_utils test_array_reduce
              interop.test_torch interop.test_dlpack cuda.test_conditional_captures
              cuda.test_hip_conditional_emulation geometry.test_bvh geometry.test_mesh
              tile.test_tile_matmul tile.test_tile_reduce tile.test_tile_sort)
t0=$SECONDS
if run_step warp-tests 900 "$py" "$root/tools/run_warp_tests.py" -j 1 --timeout 300 \
        --out "$out/warp-tests" --modules "${warp_modules[@]}"
then record warp-tests PASS $((SECONDS - t0)); else record warp-tests FAIL $((SECONDS - t0)); fi

# 3. MuJoCo-Warp unit tests (kinematics, solver) and a Newton G1 example on the MuJoCo solver.
t0=$SECONDS
if run_step mujoco-warp 600 env -C "$root/third_party/mujoco_warp" "$py" -m pytest -q -x -p no:cacheprovider \
        mujoco_warp/_src/smooth_test.py mujoco_warp/_src/solver_test.py
then record mujoco-warp PASS $((SECONDS - t0)); else record mujoco-warp FAIL $((SECONDS - t0)); fi

t0=$SECONDS
if run_step newton-g1 600 "$py" -m newton.examples robot_g1 --viewer null --num-frames 200 --test --quiet
then record newton-g1 PASS $((SECONDS - t0)); else record newton-g1 FAIL $((SECONDS - t0)); fi

# 4. Short Isaac Lab trainings (rsl_rl, headless). Run dirs go under $out/logs/rsl_rl.
train="$root/isaaclab/scripts/reinforcement_learning/train.py"
t0=$SECONDS
if run_step train-cartpole 600 "$py" "$train" --rl_library rsl_rl --task Isaac-Cartpole \
        --num_envs 256 --max_iterations 20 --device cuda:0 --seed 1 \
   && check_train "$out/train-cartpole.log" 20 >>"$out/train-cartpole.log" 2>&1
then record train-cartpole PASS $((SECONDS - t0)); else record train-cartpole FAIL $((SECONDS - t0)); fi

t0=$SECONDS
if run_step train-g1-flat 900 "$py" "$train" --rl_library rsl_rl --task Isaac-Velocity-Flat-G1 \
        --num_envs 512 --max_iterations 5 --device cuda:0 --seed 1 \
   && check_train "$out/train-g1-flat.log" 5 >>"$out/train-g1-flat.log" 2>&1
then record train-g1-flat PASS $((SECONDS - t0)); else record train-g1-flat FAIL $((SECONDS - t0)); fi

# 5. Kernel log: amdgpu ring timeouts, resets and faults since the start of this script. Boot
# messages can carry skewed wall-clock times, so also filter on the monotonic timestamp.
t0=$SECONDS
timeout 10 journalctl -k -b --since "$start_wall" -o short-monotonic --no-pager >"$out/kernel.log" 2>&1
jrc=$?
faults="$(awk -v t0="$start_mono" '{ ts = $1; gsub(/[][]/, "", ts); if (ts + 0 >= t0 + 0) print }' "$out/kernel.log" \
    | grep -iE 'amdgpu|drm|kfd' \
    | grep -iE 'ring .*timeout|timed out|job_timedout|gpu reset|reset (begin|succeeded|failed)|gpu recovery|page fault|protection fault|vm fault|mes failed|hang|wedged' || true)"
if [ $jrc -ne 0 ]; then
    echo "journalctl failed (rc=$jrc)" >>"$out/kernel.log"; record kernel-log FAIL $((SECONDS - t0))
elif [ -n "$faults" ]; then
    echo "$faults" >"$out/kernel-faults.log"; record kernel-log FAIL $((SECONDS - t0))
    echo "$faults" | head -20
else
    record kernel-log PASS $((SECONDS - t0))
fi

# Summary
failed=0
for r in "${results[@]}"; do [ "$r" = PASS ] || failed=$((failed + 1)); done
{
    echo
    echo "IsaacLab-rocm health check $stamp ($root)"
    for i in "${!names[@]}"; do
        printf '%-28s %-4s %4ss\n' "${names[$i]}" "${results[$i]}" "${times[$i]}"
    done
    echo "total $((SECONDS - t_all))s"
    if [ $failed -eq 0 ]; then echo "RESULT: PASS"; else echo "RESULT: FAIL ($failed checks failed, logs in $out)"; fi
} | tee "$out/summary.txt"
[ $failed -eq 0 ]
