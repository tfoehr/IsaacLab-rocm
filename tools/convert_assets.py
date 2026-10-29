#!/usr/bin/env python
"""Regenerate the locally converted robot USDs used by isaaclab_rocm_robots (CPU only).

    source scripts/env.sh
    python tools/convert_assets.py              # fetch + convert + check every asset
    python tools/convert_assets.py --check      # only check the existing USDs

Assets:

* ``reachy_mini``: Pollen Robotics' official Reachy Mini MJCF (github.com/pollen-robotics/reachy_mini,
  pinned commit below) -> ``assets_generated/reachy_mini/reachy_mini.usda`` with Isaac Lab's
  kit-less MJCF converter (fixed base). The hardware files are CC BY-NC-SA: the source checkout
  and the USD stay in the gitignored ``assets_generated/`` and must not be committed.

The SO-101 and MicroDuck USDs need no conversion: Isaac Lab downloads NVIDIA's hosted USDs on
first use.

The importers come from ``requirements/importers.txt`` (scripts/install.sh installs them). To use
a separate venv instead, create one that sees the main venv's packages and run this script with
its python; nothing here touches the GPU.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# CPU only: hide the GPU from HIP before anything can initialize it. (ROCR_VISIBLE_DEVICES=-1
# breaks the converter, HIP_VISIBLE_DEVICES does not.)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = "-1"

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = Path(os.environ.get("ISAACLAB_ROCM_ASSETS_DIR", ROOT / "assets_generated"))

REACHY_REPO = "https://github.com/pollen-robotics/reachy_mini"
REACHY_COMMIT = "5feff22cbe9614709b8131f3de4209d0ac24452c"
REACHY_MJCF = "src/reachy_mini/descriptions/reachy_mini/mjcf/reachy_mini.xml"


def fetch_reachy_mini(src: Path) -> Path:
    """Sparse, shallow checkout of the pinned reachy_mini commit (MJCF and its meshes only).

    The meshes are Git LFS objects, so ``git-lfs`` must be installed.
    """
    mjcf_dir = "src/reachy_mini/descriptions/reachy_mini/mjcf"

    def git(*args):
        subprocess.run(["git", "-C", str(src), *args], check=True)

    if not (src / ".git").exists():
        src.mkdir(parents=True, exist_ok=True)
        git("init", "-q")
        git("remote", "add", "origin", REACHY_REPO)
        git("sparse-checkout", "set", mjcf_dir)
        git("fetch", "-q", "--depth", "1", "origin", REACHY_COMMIT)
        git("checkout", "-q", "FETCH_HEAD")
    # replace LFS pointer files by the meshes (no-op when they are already there)
    git("lfs", "install", "--local")
    git("lfs", "pull", "--include", f"{mjcf_dir}/**")
    return src / REACHY_MJCF


def convert_reachy_mini(out: Path, force: bool) -> Path:
    from isaaclab.utils.version import standalone_importers_available

    if not standalone_importers_available():
        sys.exit(
            "The kit-less importers are not installed. Install them with\n"
            "  uv pip install --python .venv/bin/python --index-url https://pypi.nvidia.com "
            "--extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match "
            "-r requirements/importers.txt"
        )
    from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg

    mjcf = fetch_reachy_mini(out / "src" / "reachy_mini")
    cfg = MjcfConverterCfg(asset_path=str(mjcf), usd_dir=str(out), fix_base=True, force_usd_conversion=force)
    usd = Path(MjcfConverter(cfg).usd_path)
    print(f"reachy_mini: {usd}")
    return usd


def check_reachy_mini(usd: Path) -> None:
    """Load the USD with Newton on the CPU, compare the head zero pose and step 1 s in MuJoCo C."""
    import numpy as np
    import warp as wp

    wp.config.log_level = getattr(wp, "LOG_WARNING", wp.config.log_level)
    import newton
    from pxr import Usd

    sys.path.insert(0, str(ROOT / "extensions" / "isaaclab_rocm_robots"))
    from isaaclab_rocm_robots.assets import reachy_mini as rm

    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.add_usd(Usd.Stage.Open(str(usd)), load_visual_shapes=False)
    model = builder.finalize(device="cpu")
    jt = model.joint_type.numpy()
    if (jt == newton.JointType.FREE).any():
        sys.exit("reachy_mini check: the base is floating (expected a fixed base)")
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    labels = list(model.body_label if hasattr(model, "body_label") else model.body_key)
    head = next(i for i, k in enumerate(labels) if k.endswith("/" + rm.REACHY_MINI_HEAD_BODY))
    center = next(i for i, k in enumerate(labels) if k.endswith(rm.REACHY_MINI_HEAD_BODY + "/Body"))
    bq = state.body_q.numpy()
    dpos = np.abs(bq[head][:3] - rm.REACHY_MINI_HEAD_ZERO_POS).max()
    q, q0 = bq[head][3:], np.array(rm.REACHY_MINI_HEAD_ZERO_QUAT)
    dquat = min(np.abs(q - q0).max(), np.abs(q + q0).max())
    dcen = np.abs(bq[center][:3] - rm.REACHY_MINI_HEAD_CENTER_ZERO_POS).max()
    print(f"  head zero pose: pos {bq[head][:3].round(4)} quat {q.round(4)} centre {bq[center][:3].round(4)}")
    if max(dpos, dquat, dcen) > 2e-3:
        sys.exit("reachy_mini check: head zero pose differs from isaaclab_rocm_robots/assets/reachy_mini.py")
    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=True)
    mj = solver.mj_model
    s0, s1, ctrl = model.state(), model.state(), model.control()
    q_init = s0.joint_q.numpy().copy()
    for _ in range(500):
        s0.clear_forces()
        solver.step(s0, s1, ctrl, None, 0.002)
        s0, s1 = s1, s0
    qn = s0.joint_q.numpy()
    print(f"  MuJoCo model: nq={mj.nq} nv={mj.nv} nu={mj.nu} neq={mj.neq}; 1 s passive: max |dq| {np.abs(qn - q_init).max():.2e}")
    if not np.isfinite(qn).all():
        sys.exit("reachy_mini check: simulation diverged")
    print("  OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("assets", nargs="*", choices=["reachy_mini"], help="default: all")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory (default: %(default)s)")
    parser.add_argument("--force", action="store_true", help="convert even if the USD is up to date")
    parser.add_argument("--check", action="store_true", help="only check existing USDs")
    args = parser.parse_args()
    for name in args.assets or ["reachy_mini"]:
        if name == "reachy_mini":
            usd = args.out / "reachy_mini" / "reachy_mini.usda"
            if not args.check:
                usd = convert_reachy_mini(args.out, args.force)
            check_reachy_mini(usd)


if __name__ == "__main__":
    main()
