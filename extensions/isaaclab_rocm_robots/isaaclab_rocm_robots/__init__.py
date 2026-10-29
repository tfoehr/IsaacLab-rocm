"""Robot assets and RL tasks for experiments on IsaacLab-rocm (Newton + MuJoCo-Warp on ROCm).

Importing :mod:`isaaclab_rocm_robots.tasks` registers the ``ROCm-*`` gym ids. Isaac Lab's
``train.py``/``play.py`` do that automatically through the ``isaaclab.tasks`` entry point.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
"""Top-level IsaacLab-rocm checkout (the package lives in ``extensions/isaaclab_rocm_robots``)."""

GENERATED_ASSETS_DIR = Path(os.environ.get("ISAACLAB_ROCM_ASSETS_DIR", REPO_ROOT / "assets_generated"))
"""Where ``tools/convert_assets.py`` writes converted USDs (gitignored; licenses forbid committing them)."""
