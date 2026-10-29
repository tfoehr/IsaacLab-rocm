"""Registers the ROCm-* gym ids (loaded through the ``isaaclab.tasks`` entry point).

MicroDuck needs no wrapper: use Isaac Lab's ``IsaacContrib-Velocity-Flat-MicroDuck`` and
``IsaacContrib-Velocity-Rough-MicroDuck`` directly.
"""

from . import reachy_mini_head, so101_lift, so101_reach  # noqa: F401
