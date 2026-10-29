"""Head pose command for Reachy Mini.

The command is a rotation (roll, pitch, yaw about the root frame axes) of the head away from its
zero pose, about the head centre. It is stored like :class:`UniformPoseCommand`'s command -- the
absolute target pose of the head body in the root frame -- so Isaac Lab's pose metrics, debug
markers and ``position_command_error``/``orientation_command_error`` rewards apply unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch

from isaaclab.envs.mdp import UniformPoseCommand, UniformPoseCommandCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_mul, quat_unique


class HeadPoseCommand(UniformPoseCommand):
    """Rotations of a body about a fixed point, relative to the body's zero pose."""

    cfg: HeadPoseCommandCfg

    def __init__(self, cfg: HeadPoseCommandCfg, env):
        super().__init__(cfg, env)
        dev = self.device
        self._zero_quat = torch.tensor(cfg.zero_quat, device=dev).repeat(self.num_envs, 1)
        self._center = torch.tensor(cfg.center_pos, device=dev).repeat(self.num_envs, 1)
        # body origin relative to the rotation centre at the zero pose (root frame)
        self._origin_from_center = torch.tensor(cfg.zero_pos, device=dev) - torch.tensor(cfg.center_pos, device=dev)
        self._origin_from_center = self._origin_from_center.repeat(self.num_envs, 1)
        self._resample_command(torch.arange(self.num_envs, device=dev))

    def _resample_command(self, env_ids: Sequence[int]):
        if not hasattr(self, "_zero_quat"):  # called by the base constructor before our buffers exist
            return
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        n = len(env_ids)
        r = self.cfg.ranges
        euler = torch.empty(n, 3, device=self.device)
        euler[:, 0].uniform_(*r.roll)
        euler[:, 1].uniform_(*r.pitch)
        euler[:, 2].uniform_(*r.yaw)
        if self.cfg.zero_command_prob > 0.0:
            euler[torch.rand(n, device=self.device) < self.cfg.zero_command_prob] = 0.0
        q_rel = quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])
        q_target = quat_unique(quat_mul(q_rel, self._zero_quat[env_ids]))
        self.pose_command_b[env_ids, :3] = self._center[env_ids] + quat_apply(q_rel, self._origin_from_center[env_ids])
        self.pose_command_b[env_ids, 3:] = q_target


@configclass
class HeadPoseCommandCfg(UniformPoseCommandCfg):
    """Configuration of :class:`HeadPoseCommand`; ``ranges.pos_*`` are unused."""

    class_type: type = HeadPoseCommand

    zero_pos: tuple[float, float, float] = MISSING
    """Body origin at the zero pose, root frame [m]."""

    zero_quat: tuple[float, float, float, float] = MISSING
    """Body orientation at the zero pose, root frame (x, y, z, w)."""

    center_pos: tuple[float, float, float] = MISSING
    """Point the commanded rotations turn about, root frame [m]."""

    zero_command_prob: float = 0.1
    """Share of resamples that command the zero pose (look straight ahead)."""

    ranges: UniformPoseCommandCfg.Ranges = UniformPoseCommandCfg.Ranges(
        pos_x=(0.0, 0.0), pos_y=(0.0, 0.0), pos_z=(0.0, 0.0), roll=(-0.4, 0.4), pitch=(-0.4, 0.4), yaw=(-1.0, 1.0)
    )
