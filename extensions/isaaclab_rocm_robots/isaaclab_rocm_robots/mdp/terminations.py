"""Termination terms that catch runaway or non-finite simulation states."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_out_of_bounds(
    env: ManagerBasedRLEnv,
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """True when the object's root leaves an axis-aligned box around the env origin, or is non-finite."""
    obj = env.scene[object_cfg.name]
    pos = obj.data.root_pos_w.torch - env.scene.env_origins
    lo = torch.tensor([b[0] for b in bounds], device=pos.device)
    hi = torch.tensor([b[1] for b in bounds], device=pos.device)
    inside = ((pos >= lo) & (pos <= hi)).all(dim=1)  # NaN compares False -> outside
    return ~inside


def joint_state_invalid(
    env: ManagerBasedRLEnv, max_joint_vel: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """True when any joint position/velocity is non-finite or a joint speed exceeds ``max_joint_vel`` [rad/s]."""
    asset = env.scene[asset_cfg.name]
    q = asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
    qd = asset.data.joint_vel.torch[:, asset_cfg.joint_ids]
    ok = torch.isfinite(q).all(dim=1) & (qd.abs() <= max_joint_vel).all(dim=1)  # NaN fails the comparison
    return ~ok
