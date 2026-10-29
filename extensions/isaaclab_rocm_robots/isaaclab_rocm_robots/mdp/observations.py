"""Observation terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_unique, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def body_pose_b(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Pose (position, xyzw quaternion with positive w) of one body in the asset's root frame."""
    asset = env.scene[asset_cfg.name]
    pos_b, quat_b = subtract_frame_transforms(
        asset.data.root_pos_w.torch,
        asset.data.root_quat_w.torch,
        asset.data.body_pos_w.torch[:, asset_cfg.body_ids[0]],
        asset.data.body_quat_w.torch[:, asset_cfg.body_ids[0]],
    )
    return torch.cat((pos_b, quat_unique(quat_b)), dim=-1)


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Position of the object in the robot's root frame."""
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, obj.data.root_pos_w.torch
    )
    return pos_b
