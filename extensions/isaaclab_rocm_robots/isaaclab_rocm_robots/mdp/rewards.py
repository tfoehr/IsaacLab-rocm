"""Reward terms.

The end-effector terms take a point offset in the body frame because the SO-101 USD has no TCP
frame (and Isaac Lab's FrameTransformer is not needed for one point).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_apply, quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _body_point_w(asset, body_id: int, offset: tuple[float, float, float]) -> torch.Tensor:
    pos = asset.data.body_pos_w.torch[:, body_id]
    quat = asset.data.body_quat_w.torch[:, body_id]
    off = torch.tensor(offset, device=pos.device, dtype=pos.dtype).expand_as(pos)
    return pos + quat_apply(quat, off)


def ee_position_command_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Distance [m] between the commanded position (root frame) and a point fixed on a body."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w.torch, asset.data.root_quat_w.torch, command[:, :3])
    return torch.linalg.norm(_body_point_w(asset, asset_cfg.body_ids[0], offset) - des_pos_w, dim=1)


def ee_position_command_error_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """``1 - tanh(error / std)`` of :func:`ee_position_command_error`."""
    return 1.0 - torch.tanh(ee_position_command_error(env, command_name, asset_cfg, offset) / std)


def orientation_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """``1 - tanh(angle / std)`` of the shortest-path angle between the commanded and body orientation."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_quat_w = quat_mul(asset.data.root_quat_w.torch, command[:, 3:7])
    angle = quat_error_magnitude(asset.data.body_quat_w.torch[:, asset_cfg.body_ids[0]], des_quat_w)
    return 1.0 - torch.tanh(angle / std)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    robot_cfg: SceneEntityCfg,
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """``1 - tanh(d / std)`` of the distance between the object and a grasp point on a robot body."""
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    d = torch.linalg.norm(obj.data.root_pos_w.torch - _body_point_w(robot, robot_cfg.body_ids[0], offset), dim=1)
    return 1.0 - torch.tanh(d / std)


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """1 while the object's centre is above ``minimal_height`` [m] over the env origin."""
    obj = env.scene[object_cfg.name]
    height = obj.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    return (height > minimal_height).float()


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """``1 - tanh(d / std)`` of the object-to-goal distance, paid only while the object is lifted."""
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, command[:, :3])
    d = torch.linalg.norm(des_pos_w - obj.data.root_pos_w.torch, dim=1)
    return object_is_lifted(env, minimal_height, object_cfg) * (1.0 - torch.tanh(d / std))
