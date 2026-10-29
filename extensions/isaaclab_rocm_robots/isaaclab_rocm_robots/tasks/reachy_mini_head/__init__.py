"""Reachy Mini head pose (experimental). Needs ``python tools/convert_assets.py reachy_mini`` first."""

import gymnasium as gym

for _id, _cfg in (
    ("ROCm-HeadPose-ReachyMini", "ReachyMiniHeadPoseEnvCfg"),
    ("ROCm-HeadPose-ReachyMini-Play", "ReachyMiniHeadPoseEnvCfg_PLAY"),
):
    gym.register(
        id=_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.head_env_cfg:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:ReachyMiniHeadPPORunnerCfg",
        },
    )
