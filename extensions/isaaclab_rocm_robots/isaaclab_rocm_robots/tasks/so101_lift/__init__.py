import gymnasium as gym

for _id, _cfg in (("ROCm-Lift-Cube-SO101", "SO101LiftEnvCfg"), ("ROCm-Lift-Cube-SO101-Play", "SO101LiftEnvCfg_PLAY")):
    gym.register(
        id=_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.lift_env_cfg:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:SO101LiftPPORunnerCfg",
        },
    )
