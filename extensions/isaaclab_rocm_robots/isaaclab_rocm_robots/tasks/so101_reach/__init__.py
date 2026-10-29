import gymnasium as gym

for _id, _cfg in (("ROCm-Reach-SO101", "SO101ReachEnvCfg"), ("ROCm-Reach-SO101-Play", "SO101ReachEnvCfg_PLAY")):
    gym.register(
        id=_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.reach_env_cfg:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:SO101ReachPPORunnerCfg",
        },
    )
