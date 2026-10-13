import gymnasium as gym

gym.register(
    id="Unitree-G1-29dof-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-Running",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.running_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.running_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:RunningPPORunnerCfg",
    },
)

# Ablation A: the baseline above plus normalized tau^2 and the landing-impulse penalty (2026-10-12;
# joint_power dropped, see running_reward_env_cfg.py).
gym.register(
    id="Unitree-G1-29dof-Running-Reward",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.running_reward_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.running_reward_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:RunningPPORunnerCfg",
    },
)

# Combined penalties plus a speed-normalized power cost (per-step CoT, Liang et al. 2024).
# Constant weights, no ramp; see running_reward_env_cfg.py for the weight arithmetic.
gym.register(
    id="Unitree-G1-29dof-Running-CoT",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.running_reward_env_cfg:CotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.running_reward_env_cfg:CotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:RunningPPORunnerCfg",
    },
)

# PACE variant commented out 2026-10-12 together with its cfg block in running_reward_env_cfg.py.
# # Single physics-grounded power term after PACE (arXiv:2509.06342); see running_reward_env_cfg.py.
# gym.register(
#     id="Unitree-G1-29dof-Running-PACE",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.running_reward_env_cfg:PaceEnvCfg",
#         "play_env_cfg_entry_point": f"{__name__}.running_reward_env_cfg:PacePlayEnvCfg",
#         "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:RunningPPORunnerCfg",
#     },
# )

# Ablation B: the baseline plus an AMP style reward learned from human running motions. The
# environment side is only the discriminator observation group; see running_amp_env_cfg.py.
gym.register(
    id="Unitree-G1-29dof-Running-AMP",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.running_amp_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.running_amp_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:RunningAmpPPORunnerCfg",
    },
)
