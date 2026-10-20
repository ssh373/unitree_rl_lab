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

# Constrained RL (Kim et al. 2024 style): rewards keep only alive, the energy term and the clock-based gait
# terms; command tracking and every other penalty are constraints (ConstraintsCfg in the env cfg) solved
# by a per-term Lagrangian multiplier in the ExtendedPPO extension list.
gym.register(
    id="Unitree-G1-29dof-Running-Constrained",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.running_constrained_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.running_constrained_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:RunningConstrainedPPORunnerCfg",
    },
)

# Safety-CRL v1 (2026-10-20): the PACE arm's rewards + three safety constraints (joint-limit excess, joint-limit
# violation rate, non-foot contacts) with bounds measured on stochastic training-condition baseline rollouts.
gym.register(
    id="Unitree-G1-29dof-Running-SafetyCRL",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.running_safety_crl_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.running_safety_crl_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:RunningSafetyCrlPPORunnerCfg",
    },
)

# Safety-CRL v2 (2026-10-20; id without "-v2", gymnasium reads that as a version): v1 + commanded-torque limit (excess mean + over-limit prob), joint-velocity limit and
# dangerous-tilt constraints, sim2real definitions, bounds at the baseline's stochastic training-condition level.
gym.register(
    id="Unitree-G1-29dof-Running-SafetyCRL2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.running_safety_crl_v2_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.running_safety_crl_v2_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:RunningSafetyCrlPPORunnerCfg",
    },
)

