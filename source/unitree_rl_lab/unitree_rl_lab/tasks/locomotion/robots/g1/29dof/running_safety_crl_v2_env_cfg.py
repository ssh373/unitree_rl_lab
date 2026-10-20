"""Safety-CRL v2: v1 plus actuator-torque, joint-velocity and dangerous-tilt constraints.

Same rewards as v1 (= the PACE arm). The added constraints use the sim2real safety definitions
(deploy/robots/g1_29dof/config/config.yaml): commanded torque above 0.97 x the motor peak (what the real-robot
clamp cuts), joint velocity above the upper-body kill threshold 15 rad/s (asset limit elsewhere), and a base tilt
past DANGEROUS_TILT_DEG.

Bounds (2026-10-20, stochastic policy, training conditions, `logs/constraint_stats_*_trainregime_stochastic.csv`):
the first run starts at the **baseline's** level so the problem is feasible from the start (v0 lesson); the
plan is to tighten per run (PACE level 0.12 / 26%, then lower) and to report the *deterministic* violation
rate as the deployment safety metric (baseline 1.9%, PACE 6.1%, v0 3.3% -- exploration noise through the PD
gains dominates the stochastic numbers: 38% / 26% / 9%). Joint velocity is essentially never exceeded by any
policy (sanity constraint). Tilt: p99 8-10 deg, max 45 deg (the termination), so 20 deg is only crossed while
falling; the 0.1% allowance is to be verified.

Caveat: a cost measured on exploration rollouts can also be met by shrinking the policy's noise std; if
`Policy/mean_noise_std` drops markedly, switch the torque cost to the mean action (tau = Kp (a_mean - q) - Kd qdot).
"""

from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.mdp.constraints import ConstraintTermCfg as ConTerm

from .running_safety_crl_env_cfg import ConstraintsCfg as V1ConstraintsCfg
from .running_safety_crl_env_cfg import CurriculumCfg, RewardsCfg
from .running_safety_crl_env_cfg import RobotEnvCfg as V1RobotEnvCfg
from .running_safety_crl_env_cfg import RobotPlayEnvCfg as V1RobotPlayEnvCfg

TORQUE_CLAMP_RATIO = 0.97  # sim2real TORQUE_CLAMP_RATIO
TORQUE_EXCESS_BOUND = 0.25  # mean sum max(0, ratio - 1)          baseline 0.2456 (PACE 0.1206, v0 0.0221)
TORQUE_OVER_LIMIT_ALLOWANCE = 0.38  # fraction of steps with any joint over   baseline 0.3835 (PACE 0.2635, v0 0.0897)
JOINT_VEL_OVER_LIMIT_ALLOWANCE = 0.001  # baseline 1.4e-4 stochastic, 0 deterministic
DANGEROUS_TILT_DEG = 20.0  # p99 of all policies is 8-10 deg; 20 deg is only crossed while falling
DANGEROUS_TILT_ALLOWANCE = 0.001  # to be verified against the measured frequency

# Initial multipliers: no baseline weight to match (these were never rewards), so start where the penalty at the
# bound is ~0.01 reward units per step (mean) / per violating step (prob); 10x cap -> 0.1.
LAMBDA_SAFETY_MEAN = 0.01
LAMBDA_SAFETY_PROB = 0.01  # multiplied by the allowance below, since a prob cost is 1 / allowance per violation


@configclass
class ConstraintsCfg(V1ConstraintsCfg):
    cmd_torque_excess = ConTerm(
        func=mdp.cmd_torque_excess,
        kind="mean",
        bound=TORQUE_EXCESS_BOUND,
        lambda_init=LAMBDA_SAFETY_MEAN,
        params={"clamp_ratio": TORQUE_CLAMP_RATIO},
    )
    cmd_torque_over_limit = ConTerm(
        func=mdp.cmd_torque_over_limit,
        kind="prob",
        limit=0.0,
        bound=TORQUE_OVER_LIMIT_ALLOWANCE,
        lambda_init=LAMBDA_SAFETY_PROB * TORQUE_OVER_LIMIT_ALLOWANCE,
        params={"clamp_ratio": TORQUE_CLAMP_RATIO},
    )
    joint_vel_over_limit = ConTerm(
        func=mdp.joint_vel_over_limit,
        kind="prob",
        limit=0.0,
        bound=JOINT_VEL_OVER_LIMIT_ALLOWANCE,
        lambda_init=LAMBDA_SAFETY_PROB * JOINT_VEL_OVER_LIMIT_ALLOWANCE,
    )
    dangerous_tilt = ConTerm(
        func=mdp.tilt_angle_deg,
        kind="prob",
        limit=DANGEROUS_TILT_DEG,
        bound=DANGEROUS_TILT_ALLOWANCE,
        lambda_init=LAMBDA_SAFETY_PROB * DANGEROUS_TILT_ALLOWANCE,
    )


@configclass
class RobotEnvCfg(V1RobotEnvCfg):
    rewards: RewardsCfg = RewardsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    constraints: ConstraintsCfg = ConstraintsCfg()


@configclass
class RobotPlayEnvCfg(V1RobotPlayEnvCfg):
    rewards: RewardsCfg = RewardsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    constraints: ConstraintsCfg = ConstraintsCfg()
