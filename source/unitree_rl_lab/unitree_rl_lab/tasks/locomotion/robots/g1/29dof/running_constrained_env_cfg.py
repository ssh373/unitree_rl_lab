"""G1 running with constraints instead of penalty rewards (Kim et al., "Not Only Rewards But Also
Constraints", T-RO 2024), for the energy-efficiency study.

Objective = minimize energy subject to baseline-level behaviour:

* **Rewards** keep ``alive``, the clock-based gait terms (``run_contact_pattern``, ``push_off``,
  ``short_contact``, ``stand_feet_contact`` -- left as rewards for now, 2026-10-14) and the PACE power
  cost with a fixed weight (energy is to be *minimized*, so it is a reward, not a bounded constraint).
* **Constraints** (:class:`ConstraintsCfg`) take over command tracking and every other baseline penalty.
  Tracking is a constraint rather than a reward so the energy/performance trade-off is fixed by a bound
  ("tracking error no worse than baseline") instead of a weight ratio.

Thresholds (set 2026-10-14) come from the baseline policy's per-step statistics
(``scripts/rsl_rl/measure_constraint_stats.py``, run 2026-09-15_10-44-43_baseline model_29999, 1024 envs x
1500 steps, training command distribution incl. 10% standing, flat ground;
``logs/constraint_stats_baseline_29999.csv``): ``mean`` terms get ``bound = baseline mean`` exactly (no
slack, user decision), ``prob`` terms get ``limit = baseline p97.5`` with the paper's 2.5% allowance for
every prob term (also for dof_pos_limits / undesired_contacts, which the baseline violates far less often --
user decision). Yaw tracking is the one bound set *tighter* than baseline (0.08 rad/s vs. the measured
0.31): the user wants yaw tracked about as well as x. If a bound is infeasible its multiplier sits at the cap:
watch ``Loss/constraint/track_ang_vel_z/lambda`` (bounds can be changed and the run resumed; multipliers are
checkpointed by name).

The tracking multipliers start at the value where the penalty's gradient d(lambda * dt * e / bound)/de
matches the baseline kernel reward's gradient near its operating point (e ~ 0.1): lin 0.3, yaw 1.0
(2026-10-14; 2.0 made the initial yaw penalty 40x the baseline's tracking signal). Together with the
solver's ``violation_clip = 1.0`` (lambda moves at most lr per iteration) this keeps the early penalty on
the scale of the rewards; the slow multiplier growth is the implicit curriculum.

The baseline cfg is untouched; this module only overrides terms.
"""

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_K_EL
from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.mdp.constraints import ConstraintTermCfg as ConTerm

from .running_env_cfg import CurriculumCfg as BaseCurriculumCfg
from .running_env_cfg import RewardsCfg as BaseRewardsCfg
from .running_env_cfg import RobotEnvCfg as BaseRobotEnvCfg
from .running_env_cfg import RobotPlayEnvCfg as BaseRobotPlayEnvCfg

# -- energy reward: same PACE settings as the Energy reward arm; weight set by the standing gate ----------
PACE_K_EL = UNITREE_G1_29DOF_K_EL
PACE_K_REGEN = 0.0
PACE_SPEED_EXPONENT = 1.0
PACE_ENERGY_WEIGHT = -5e-3  # the Energy reward arm's value; revisit once the constraints hold
STAND_EP_LEN = 500.0
STAND_RAMP_ITERS = 500

# -- constraint thresholds: baseline model_29999 statistics (see the module docstring) -------------------
PROB_ALLOWANCE = 0.025  # paper: 2.5% of steps may violate a probabilistic constraint
# mean terms: bound = baseline mean
TRACK_LIN_VEL_BOUND = 0.077  # m/s   mean |v_cmd - v|            baseline 0.0771 (p50 0.032, p97.5 0.66)
TRACK_ANG_VEL_BOUND = 0.08  # rad/s mean |w_cmd - w_z|          baseline 0.311 -> tightened to ~x level (user, 2026-10-14)
LIN_VEL_Z_BOUND = 0.121  # (m/s)^2 mean v_z^2                   baseline 0.1207
BASE_HEIGHT_BOUND = 0.00241  # m^2 mean (h - 0.78)^2            baseline 0.00241 (mean height 0.737 m)
JOINT_VEL_BOUND = 57.4  # (rad/s)^2 mean sum qdot^2             baseline 57.38
JOINT_ACC_BOUND = 1.38e5  # (rad/s^2)^2 mean sum qddot^2        baseline 1.378e5 (p50 2.5e4; tail = command switches)
ACTION_RATE_BOUND = 1.19  # mean sum (a_t - a_{t-1})^2          baseline 1.185
DEV_ARM_SWING_BOUND = 0.781  # rad mean sum|q - q0| shoulder pitch, elbow   baseline 0.7808
DEV_ARM_LATERAL_BOUND = 0.130  # rad shoulder roll/yaw, wrists             baseline 0.1303
DEV_WAIST_BOUND = 0.0414  # rad waist                                      baseline 0.04144
DEV_LEGS_BOUND = 0.130  # rad hip roll/yaw                                 baseline 0.1296
FEET_SLIDE_BOUND = 0.0558  # m/s mean foot speed in contact                baseline 0.05582
# prob terms: limit = baseline p97.5 (or 0 where the baseline is at 0 for >97.5% of steps)
ANG_VEL_XY_LIMIT = 0.91  # (rad/s)^2 |w_xy|^2 per step          baseline p97.5 0.9104
FLAT_ORIENTATION_LIMIT = 0.0101  # gravity xy^2 per step        baseline p97.5 0.01008 (~5.8 deg tilt)
DOF_POS_LIMIT = 0.0  # rad; any soft-limit excess is a violation  baseline violates 0.4% of steps
UNDESIRED_CONTACT_LIMIT = 0.0  # count; any non-foot contact      baseline violates ~0% of steps
TRACK_LIN_LAMBDA_INIT = 0.3  # gradient-matched to the baseline's track_lin_vel_xy kernel (std sqrt(0.1)) near e ~ 0.1 m/s
TRACK_ANG_LAMBDA_INIT = 1.0  # same for track_ang_vel_z (std 0.2) near e ~ 0.1 rad/s; both ungated


@configclass
class RewardsCfg(BaseRewardsCfg):
    """alive + gait clock terms + energy. Everything else moved to :class:`ConstraintsCfg`."""

    track_lin_vel_xy = None
    track_ang_vel_z = None
    base_linear_velocity = None
    base_angular_velocity = None
    joint_vel = None
    joint_acc = None
    action_rate = None
    dof_pos_limits = None
    joint_deviation_arm_swing = None
    joint_deviation_arm_lateral = None
    joint_deviation_waists = None
    joint_deviation_legs = None
    flat_orientation_l2 = None
    base_height = None
    feet_slide = None
    undesired_contacts = None

    # Starts at 0 and is ramped in by the standing gate (CurriculumCfg), as in the Energy reward arm.
    joint_energy_pace = RewTerm(
        func=mdp.pace_energy_cost,
        weight=0.0,
        params={
            "k_el": PACE_K_EL,
            "k_regen": PACE_K_REGEN,
            "include_potential": False,
            "speed_exponent": PACE_SPEED_EXPONENT,
        },
    )


@configclass
class ConstraintsCfg:
    """Read by :class:`mdp.constraints.ConstraintManager`; ``mean``: E[f] <= bound, ``prob``: P(f > limit) <= bound."""

    # -- task performance: not gated, so the policy has a reason to move from the first iteration
    track_lin_vel_xy = ConTerm(
        func=mdp.track_lin_vel_xy_error,
        kind="mean",
        bound=TRACK_LIN_VEL_BOUND,
        lambda_init=TRACK_LIN_LAMBDA_INIT,
        gated=False,
        params={"command_name": "base_velocity"},
    )
    track_ang_vel_z = ConTerm(
        func=mdp.track_ang_vel_z_error,
        kind="mean",
        bound=TRACK_ANG_VEL_BOUND,
        lambda_init=TRACK_ANG_LAMBDA_INIT,
        gated=False,
        params={"command_name": "base_velocity"},
    )

    # -- base
    lin_vel_z = ConTerm(func=mdp.lin_vel_z_l2, kind="mean", bound=LIN_VEL_Z_BOUND)
    ang_vel_xy = ConTerm(func=mdp.ang_vel_xy_l2, kind="prob", limit=ANG_VEL_XY_LIMIT, bound=PROB_ALLOWANCE)
    flat_orientation = ConTerm(
        func=mdp.flat_orientation_l2, kind="prob", limit=FLAT_ORIENTATION_LIMIT, bound=PROB_ALLOWANCE
    )
    base_height = ConTerm(
        func=mdp.base_height_l2, kind="mean", bound=BASE_HEIGHT_BOUND, params={"target_height": 0.78}
    )

    # -- joints
    joint_vel = ConTerm(func=mdp.joint_vel_l2, kind="mean", bound=JOINT_VEL_BOUND)
    joint_acc = ConTerm(func=mdp.joint_acc_l2, kind="mean", bound=JOINT_ACC_BOUND)
    action_rate = ConTerm(func=mdp.action_rate_l2, kind="mean", bound=ACTION_RATE_BOUND)
    dof_pos_limits = ConTerm(func=mdp.joint_pos_limits, kind="prob", limit=DOF_POS_LIMIT, bound=PROB_ALLOWANCE)
    joint_deviation_arm_swing = ConTerm(
        func=mdp.joint_deviation_l1,
        kind="mean",
        bound=DEV_ARM_SWING_BOUND,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_pitch_joint", ".*_elbow_joint"])},
    )
    joint_deviation_arm_lateral = ConTerm(
        func=mdp.joint_deviation_l1,
        kind="mean",
        bound=DEV_ARM_LATERAL_BOUND,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_shoulder_roll_joint", ".*_shoulder_yaw_joint", ".*_wrist_.*"]
            )
        },
    )
    joint_deviation_waists = ConTerm(
        func=mdp.joint_deviation_l1,
        kind="mean",
        bound=DEV_WAIST_BOUND,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist.*"])},
    )
    joint_deviation_legs = ConTerm(
        func=mdp.joint_deviation_l1,
        kind="mean",
        bound=DEV_LEGS_BOUND,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )

    # -- feet / contacts
    feet_slide = ConTerm(
        func=mdp.feet_slide,
        kind="mean",
        bound=FEET_SLIDE_BOUND,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    undesired_contacts = ConTerm(
        func=mdp.undesired_contacts,
        kind="prob",
        limit=UNDESIRED_CONTACT_LIMIT,
        bound=PROB_ALLOWANCE,
        params={"threshold": 1, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"])},
    )


TRACK_KERNEL_STD = math.sqrt(0.1)  # the baseline's track_lin_vel_xy kernel std, for the curriculum gate


@configclass
class CurriculumCfg(BaseCurriculumCfg):
    """Baseline curriculum with the velocity gate read from the tracking metric, plus the energy gate."""

    # The baseline gate reads the track_lin_vel_xy reward term, which is a constraint here; same 0.8 kernel
    # threshold and the same 0.05 yaw widening, read from velocity/track_xy_kernel instead.
    lin_vel_cmd_levels = CurrTerm(
        func=mdp.lin_vel_cmd_levels_from_metric, params={"threshold": 0.8, "ang_vel_delta": 0.05}
    )

    joint_energy_pace_weight = CurrTerm(
        func=mdp.reward_weight_after_standing,
        params={
            "term_name": "joint_energy_pace",
            "end_weight": PACE_ENERGY_WEIGHT,
            "ep_len_threshold": STAND_EP_LEN,
            "ramp_iters": STAND_RAMP_ITERS,
        },
    )


@configclass
class RobotEnvCfg(BaseRobotEnvCfg):
    rewards: RewardsCfg = RewardsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    constraints: ConstraintsCfg = ConstraintsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.track_kernel_std = TRACK_KERNEL_STD


@configclass
class RobotPlayEnvCfg(BaseRobotPlayEnvCfg):
    rewards: RewardsCfg = RewardsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    constraints: ConstraintsCfg = ConstraintsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.track_kernel_std = TRACK_KERNEL_STD
