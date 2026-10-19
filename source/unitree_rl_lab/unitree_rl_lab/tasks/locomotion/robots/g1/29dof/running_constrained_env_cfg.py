"""G1 running with constraints instead of penalty rewards (Kim et al., "Not Only Rewards But Also
Constraints", T-RO 2024), for the energy-efficiency study.

Objective = minimize energy subject to baseline-level behaviour:

* **Rewards** keep the baseline's task terms -- command tracking (exp kernels, weight 2.0) and ``alive`` --
  the clock-based gait terms (``run_contact_pattern``, ``push_off``, ``short_contact``,
  ``stand_feet_contact``; left as rewards for now, 2026-10-14) and the PACE power cost with a fixed weight
  (energy is to be *minimized*, so it is a reward, not a bounded constraint).
* **Constraints** (:class:`ConstraintsCfg`) take over every other baseline penalty, plus two *guard*
  constraints on the tracking error (gated, ``lambda_init = 0``): they do nothing until the energy reward
  erodes tracking below the baseline's mean error, and then push back. The energy/performance trade-off is
  therefore pinned by a bound, not by the weight ratio, while the tracking reward keeps the per-step return
  positive. The first attempt (2026-10-14 run ``17-49-26_constrained``) had tracking *only* as constraints:
  with no positive reward left, the policy learned to terminate within 3 steps (flipping itself over) and
  the multipliers ran away -- a penalty can punish, it cannot give a reason to stay alive.

Thresholds (set 2026-10-14) come from the baseline policy's per-step statistics
(``scripts/rsl_rl/measure_constraint_stats.py``, run 2026-09-15_10-44-43_baseline model_29999, 1024 envs x
1500 steps, training command distribution incl. 10% standing, flat ground;
``logs/constraint_stats_baseline_29999.csv``): ``mean`` terms get ``bound = baseline mean`` exactly (no
slack, user decision); the two ``prob`` terms (joint limits, non-foot contact) use limit 0 with the paper's
2.5% allowance (user decision). 2026-10-15: ang_vel_xy, flat_orientation and the joint-limit excess became
mean terms after run ``18-57-49`` never stood with them as indicators (no gradient, 30x too weak at the
limits); the joint-limit indicator stays as a second, specification-style constraint on the same quantity. Yaw tracking is the one bound set *tighter* than baseline (0.25 rad/s vs. the measured
0.31; the earlier 0.08 was 6x violated throughout and only acted as a constant penalty). If a bound is infeasible its multiplier sits at the cap:
watch ``Loss/constraint/track_ang_vel_z/lambda`` (bounds can be changed and the run resumed; multipliers are
checkpointed by name).

Multipliers start at the **baseline-equivalent weight**: the penalty ``lambda * c / bound`` equals the
baseline's ``weight * f`` when ``lambda = weight * bound`` (mean terms), so before the standing gate opens
the objective is the baseline's, with the gait and energy terms as above. For the prob terms the indicator
costs ``1 / allowance = 40`` per violating step, so ``lambda = baseline penalty per violating step * allowance``.
The tracking guards have no baseline counterpart and start small. From there the solver moves each
multiplier *multiplicatively* (1% per iteration of clipped violation, cap 10x its start), so the baseline's
weight ratios are kept and only the violated terms grow. Run ``18-31-38`` (all terms from 0, additive
0.01/iter, cap 2) had every multiplier at 1.14 = 10-200x the baseline weights 100 iterations after the gate
opened, penalty 7.6/step, and collapsed.

The baseline cfg is untouched; this module only overrides terms.
"""

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
TRACK_ANG_VEL_BOUND = 0.25  # rad/s mean |w_cmd - w_z|          baseline 0.311, tightened ~20% (user, 2026-10-19); 0.08 was never met (6x violated)
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
# posture-shaping terms as mean constraints (2026-10-15): run 18-57-49 never stood with these as prob
# indicators -- an indicator has no gradient and, with the baseline-equivalent multiplier, charged 30x less
# than the baseline's continuous penalty while the policy sat at the joint limits (97.5% of steps).
ANG_VEL_XY_BOUND = 0.246  # (rad/s)^2 mean |w_xy|^2                baseline 0.2464
FLAT_ORIENTATION_BOUND = 0.00289  # mean gravity xy^2               baseline 0.002891
DOF_POS_EXCESS_BOUND = 0.000119  # rad mean soft-limit excess       baseline 0.0001189
# prob terms: limit 0 where the baseline is at 0 for >97.5% of steps
DOF_POS_LIMIT = 0.0  # rad; any soft-limit excess is a violation  baseline violates 0.4% of steps
UNDESIRED_CONTACT_LIMIT = 0.0  # count; any non-foot contact      baseline violates ~0% of steps

# -- initial multipliers: baseline weight x bound (mean), baseline penalty per violating step x allowance (prob) --
def _lam(weight: float, bound: float) -> float:
    return weight * bound


LAMBDA_TRACK_GUARD = 0.005  # no baseline counterpart; ~0.0006/step at 6x bound, negligible until it grows
LAMBDA_LIN_VEL_Z = _lam(0.05, LIN_VEL_Z_BOUND)  # 0.006
LAMBDA_BASE_HEIGHT = _lam(10.0, BASE_HEIGHT_BOUND)  # 0.024
LAMBDA_JOINT_VEL = _lam(0.001, JOINT_VEL_BOUND)  # 0.057
LAMBDA_JOINT_ACC = _lam(2.5e-7, JOINT_ACC_BOUND)  # 0.035
LAMBDA_ACTION_RATE = _lam(0.05, ACTION_RATE_BOUND)  # 0.060
LAMBDA_DEV_ARM_SWING = _lam(0.02, DEV_ARM_SWING_BOUND)  # 0.016
LAMBDA_DEV_ARM_LATERAL = _lam(0.5, DEV_ARM_LATERAL_BOUND)  # 0.065
LAMBDA_DEV_WAIST = _lam(1.0, DEV_WAIST_BOUND)  # 0.041
LAMBDA_DEV_LEGS = _lam(1.0, DEV_LEGS_BOUND)  # 0.130
LAMBDA_FEET_SLIDE = _lam(0.2, FEET_SLIDE_BOUND)  # 0.011
LAMBDA_ANG_VEL_XY = _lam(0.05, ANG_VEL_XY_BOUND)  # 0.0123
LAMBDA_FLAT_ORIENTATION = _lam(2.0, FLAT_ORIENTATION_BOUND)  # 0.0058
LAMBDA_DOF_POS_EXCESS = _lam(5.0, DOF_POS_EXCESS_BOUND)  # 0.0006
LAMBDA_DOF_POS_LIMITS = 5.0 * 0.01 * PROB_ALLOWANCE  # prob guard: baseline 5.0 * excess ~ 0.01 rad per violating step -> 0.00125
LAMBDA_UNDESIRED_CONTACTS = 1.0 * 1.0 * PROB_ALLOWANCE  # baseline 1.0 per contacting body -> 0.025



@configclass
class RewardsCfg(BaseRewardsCfg):
    """Tracking + alive + gait clock terms + energy. Every penalty moved to :class:`ConstraintsCfg`."""

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

    # Falling must never be the cheap way out of the penalties: with the multipliers free to grow, three
    # runs (10-14, 10-15, 10-19) slid into "end the episode early" once penalty + energy outweighed the
    # positive per-step reward. -200 is Isaac Lab's official G1 value; -4 per fall (x dt) exceeds anything an
    # early termination can save. Time-outs are not charged.
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

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
    """Read by :class:`mdp.constraints.ConstraintManager`; ``mean``: E[f] <= bound, ``prob``: P(f > limit) <= bound.
    All terms are gated (multipliers hold their baseline-equivalent start until the policy stands)."""

    # -- task performance guards: the tracking *reward* drives tracking; these only act once the energy
    # term pushes the error above the baseline's level (gated, start at 0 like every other constraint)
    track_lin_vel_xy = ConTerm(
        func=mdp.track_lin_vel_xy_error,
        kind="mean",
        bound=TRACK_LIN_VEL_BOUND,
        lambda_init=LAMBDA_TRACK_GUARD,
        params={"command_name": "base_velocity"},
    )
    track_ang_vel_z = ConTerm(
        func=mdp.track_ang_vel_z_error,
        kind="mean",
        bound=TRACK_ANG_VEL_BOUND,
        lambda_init=LAMBDA_TRACK_GUARD,
        params={"command_name": "base_velocity"},
    )

    # -- base
    lin_vel_z = ConTerm(func=mdp.lin_vel_z_l2, kind="mean", bound=LIN_VEL_Z_BOUND, lambda_init=LAMBDA_LIN_VEL_Z)
    ang_vel_xy = ConTerm(func=mdp.ang_vel_xy_l2, kind="mean", bound=ANG_VEL_XY_BOUND, lambda_init=LAMBDA_ANG_VEL_XY)
    flat_orientation = ConTerm(
        func=mdp.flat_orientation_l2, kind="mean", bound=FLAT_ORIENTATION_BOUND, lambda_init=LAMBDA_FLAT_ORIENTATION
    )
    base_height = ConTerm(
        func=mdp.base_height_l2,
        kind="mean",
        bound=BASE_HEIGHT_BOUND,
        lambda_init=LAMBDA_BASE_HEIGHT,
        params={"target_height": 0.78},
    )

    # -- joints
    joint_vel = ConTerm(func=mdp.joint_vel_l2, kind="mean", bound=JOINT_VEL_BOUND, lambda_init=LAMBDA_JOINT_VEL)
    joint_acc = ConTerm(func=mdp.joint_acc_l2, kind="mean", bound=JOINT_ACC_BOUND, lambda_init=LAMBDA_JOINT_ACC)
    action_rate = ConTerm(func=mdp.action_rate_l2, kind="mean", bound=ACTION_RATE_BOUND, lambda_init=LAMBDA_ACTION_RATE)
    # Joint limits twice, on purpose: the mean of the soft-limit excess is the baseline's continuous penalty
    # (the learning signal: how far, which way), the indicator is the paper-style specification (at most 2.5%
    # of steps beyond the soft limit). Once the mean is under its bound its multiplier decays and the prob
    # term is what remains. The hard limits are enforced by PhysX anyway; these keep the policy off them.
    dof_pos_excess = ConTerm(
        func=mdp.joint_pos_limits, kind="mean", bound=DOF_POS_EXCESS_BOUND, lambda_init=LAMBDA_DOF_POS_EXCESS
    )
    dof_pos_limits = ConTerm(
        func=mdp.joint_pos_limits, kind="prob", limit=DOF_POS_LIMIT, bound=PROB_ALLOWANCE, lambda_init=LAMBDA_DOF_POS_LIMITS
    )
    joint_deviation_arm_swing = ConTerm(
        func=mdp.joint_deviation_l1,
        kind="mean",
        bound=DEV_ARM_SWING_BOUND,
        lambda_init=LAMBDA_DEV_ARM_SWING,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_pitch_joint", ".*_elbow_joint"])},
    )
    joint_deviation_arm_lateral = ConTerm(
        func=mdp.joint_deviation_l1,
        kind="mean",
        bound=DEV_ARM_LATERAL_BOUND,
        lambda_init=LAMBDA_DEV_ARM_LATERAL,
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
        lambda_init=LAMBDA_DEV_WAIST,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist.*"])},
    )
    joint_deviation_legs = ConTerm(
        func=mdp.joint_deviation_l1,
        kind="mean",
        bound=DEV_LEGS_BOUND,
        lambda_init=LAMBDA_DEV_LEGS,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )

    # -- feet / contacts
    feet_slide = ConTerm(
        func=mdp.feet_slide,
        kind="mean",
        bound=FEET_SLIDE_BOUND,
        lambda_init=LAMBDA_FEET_SLIDE,
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
        lambda_init=LAMBDA_UNDESIRED_CONTACTS,
        params={"threshold": 1, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"])},
    )


@configclass
class CurriculumCfg(BaseCurriculumCfg):
    """Baseline curriculum (velocity gate on the tracking reward, as before) plus the energy gate."""

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


@configclass
class RobotPlayEnvCfg(BaseRobotPlayEnvCfg):
    rewards: RewardsCfg = RewardsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    constraints: ConstraintsCfg = ConstraintsCfg()
