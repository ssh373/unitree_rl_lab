"""Safety-CRL v1: the Energy reward (PACE) arm's objective plus three *safety* constraints.

Research question (2026-10-20): in energy-efficient humanoid running, how does adding physical safety
constraints change the energy / performance / safety trade-off?

    max_pi  R_task(pi) - w_E * E(pi)     s.t.  J_safety,i(pi) <= d_i

* **Rewards are the PACE arm's, unchanged** (baseline running rewards + the PACE power cost, weight -5e-3 via the
  standing gate), so the only difference to the Energy reward arm is the constraint set. Two baseline penalty
  terms -- the soft joint-limit excess and non-foot contacts -- move from the reward into the constraints below
  (not duplicated); their multipliers start at the baseline-equivalent weight, so until a multiplier moves the
  objective is exactly the PACE arm's. No termination penalty (user decision 2026-10-20: keep the reward
  identical to PACE; with three constraints the penalty cannot outweigh the task reward).
* **Constraints = what must not be violated**, not how the baseline moved: joint-limit excess (mean),
  joint-limit violation rate (prob) and dangerous contacts (prob). Shaping terms (action rate, joint
  deviation, slide, height, ...) stay rewards. v2 adds actuator torque and joint velocity limits
  (measure them with the pre-clipping torque first); a "dangerous tilt" constraint replaces flat_orientation
  once the angle is chosen from the baseline / PACE tilt distributions.
* **Bounds are measured the way the solver measures costs**: stochastic policy (trained noise std), training env
  cfg with the command ranges opened, generator terrain, 1024 envs x 1500 steps
  (``scripts/rsl_rl/measure_constraint_stats.py --train_cfg --terrain generator --stochastic``,
  ``logs/constraint_stats_baseline_trainregime_stochastic.csv``). The earlier constrained task used
  deterministic flat-ground bounds, which every policy violates 2-12x under exploration noise -- its multipliers
  saturated for that reason alone. These bounds are the *feasibility* reference (baseline level); the actual
  safety targets (hardware limits, acceptable risk) are tightened afterwards.
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

# -- energy reward: identical to the 2026-09-23 PACE run (params/env.yaml of 2026-09-23_08-28-15_pace) ---------
PACE_K_EL = UNITREE_G1_29DOF_K_EL
PACE_K_REGEN = 0.0
PACE_SPEED_EXPONENT = 1.0
PACE_ENERGY_WEIGHT = -5e-3
STAND_EP_LEN = 500.0
STAND_RAMP_ITERS = 500

# -- constraint bounds: baseline model_29999, stochastic policy, training conditions (2026-10-20) -------------------
DOF_POS_EXCESS_BOUND = 0.00148  # rad, mean soft-limit excess summed over joints   baseline 0.001481
DOF_POS_LIMIT_ALLOWANCE = 0.065  # fraction of steps with any soft-limit excess       baseline 0.0654 (initial reference; tighten later)
UNDESIRED_CONTACT_ALLOWANCE = 0.001  # fraction of steps with a non-foot contact   baseline 0.00034 (noisy estimate; user: start at 0.1%)

# -- initial multipliers = baseline-equivalent weights, so the pre-adaptation objective is the PACE arm's ------------
LAMBDA_DOF_POS_EXCESS = 5.0 * DOF_POS_EXCESS_BOUND  # 0.0074: lambda * f / bound == 5.0 * f
# prob terms cost 1/allowance per violating step; the baseline charged 5.0 * excess (~0.023 rad per violating step)
# and 1.0 per contacting body.
LAMBDA_DOF_POS_LIMITS = 5.0 * (DOF_POS_EXCESS_BOUND / DOF_POS_LIMIT_ALLOWANCE) * DOF_POS_LIMIT_ALLOWANCE  # 0.0074
LAMBDA_UNDESIRED_CONTACTS = 1.0 * 1.0 * UNDESIRED_CONTACT_ALLOWANCE  # 0.001


@configclass
class RewardsCfg(BaseRewardsCfg):
    """PACE arm rewards; joint-limit and contact penalties are constraints instead."""

    dof_pos_limits = None
    undesired_contacts = None

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
    """Safety constraints; read by :class:`mdp.constraints.ConstraintManager`. All gated (standing)."""

    dof_pos_excess = ConTerm(
        func=mdp.joint_pos_limits, kind="mean", bound=DOF_POS_EXCESS_BOUND, lambda_init=LAMBDA_DOF_POS_EXCESS
    )
    dof_pos_limits = ConTerm(
        func=mdp.joint_pos_limits,
        kind="prob",
        limit=0.0,
        bound=DOF_POS_LIMIT_ALLOWANCE,
        lambda_init=LAMBDA_DOF_POS_LIMITS,
    )
    undesired_contacts = ConTerm(
        func=mdp.undesired_contacts,
        kind="prob",
        limit=0.0,
        bound=UNDESIRED_CONTACT_ALLOWANCE,
        lambda_init=LAMBDA_UNDESIRED_CONTACTS,
        params={"threshold": 1, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"])},
    )


@configclass
class CurriculumCfg(BaseCurriculumCfg):
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
