"""Running task with energy-efficiency reward shaping.

This is the pure-reward baseline in ``running_env_cfg.py`` plus two normalized actuation
cost terms, and nothing else. Everything the two configurations share -- scene, events,
commands, actions, observations, terminations and all 21 baseline reward terms -- is
inherited rather than copied, so the only difference between the two runs is the
intervention being measured.

Measured against the baseline at matched training budget (both at iteration 25500,
256 envs, flat ground, fixed command speed): cost of transport down 12.4-12.5% across
0.5-2.5 m/s, sum (tau/tau_limit)^2 down ~24%, survival 100%, at the cost of roughly
0.014 m/s of extra velocity tracking error at low speed.
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_K_EL
from unitree_rl_lab.tasks.locomotion import mdp

from .running_env_cfg import CurriculumCfg as BaseCurriculumCfg
from .running_env_cfg import RewardsCfg as BaseRewardsCfg
from .running_env_cfg import RobotEnvCfg as BaseRobotEnvCfg
from .running_env_cfg import RobotPlayEnvCfg as BaseRobotPlayEnvCfg

# Both cost terms are normalized (power by ENERGY_POWER_REF_W, torque by each actuator's
# effort limit), so the weights below are on the same scale as the task rewards
# (track_lin_vel_xy = 2.0) instead of the usual hard-to-read 1e-5 range.
#
# Switch-on: every energy term starts at weight 0 and is gated on standing (user decision
# 2026-09-22, see mdp.reward_weight_after_standing). The 09-21 wandb runs showed why: with the
# penalties on from step 0 the falling phase, which earns ~0.1/step, carried a 0.03-0.07/step
# penalty, and the first stand moved from 4.3k iterations (baseline) to 6k (-0.20/-0.10), 10.6k
# (-0.40/-0.20) and beyond 12k (-0.25/-0.35 and PACE). Once standing, the same penalties were
# never a problem (~3.6/step of positive reward against 0.3-0.8). The gate opens when the mean
# episode length passes STAND_EP_LEN and ramps the weights in over STAND_RAMP_ITERS iterations.
#
# power_ref: joint_power_l1 charges per-motor positive work, which on the 2 m/s baseline gait is
# an estimated 290-350 W (gross sum |tau qdot| was 446 W; per-motor braking share 20-35%).
# 300 W puts the term near 1 there. Check against energy/pos_mech_power_w on the next run.
#
# Ratio (user decision 2026-09-21, after the 09-16 single-term runs and the 3:7 split):
# torque above power. On a quasi-direct-drive robot copper loss is the larger share of battery
# energy (roughly 7:3 vs mechanical work at the raw k_el estimate, 55:45 at scale 1/3), and the
# 09-16 ablations at 14-16k iterations showed the two terms equally cost-effective on mechanical
# CoT per unit of penalty (3.5% vs 3.4% per 0.1) while the torque term was 7x more effective on
# sum (tau/tau_limit)^2 (14% vs 2% per 0.1) with no gait or tracking change up to -0.145.
# At 2 m/s: power ~1.07 x 0.25 = -0.27, torque 1.45 x 0.35 = -0.51, total ~ -0.78, about 22% of
# the ~3.6 of positive reward (was ~10% at -0.20/-0.10 on gross power). Watch base_height and
# knee angle: a torque weight this high is untested and may straighten the knee.
ENERGY_POWER_REF_W = 300.0
ENERGY_POWER_WEIGHT = -0.25
ENERGY_TORQUE_WEIGHT = -0.35
STAND_EP_LEN = 500.0  # steps; half the 1000-step episode. Falling runs sit at 50-130.
STAND_RAMP_ITERS = 500


def _gate(term_name: str, end_weight: float) -> CurrTerm:
    """Standing gate shared by every energy variant, so the switch-on can never drift."""
    return CurrTerm(
        func=mdp.reward_weight_after_standing,
        params={
            "term_name": term_name,
            "end_weight": end_weight,
            "ep_len_threshold": STAND_EP_LEN,
            "ramp_iters": STAND_RAMP_ITERS,
        },
    )


@configclass
class RewardsCfg(BaseRewardsCfg):
    """Baseline reward terms plus the two actuation-cost penalties (weights set by the gate)."""

    # joint_power is the positive mechanical work; joint_torque is the I^2 R share that
    # mechanical power misses, and is what reaches joints that hold load without doing work
    # (waist_roll draws 0.7% of total power but sits at 79% of its effort limit).
    # Both start at 0 -- RewardManager skips zero-weight terms, so before the gate opens this
    # is bit-identical to the baseline objective.
    joint_power = RewTerm(
        func=mdp.joint_power_l1,
        weight=0.0,
        params={"power_ref": ENERGY_POWER_REF_W},
    )
    joint_torque = RewTerm(func=mdp.joint_torque_normalized_l2, weight=0.0)


@configclass
class CurriculumCfg(BaseCurriculumCfg):
    """Baseline curriculum plus the standing gate for the two cost terms."""

    joint_power_weight = _gate("joint_power", ENERGY_POWER_WEIGHT)
    joint_torque_weight = _gate("joint_torque", ENERGY_TORQUE_WEIGHT)


@configclass
class RobotEnvCfg(BaseRobotEnvCfg):
    rewards: RewardsCfg = RewardsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()


@configclass
class RobotPlayEnvCfg(BaseRobotPlayEnvCfg):
    rewards: RewardsCfg = RewardsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()


"""
CoT variant: the combined penalties plus a speed-normalized power cost.

The two terms above penalize power regardless of how fast the robot was asked to go, so the
slowest -- and least efficient -- commands get the least pressure. ``joint_cot`` adds the
adaptive energy regularization of Liang et al. (2024), a per-step cost of transport, on top of
them. It is a separate task so the combined Running-Reward configuration stays untouched.

All three terms are gated on standing like Running-Reward, then held at their final weights.

Weight: at 2 m/s the baseline's 446 W gives an exponent of 446/(2800*1.98) = 0.080 and a cost
of 0.077, so -1.5 adds -0.12 to the existing -0.37 of energy pressure (13% of the ~3.6 of
positive reward, up from 10%). At 0.5 m/s the same weight roughly doubles the pressure, which
is the point. -3.0 would reproduce the power term's 2 m/s pressure on its own, i.e. the value
to use with ENERGY_POWER_WEIGHT set to 0 for a clean replace-instead-of-add comparison.
"""

ENERGY_COT_SIGMA_LIN = 2800.0  # N; 8.5 x G1 weight, matches the paper's regime
ENERGY_COT_SIGMA_ANG = 1400.0  # N*m/rad; paper ratio sigma_ang = 0.5 * sigma_lin
ENERGY_COT_V_FLOOR = 0.3  # m/s; same floor as the training metric's cot_min_speed
ENERGY_COT_WEIGHT = -1.0  # = Liang et al.'s alpha_en; -1.5 made a stiff heel walk the first attractor at 0.4 m/s


@configclass
class CotRewardsCfg(BaseRewardsCfg):
    """Power + torque plus the speed-normalized power cost; all three weights set by the gate."""

    joint_power = RewTerm(
        func=mdp.joint_power_l1,
        weight=0.0,
        params={"power_ref": ENERGY_POWER_REF_W},
    )
    joint_torque = RewTerm(func=mdp.joint_torque_normalized_l2, weight=0.0)
    joint_cot = RewTerm(
        func=mdp.joint_power_cot_exp,
        weight=0.0,
        params={
            "sigma_lin": ENERGY_COT_SIGMA_LIN,
            "sigma_ang": ENERGY_COT_SIGMA_ANG,
            "v_floor": ENERGY_COT_V_FLOOR,
        },
    )


@configclass
class CotCurriculumCfg(BaseCurriculumCfg):
    joint_power_weight = _gate("joint_power", ENERGY_POWER_WEIGHT)
    joint_torque_weight = _gate("joint_torque", ENERGY_TORQUE_WEIGHT)
    joint_cot_weight = _gate("joint_cot", ENERGY_COT_WEIGHT)


@configclass
class CotEnvCfg(BaseRobotEnvCfg):
    rewards: CotRewardsCfg = CotRewardsCfg()
    curriculum: CotCurriculumCfg = CotCurriculumCfg()


@configclass
class CotPlayEnvCfg(BaseRobotPlayEnvCfg):
    rewards: CotRewardsCfg = CotRewardsCfg()
    curriculum: CotCurriculumCfg = CotCurriculumCfg()


"""
PACE variant: one physics-grounded power term instead of power + torque.

After "Towards bridging the gap: Systematic sim-to-real transfer for diverse legged robots"
(arXiv:2509.06342, eqs. 11-16): P_total = Joule heating + positive mechanical power, normalized
by 1 / (|v_cmd|^2 + 1). Everything is in Watt, so a single weight replaces the two dimensionless
terms above. One departure from the paper: the mechanical term is per-motor positive work,
sum max(tau_j qdot_j, 0), with braking work dissipated (k_regen = 0) -- the paper nets the
joints over a shared DC bus first. Same convention as the joint_power / joint_cot terms above.
For scale, the 2 m/s inspect_torque data gives gross sum |tau qdot| = 441 W and bus-net = 140 W;
the per-motor positive figure lies between the two and has not been measured yet.

Joule coefficients k_el = R / (r k_i)^2 are UNITREE_G1_29DOF_K_EL in the robot asset, where
their torque-speed-curve derivation and the 1/3 scale assumption are documented. Against the
1/tau_limit^2 weighting of joint_torque_normalized_l2 the hip:knee ratio is the same (1 : 0.4)
but the small motors are weighted 6x and the wrists 28x less here.

Weight: at 2 m/s the term is gamma * P_total = (1/5) * (386 + 140) W ~ 105, and -3.5e-3 makes it
~ -0.37/s, the same ~10% of positive reward the other variants use at that speed. (The paper's
-16e-5 is on a different reward scale and would be ~1% here.) Gated on standing like the
other variants: the 09-21 PACE run with the weight on from step 0 had not stood by 12k iterations.
"""

# k_el (with its 1/3 scale assumption) lives with the robot asset so the reward and the
# energy/elec_* metrics use the same numbers.
PACE_K_EL = UNITREE_G1_29DOF_K_EL
PACE_K_REGEN = 0.0  # per-motor braking work recovered at this efficiency; 0 = dissipated (no regen path on the G1)
PACE_ENERGY_WEIGHT = -3.5e-3  # per Watt (after gamma); ~10% of positive reward at 2 m/s


@configclass
class PaceRewardsCfg(BaseRewardsCfg):
    """Baseline terms plus the single PACE power cost (weight set by the gate); no joint_power / joint_torque."""

    joint_energy_pace = RewTerm(
        func=mdp.pace_energy_cost,
        weight=0.0,
        params={"k_el": PACE_K_EL, "k_regen": PACE_K_REGEN, "include_potential": False},
    )


@configclass
class PaceCurriculumCfg(BaseCurriculumCfg):
    joint_energy_pace_weight = _gate("joint_energy_pace", PACE_ENERGY_WEIGHT)


@configclass
class PaceEnvCfg(BaseRobotEnvCfg):
    rewards: PaceRewardsCfg = PaceRewardsCfg()
    curriculum: PaceCurriculumCfg = PaceCurriculumCfg()


@configclass
class PacePlayEnvCfg(BaseRobotPlayEnvCfg):
    rewards: PaceRewardsCfg = PaceRewardsCfg()
    curriculum: PaceCurriculumCfg = PaceCurriculumCfg()
