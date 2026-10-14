
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_K_EL
from unitree_rl_lab.tasks.locomotion import mdp

from .running_env_cfg import GAIT_FEET_BODY_NAMES
from .running_env_cfg import CurriculumCfg as BaseCurriculumCfg
from .running_env_cfg import RewardsCfg as BaseRewardsCfg
from .running_env_cfg import RobotEnvCfg as BaseRobotEnvCfg
from .running_env_cfg import RobotPlayEnvCfg as BaseRobotPlayEnvCfg
ENERGY_POWER_REF_W = 300.0
ENERGY_POWER_WEIGHT = -0.5  # was -0.25 (R3); since 2026-10-12 only the CoT variant below uses joint_power
ENERGY_TORQUE_WEIGHT = -0.7  # was -0.35 (R3)
# Landing impulse (2026-10-12): J40 ~ 18 N*s per touchdown x 4 touchdowns/s = 1.4 N*s per step on average,
# so -0.3 gives ~ -0.4 of Episode_Reward, the 12-13% of positive reward that joint_power (-0.38) used to
# take. Together with joint_torque (-0.77, 25%) the total pressure matches the 09-28 Reward run (37%).
IMPACT_IMPULSE_WEIGHT = -0.3
IMPACT_WINDOW_S = 0.04  # the Energy reward spike sits at 10-20 ms after touchdown; 40 ms holds it at 1-2 m/s
IMPACT_MIN_AIR_S = 0.04  # shorter gaps are sole flicker inside one stance, not a touchdown
# PACE power cost (2026-10-13, back from the torque^2 term): same numbers as the 09-23 Energy reward run.
# k_el (with its 1/3 scale assumption) lives with the robot asset so the reward and the energy/elec_*
# metrics use the same numbers.
PACE_K_EL = UNITREE_G1_29DOF_K_EL
PACE_K_REGEN = 0.0  # per-motor braking work recovered at this efficiency; 0 = dissipated (no regen path on the G1)
PACE_SPEED_EXPONENT = 1.0  # gamma = 1 / (|v_cmd|^p + 1); paper p = 2, 1 keeps the high-speed pressure
PACE_ENERGY_WEIGHT = -5e-3  # per Watt (after gamma); ~25% of positive reward at 2 m/s
# Peak ground reaction force (2026-10-13, replaces the landing impulse): per policy step, each foot's
# largest |F| over the 4 physics substeps above FOOT_FORCE_THRESHOLD, summed over feet [N]. No time
# window, so loading cannot be deferred past it (the impulse run touched down softly for 40 ms and
# loaded afterwards: 40 ms peak -58%, stance peak only -4%). Running stance peaks are 2.2-2.6 BW, so
# 1.8 BW presses on the top of every stance without touching the ~1.4 BW support level a 30% flight
# gait needs. Weight: ~20-30 N mean violation per step expected -> ~-0.3..-0.45 of Episode_Reward,
# the share feet_impact (-0.4) had; recalibrate from Episode_Reward/feet_force after the gate opens.
G1_BODY_WEIGHT_N = 33.34 * 9.81
FOOT_FORCE_THRESHOLD_BW = 2.0
FOOT_FORCE_THRESHOLD_N = FOOT_FORCE_THRESHOLD_BW * G1_BODY_WEIGHT_N  # ~589 N
FOOT_FORCE_WEIGHT = -1.5e-2  # per N of violation
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
    """Baseline reward terms plus the PACE power cost and a peak-GRF penalty (weights set by the gate).

    2026-10-13: the pair under test is "PACE power + peak contact force". The 10-12 run (torque^2 +
    landing impulse, params/ copy in logs/.../2026-10-12_18-56-43_torque) cut the 40 ms landing impulse
    by ~75% but left the stance peak at -4% and did not lower mechanical work at all; the 09-28 Reward
    run (joint_power + joint_torque) and the 09-23 Energy reward run (PACE only) keep their params/ copies.
    Both terms start at 0 -- RewardManager skips zero-weight terms, so before the gate opens this is
    bit-identical to the baseline objective.
    """

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
    # Isaac Lab's contact_forces: max over the sensor history (one policy step of substeps, see
    # RobotEnvCfg.__post_init__) of |F| per foot, minus the threshold, clipped at 0, summed over feet.
    feet_force = RewTerm(
        func=mdp.contact_forces,
        weight=0.0,
        params={
            "threshold": FOOT_FORCE_THRESHOLD_N,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=GAIT_FEET_BODY_NAMES),
        },
    )


@configclass
class CurriculumCfg(BaseCurriculumCfg):
    """Baseline curriculum plus the standing gate for the two cost terms."""

    joint_energy_pace_weight = _gate("joint_energy_pace", PACE_ENERGY_WEIGHT)
    feet_force_weight = _gate("feet_force", FOOT_FORCE_WEIGHT)


@configclass
class RobotEnvCfg(BaseRobotEnvCfg):
    rewards: RewardsCfg = RewardsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # One policy step of force samples (3 -> 4 substeps) so feet_force sees the peak at every substep.
        # Observations do not read the history; undesired_contacts sees one more substep in its max.
        self.scene.contact_forces.history_length = self.decimation


@configclass
class RobotPlayEnvCfg(BaseRobotPlayEnvCfg):
    rewards: RewardsCfg = RewardsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.contact_forces.history_length = self.decimation



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



# ---- PACE variant: commented out 2026-10-12 (user request); the 09-23 PACE run keeps its params/ copy.
# Uncomment this block and the Running-PACE registration in __init__.py to re-enable it.
# # k_el (with its 1/3 scale assumption) lives with the robot asset so the reward and the
# # energy/elec_* metrics use the same numbers.
# PACE_K_EL = UNITREE_G1_29DOF_K_EL
# PACE_K_REGEN = 0.0  # per-motor braking work recovered at this efficiency; 0 = dissipated (no regen path on the G1)
# PACE_SPEED_EXPONENT = 1.0  # gamma = 1 / (|v_cmd|^p + 1); paper p = 2 (used by the 09-22 run), 1 leans on high speed
# PACE_ENERGY_WEIGHT = -5e-3  # per Watt (after gamma); ~25% of positive reward at 2 m/s (was -3.5e-3 with p = 2)
#
#
# @configclass
# class PaceRewardsCfg(BaseRewardsCfg):
#     """Baseline terms plus the single PACE power cost (weight set by the gate); no joint_power / joint_torque."""
#
#     joint_energy_pace = RewTerm(
#         func=mdp.pace_energy_cost,
#         weight=0.0,
#         params={
#             "k_el": PACE_K_EL,
#             "k_regen": PACE_K_REGEN,
#             "include_potential": False,
#             "speed_exponent": PACE_SPEED_EXPONENT,
#         },
#     )
#
#
# @configclass
# class PaceCurriculumCfg(BaseCurriculumCfg):
#     joint_energy_pace_weight = _gate("joint_energy_pace", PACE_ENERGY_WEIGHT)
#
#
# @configclass
# class PaceEnvCfg(BaseRobotEnvCfg):
#     rewards: PaceRewardsCfg = PaceRewardsCfg()
#     curriculum: PaceCurriculumCfg = PaceCurriculumCfg()
#
#
# @configclass
# class PacePlayEnvCfg(BaseRobotPlayEnvCfg):
#     rewards: PaceRewardsCfg = PaceRewardsCfg()
#     curriculum: PaceCurriculumCfg = PaceCurriculumCfg()
