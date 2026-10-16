
import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.mdp.amp_observations import AmpObsCfg

from .running_env_cfg import GAIT_FEET_BODY_NAMES
from .running_env_cfg import EventCfg as BaseEventCfg
from .running_env_cfg import ObservationsCfg as BaseObservationsCfg
from .running_env_cfg import RewardsCfg as BaseRewardsCfg
from .running_env_cfg import RobotEnvCfg as BaseRobotEnvCfg
from .running_env_cfg import RobotPlayEnvCfg as BaseRobotPlayEnvCfg

from .running_env_cfg import CurriculumCfg as BaseCurriculumCfg
from .running_reward_env_cfg import (
    FOOT_FORCE_THRESHOLD_N,
    FOOT_FORCE_WEIGHT,
    PACE_ENERGY_WEIGHT,
    PACE_K_EL,
    PACE_K_REGEN,
    PACE_SPEED_EXPONENT,
    _gate,
)
from unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg import LOWCOT_CLIPS, LOWCOT_SPEED_RANGES

# 2026-10-16: the "pace+force" pair from running_reward_env_cfg (PACE power cost + peak-GRF penalty) is
# added on top of the amp_edit (10-15) rewards with the same constants, so the only difference to the
# 2026-10-15_11-18-38_amp_edit run is these two gated terms (both weight 0 until the policy stands).



@configclass
class PolicyCfg(BaseObservationsCfg.PolicyCfg):
    """Baseline actor observations without the gait clock."""

    gait_phase = None


@configclass
class CriticCfg(BaseObservationsCfg.CriticCfg):
    """Baseline critic observations without the gait clock."""

    gait_phase = None


@configclass
class ObservationsCfg(BaseObservationsCfg):
    """Clock-free policy/critic observations plus the discriminator group."""

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    amp: AmpObsCfg = AmpObsCfg()


@configclass
class RewardsCfg(BaseRewardsCfg):
    """Baseline rewards without the clock-driven gait terms and the hand-crafted posture terms, plus one
    clock-free stepping term. See the module docstring for the evidence behind each removal."""

    run_contact_pattern = None
    push_off = None

    # Posture / stride shape is the discriminator's job in this arm (2026-09-23). Effort limits and
    # joint position limits are NOT relaxed anywhere.
    joint_deviation_legs = None  # hip roll/yaw: reference uses +-0.2 rad, policy was pinned at 0.01
    joint_deviation_arm_lateral = None  # shoulder roll/yaw (and, in the base cfg, the wrists -- see below)
    joint_deviation_waists = None  # waist yaw/roll/pitch
    flat_orientation_l2 = None  # reference leans 4 deg +- 10 deg; the -2.0 term held the torso at 0.3 deg

    # Wrists only (2026-10-07): arm_lateral was the one term holding the six wrist joints, and without it
    # the AMP policies shake the wrists -- the retargeted reference wrists are IK noise (pos std 0.14-0.58
    # rad, left/right asymmetric) that the discriminator rewards and the 5 Nm W4010 motors chase. Same
    # kernel as the base term; shoulder roll/yaw stay free for the discriminator.
    # Weight starts at 0 and the standing gate in CurriculumCfg ramps it to WRIST_DEVIATION_WEIGHT once
    # the policy stands (2026-10-12): the 10-07 runs with a constant -0.5 never left the falling phase
    # in 30k iterations, so before standing this arm's rewards are now identical to the 10-06 run.
    joint_deviation_wrist = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_wrist_.*"])},
    )
    # Knee flexion: the -10 weight (set 2026-09-15) straightened the knee (std 0.16 rad vs 0.45 in the
    # reference). Back to the pre-09-15 -2 so the base may bob with a human stride; not removed, so the
    # policy cannot run in a crouch to cheat short_contact / feet_air_time.
    base_height = RewTerm(func=mdp.base_height_l2, weight=-2.0, params={"target_height": 0.78})

    # Alternating single stance, no clock. Isaac Lab's G1 walking config uses weight 0.25 with a
    # 0.4 s cap; running stances are ~0.25 s and this term has to beat the two-footed hop, so the
    # cap is tightened to 0.3 s and the weight raised to 1.0 (the weight run_contact_pattern had).
    # The function itself returns zero below a 0.1 m/s command, matching STAND_CMD_THRESHOLD.
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=GAIT_FEET_BODY_NAMES),
            "threshold": 0.3,
        },
    )

    # PACE power cost (same term and constants as Running-Reward "pace+force", re-enabled 2026-10-16).
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
    # Peak ground reaction force above FOOT_FORCE_THRESHOLD_N (2.0 BW), max over one policy step of
    # substeps (contact_forces.history_length = decimation, set in RobotEnvCfg.__post_init__), summed
    # over both feet. Same term and constants as Running-Reward "pace+force".
    feet_force = RewTerm(
        func=mdp.contact_forces,
        weight=0.0,
        params={
            "threshold": FOOT_FORCE_THRESHOLD_N,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=GAIT_FEET_BODY_NAMES),
        },
    )


WRIST_DEVIATION_WEIGHT = -0.5  # same as joint_deviation_arm_lateral in the base cfg

# Reference state initialisation (AMP paper): the fraction of resets that start from a reference frame.
# The clip list must match the discriminator's (RunningAmpPPORunnerCfg.algorithm.amp_cfg) so the policy
# only starts from states the discriminator rewards. 0.8 keeps 20% default-pose starts for robustness.
RSI_PROBABILITY = 0.0  # 2026-10-15: RSI off -- the retargeted frames are not dynamically feasible, amp_edit never stood (0.8); amp_edit_noRSI stood at 13k


@configclass
class EventCfg(BaseEventCfg):
    """Baseline events plus RSI, declared last so it overrides reset_base / reset_robot_joints."""

    reset_from_reference = EventTerm(
        func=mdp.reset_from_reference,
        mode="reset",
        params={
            "motion_files": LOWCOT_CLIPS,
            "clip_speed_ranges": LOWCOT_SPEED_RANGES,
            "probability": RSI_PROBABILITY,
            "yaw_range": (-math.pi, math.pi),
            "xy_range": 0.5,
            "height_clearance": 0.01,
        },
    )


# Standing gates (weight 0 until the mean episode length passes STAND_EP_LEN, then a 500-iter ramp),
# shared tracker and latch with the energy variants.
@configclass
class CurriculumCfg(BaseCurriculumCfg):
    joint_deviation_wrist_weight = _gate("joint_deviation_wrist", WRIST_DEVIATION_WEIGHT)
    joint_energy_pace_weight = _gate("joint_energy_pace", PACE_ENERGY_WEIGHT)
    feet_force_weight = _gate("feet_force", FOOT_FORCE_WEIGHT)


# Where the velocity curriculum starts. The baseline starts at (-0.4, 0.4); the AMP arm starts
# around 1.0 m/s (user decision 2026-09-22) because the reference clips run at >= 1.0 m/s and the
# 09-21 run that started at 0.4 settled into a walk there: walking tracked at 1.36 < the 1.6
# curriculum threshold, so the command range never left 1.0 and the policy never saw the
# speeds the reference covers. Asymmetric on purpose -- backward commands (reference has no
# sustained backward running) arrive only after the upper bound has climbed; the lower bound
# reaches -1.0 after 6 levels, the upper bound 2.5 after 13. limit_ranges are untouched, so
# the final command distribution is the same as every other arm.
# Lower bound back at the baseline's -0.4 (2026-09-22): the (0.8, 1.2) run stood only because the
# then-oversized style reward paid for standing still; with the style term in task units the slow
# commands are what let the policy learn balance first (the 0.4-start runs stood at iter 200-800).
AMP_START_LIN_VEL_X = (0.5, 1.2)


@configclass
class RobotEnvCfg(BaseRobotEnvCfg):
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()  # wrist, PACE and feet_force standing gates

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = AMP_START_LIN_VEL_X
        # One policy step of force samples (3 -> 4 substeps) so feet_force sees the peak at every substep
        # (as in running_reward_env_cfg). feet_air_time uses contact timers, not the history.
        self.scene.contact_forces.history_length = self.decimation


@configclass
class RobotPlayEnvCfg(BaseRobotPlayEnvCfg):
    """Play / evaluation: default-pose resets only, so the energy and impact protocols stay comparable
    across arms and never need the motion files."""

    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.contact_forces.history_length = self.decimation
