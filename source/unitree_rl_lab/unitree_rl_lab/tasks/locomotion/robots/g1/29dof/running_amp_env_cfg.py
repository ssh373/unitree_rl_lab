
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.mdp.amp_observations import AmpObsCfg

from .running_env_cfg import GAIT_FEET_BODY_NAMES
from .running_env_cfg import ObservationsCfg as BaseObservationsCfg
from .running_env_cfg import RewardsCfg as BaseRewardsCfg
from .running_env_cfg import RobotEnvCfg as BaseRobotEnvCfg
from .running_env_cfg import RobotPlayEnvCfg as BaseRobotPlayEnvCfg

from .running_env_cfg import CurriculumCfg as BaseCurriculumCfg
from .running_reward_env_cfg import (
    PACE_ENERGY_WEIGHT,
    PACE_K_EL,
    PACE_K_REGEN,
    PACE_SPEED_EXPONENT,
    _gate,
)



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
    # weight and kernel as the base term; shoulder roll/yaw stay free for the discriminator.
    joint_deviation_wrist = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
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

    # TEMP: PACE power cost on the AMP arm. Disabled 2026-10-02 (user request) for a pure-AMP run with the
    # speed-banded walk/run reference set; uncomment this term AND the CurriculumCfg / curriculum lines
    # below to re-enable.
    # joint_energy_pace = RewTerm(
    #     func=mdp.pace_energy_cost,
    #     weight=0.0,
    #     params={
    #         "k_el": PACE_K_EL,
    #         "k_regen": PACE_K_REGEN,
    #         "include_potential": False,
    #         "speed_exponent": PACE_SPEED_EXPONENT,
    #     },
    # )


# PACE standing gate -- disabled together with joint_energy_pace above (2026-10-02).
# @configclass
# class CurriculumCfg(BaseCurriculumCfg):
#     joint_energy_pace_weight = _gate("joint_energy_pace", PACE_ENERGY_WEIGHT)


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
    # curriculum: CurriculumCfg = CurriculumCfg()      # PACE gate, disabled 2026-10-02 (see above)

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = AMP_START_LIN_VEL_X


@configclass
class RobotPlayEnvCfg(BaseRobotPlayEnvCfg):
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
