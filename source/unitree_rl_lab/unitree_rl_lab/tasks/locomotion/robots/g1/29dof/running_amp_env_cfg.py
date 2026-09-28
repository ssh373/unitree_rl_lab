"""Running task with the hand-crafted gait clock replaced by an AMP style reward.

Ablation B of the energy-efficiency study. The baseline (``running_env_cfg.py``) drives its
running gait with a wall-clock phase: ``gait_phase`` (sin/cos of the clock) in the policy and
critic observations, and two rewards computed from the same clock -- ``run_contact_pattern``
(exact match of the expected single-support/flight contact state) and ``push_off`` (upward base
velocity in late stance). Here the clock is removed on both sides and the gait's rhythm, phase
and flight are left to the AMP style reward learned from human running motions (decision
2026-09-21). The two clock rewards go with it: without the phase in the observation the policy
cannot lock onto the clock, so an exact-match reward would be noise. ``short_contact`` stays --
it penalises stances longer than the running stance target without reading the clock, so it
keeps the gait a run at every speed while AMP decides what the run looks like.

One clock-free stepping reward is added in their place: ``feet_air_time_positive_biped`` (the
term Isaac Lab's own G1/H1/Digit/Cassie locomotion configs use). It pays only while exactly one
foot is on the ground, for the shorter of the swing foot's air time and the stance foot's
contact time, capped at ``threshold``. Double support and flight both score zero, so it says
"alternate your feet" without saying when or how fast. It is there because the first clock-free
run (``2026-09-21_10-56-46``) found the cheapest way around ``short_contact``: hopping on both
feet at once -- symmetric, never single-stance, hence never at risk of falling -- while the
speed-matched reference at 0.15 m/s (people standing about) raised no objection.

Posture terms removed as well (decision 2026-09-23, after a probe of the discriminator features):
``joint_deviation_legs`` (hip roll/yaw, -1.0), ``joint_deviation_arm_lateral`` (-0.5),
``joint_deviation_waists`` (-1.0) and ``flat_orientation_l2`` (-2.0) are dropped, and
``base_height`` is relaxed from -10 back to the pre-09-15 -2. On the baseline policy the reference
differs from the policy on exactly the joints these terms pin: hip roll/yaw amplitude x11-25,
knee x2.5-2.8, shoulder yaw / waist pitch / torso lean frozen at default (std ~0.01 rad) against
0.1-0.4 rad in the reference. A linear probe separated the two sides at 100% on those features
alone, the discriminator saturated within 50 iterations in every AMP run (acc 1.0, d_policy -0.9),
and the policy could not move toward the reference without paying these penalties -- so the style
reward degenerated into a constant alive bonus (standing still / hopping) or, at low weight, the
partial compliance dragged tracking under the curriculum gate. These terms are hand-crafted style,
which is what AMP is here to replace. Safety and effort terms stay: ``dof_pos_limits``,
``undesired_contacts``, ``action_rate``, ``joint_acc``, ``joint_vel``, ``feet_slide``, and the
actuator effort limits in the robot asset are untouched (user requirement).

The rest -- scene, events, commands, actions, the other policy/critic terms, the remaining
reward terms, terminations and curriculum -- is inherited unchanged. The comparison this arm
supports is therefore "baseline with its hand-crafted gait and posture terms replaced by AMP",
not "baseline plus AMP".

The environment side of AMP itself is only the discriminator observation group
(``mdp.amp_observations.AmpObsCfg``); the discriminator, the style reward and the reference
motion buffer live in the algorithm layer (``unitree_rl_lab.algorithms``) and are selected
through the agent configuration. The ``amp`` group is kept in the rollout buffer for the
discriminator and never reaches the actor or critic.

Deployment: the policy input is 10 narrower than the baseline's (no ``gait_phase`` x history 5),
so a policy trained here needs the ``deploy.yaml`` exported from its own run, and the gait-clock
masking in the deploy stack does not apply to it.
"""

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
    joint_deviation_arm_lateral = None  # shoulder roll/yaw
    joint_deviation_waists = None  # waist yaw/roll/pitch
    flat_orientation_l2 = None  # reference leans 4 deg +- 10 deg; the -2.0 term held the torso at 0.3 deg
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

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = AMP_START_LIN_VEL_X


@configclass
class RobotPlayEnvCfg(BaseRobotPlayEnvCfg):
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
