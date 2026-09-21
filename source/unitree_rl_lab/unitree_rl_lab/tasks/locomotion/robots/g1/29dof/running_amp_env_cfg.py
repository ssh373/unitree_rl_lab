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

The rest -- scene, events, commands, actions, the other policy/critic terms, the remaining 19
reward terms, terminations and curriculum -- is inherited unchanged. The comparison this arm
supports is therefore "baseline with its hand-crafted gait terms replaced by AMP", not
"baseline plus AMP".

The environment side of AMP itself is only the discriminator observation group
(``mdp.amp_observations.AmpObsCfg``); the discriminator, the style reward and the reference
motion buffer live in the algorithm layer (``unitree_rl_lab.algorithms``) and are selected
through the agent configuration. The ``amp`` group is kept in the rollout buffer for the
discriminator and never reaches the actor or critic.

Deployment: the policy input is 10 narrower than the baseline's (no ``gait_phase`` x history 5),
so a policy trained here needs the ``deploy.yaml`` exported from its own run, and the gait-clock
masking in the deploy stack does not apply to it.
"""

from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion.mdp.amp_observations import AmpObsCfg

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
    """Baseline rewards without the two clock-driven gait terms. ``short_contact`` is kept."""

    run_contact_pattern = None
    push_off = None


@configclass
class RobotEnvCfg(BaseRobotEnvCfg):
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()


@configclass
class RobotPlayEnvCfg(BaseRobotPlayEnvCfg):
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
