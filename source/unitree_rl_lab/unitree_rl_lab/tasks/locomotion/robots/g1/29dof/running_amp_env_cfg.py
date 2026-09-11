"""Running task plus the observation group for an AMP style reward.

Ablation B of the energy-efficiency study: the pure-reward baseline in ``running_env_cfg.py``
with human running motions mixed in through Adversarial Motion Priors. The environment side of
AMP is only the discriminator observation group (``mdp.amp_observations.AmpObsCfg``); the
discriminator, the style reward and the reference motion buffer live in the algorithm layer
(``unitree_rl_lab.algorithms``) and are selected through the agent configuration.

Everything else -- scene, events, commands, actions, policy/critic observations, all 21
baseline reward terms, terminations and curriculum -- is inherited, so the only difference to
the baseline run is the intervention being measured. The policy network input is unchanged:
the ``amp`` group is kept in the rollout buffer for the discriminator and never reaches the
actor or critic, so exported policies stay deployable with the baseline ``deploy.yaml``.
"""

from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion.mdp.amp_observations import AmpObsCfg

from .running_env_cfg import ObservationsCfg as BaseObservationsCfg
from .running_env_cfg import RobotEnvCfg as BaseRobotEnvCfg
from .running_env_cfg import RobotPlayEnvCfg as BaseRobotPlayEnvCfg


@configclass
class ObservationsCfg(BaseObservationsCfg):
    """Baseline policy/critic observations plus the discriminator group."""

    amp: AmpObsCfg = AmpObsCfg()


@configclass
class RobotEnvCfg(BaseRobotEnvCfg):
    observations: ObservationsCfg = ObservationsCfg()


@configclass
class RobotPlayEnvCfg(BaseRobotPlayEnvCfg):
    observations: ObservationsCfg = ObservationsCfg()
