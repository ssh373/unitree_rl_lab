"""Observation group scored by the AMP discriminator.

The group is declared once here and attached to every task configuration that trains with an
AMP style reward, so all AMP variants (baseline + AMP, energy reward + AMP, ...) hand the
discriminator byte-identical features. The reference motion loader in
``unitree_rl_lab.algorithms.amp`` has to reproduce exactly these terms, in this order, from
the motion npz files -- :data:`AMP_TERM_ORDER` is the contract between the two.

Constraints on the terms:

* Only terms that ``mdp.symmetry._mirror_observation_group`` knows how to mirror, so the
  left-right data augmentation of the running task keeps working unchanged.
* No commands, gait phase or last action: they have no counterpart in the reference data
  and would let the discriminator separate policy from reference trivially.
* No noise: the reference features are clean.
* ``history_length = 2`` turns the group into a flattened ``(s_{t-1}, s_t)`` transition pair
  (oldest first, see :attr:`isaaclab.utils.buffers.CircularBuffer.buffer`), which is what the
  discriminator scores. The observation manager provides this for free, so the rollout
  storage needs no separate next-observation buffer.
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

AMP_TERM_ORDER = ("base_lin_vel", "base_ang_vel", "projected_gravity", "joint_pos_rel", "joint_vel_rel")
"""Term order of :class:`AmpObsCfg`. Part of the feature definition; the reference loader depends on it."""

AMP_HISTORY_LENGTH = 2


@configclass
class AmpObsCfg(ObsGroup):
    """Discriminator features: base velocities and gravity in the base frame, joint state."""

    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
    projected_gravity = ObsTerm(func=mdp.projected_gravity)
    joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = AMP_HISTORY_LENGTH
        self.flatten_history_dim = True
