# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)

from unitree_rl_lab.tasks.locomotion.mdp.symmetry import compute_g1_29dof_symmetric_states


@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 100
    experiment_name = ""  # same as task name
    empirical_normalization = False
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class RunningPPORunnerCfg(BasePPORunnerCfg):
    """PPO configuration for G1 running with left-right data augmentation."""

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=False,
            data_augmentation_func=compute_g1_29dof_symmetric_states,
        ),
    )


@configclass
class RslRlAmpCfg:
    """Adversarial Motion Priors settings, consumed by
    :class:`unitree_rl_lab.algorithms.amp.extension.AMPExtension`."""

    motion_files: list[str] = MISSING
    """Reference motion npz files produced by ``scripts/mimic/csv_to_npz.py``."""
    obs_group: str = "amp"
    speed_range: tuple[float, float] = (0.2, 2.5)
    """Planar-speed gate on reference transitions. The upper bound is the command ceiling. The lower bound
    admits everything down to standing: the study compares the arms at 2 m/s only, so a human-like *walk*
    at slow commands is acceptable (decision 2026-09-21). Two earlier floors are documented in the project
    notes: 0.2 taught walking at every speed and stalled the velocity curriculum at 0.9 m/s; 1.0 kept the
    running gait but, once active, pulled ``track_lin_vel_xy`` under the curriculum's 0.8*weight threshold
    and stalled it at 1.3 m/s. Neither reached 2 m/s -- pair any floor with a reward ramp that starts after
    the curriculum has opened. Keep ``command_gate`` at or above this floor."""
    speed_matched_sampling: bool = True
    speed_match_window: float = 0.1
    """Pair every policy transition shown to the discriminator with a reference transition within
    ``±speed_match_window`` m/s of its speed, so the discriminator cannot separate the two sides by speed
    (first run with uniform sampling and weight 0.5 ran at 0.7 m/s regardless of a 0.2 m/s command)."""

    reward_weight: float = 2.0
    """Final weight of the style reward in reward-manager units (the extension multiplies it by ``step_dt``
    like every task reward weight), so 2.0 puts a saturated style reward on a par with ``track_lin_vel_xy``.
    Until 2026-09-22 the dt factor was missing and the old default 0.2 acted as a weight of 10: the style
    term (~4/s) outweighed the whole task reward and policies stood still on the slowest reference frames."""
    reward_weight_start: float = 0.0
    ramp_start_iter: int = 0
    ramp_end_iter: int = 0
    """Linear ramp of the style weight from ``reward_weight_start`` to ``reward_weight`` over PPO iterations.
    Equal start/end iterations disable it (the weight is then ``reward_weight`` throughout). Kept as a plain
    float rather than ``float | None`` so Hydra overrides (``agent.algorithm.amp_cfg.ramp_end_iter=...``) pass
    the config type check."""

    command_name: str | None = "base_velocity"
    command_gate: float = 0.1
    """Below this commanded planar speed an environment gets no style reward and its transitions are not
    shown to the discriminator. 0.1 excludes only the standing environments (``rel_standing_envs``), matching
    ``_command_run_gate`` in the task rewards. Must not be below ``speed_range[0]``: a policy transition
    slower than every reference transition would hand the discriminator speed as a cue again."""

    hidden_dims: tuple[int, ...] = (1024, 512)
    activation: str = "relu"
    normalize_input: bool = True
    grad_penalty_coef: float = 10.0
    learning_rate: float = 1.0e-4
    num_updates_per_iter: int = 10
    batch_size: int = 4096
    use_symmetry_augmentation: bool = True


@configclass
class RslRlExtendedPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO algorithm configuration with composable extensions (see ``unitree_rl_lab.algorithms``)."""

    class_name: str = "ExtendedPPO"
    amp_cfg: RslRlAmpCfg | None = None


_LAFAN1_RUN_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "locomotion", "motions", "lafan1_run"
)
LAFAN1_RUN_CLIPS = [
    os.path.join(_LAFAN1_RUN_DIR, f"{name}.npz")
    for name in ("run1_subject2", "run1_subject5", "run2_subject1", "run2_subject4", "sprint1_subject2", "sprint1_subject4")
]
"""LAFAN1 running takes retargeted to G1 (30 fps csv in the same directory; npz built locally)."""

LAFAN1_RUN_SINGLE_CLIP = [os.path.join(_LAFAN1_RUN_DIR, "run1_subject5.npz")]
"""Single-clip option: the take with the fewest idle frames (3%) and the highest share inside the command
range (89% within 0.2-2.5 m/s). Not the default: with speed-matched sampling the discriminator needs
reference data at every speed the policy visits, and this clip has almost none below 0.6 m/s (5%, mostly
accelerations), whereas the six clips together have 23% there."""


@configclass
class RunningAmpPPORunnerCfg(RunningPPORunnerCfg):
    """Running PPO with an AMP style reward from human running motions (ablation B).

    Identical to :class:`RunningPPORunnerCfg` except that the ``amp`` observation set is routed to
    the algorithm and PPO is extended with the AMP discriminator. The policy and the task rewards
    are untouched, so the only difference to the baseline run is the style reward.
    """

    class_name: str = "ExtendedOnPolicyRunner"
    obs_groups = {"policy": ["policy"], "critic": ["critic"], "amp": ["amp"]}
    algorithm = RslRlExtendedPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=False,
            data_augmentation_func=compute_g1_29dof_symmetric_states,
        ),
        amp_cfg=RslRlAmpCfg(motion_files=LAFAN1_RUN_CLIPS),
    )
