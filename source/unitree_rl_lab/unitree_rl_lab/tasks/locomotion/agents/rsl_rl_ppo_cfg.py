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
    clip_speed_ranges: list[tuple[float, float]] | None = None
    """Optional per-clip speed gate, one ``(lo, hi)`` per entry of ``motion_files``; ``None`` applies
    ``speed_range`` to every clip. Lets the reference set be banded by gait (walk takes at low speed, run
    takes at high speed) so speed-matched sampling pairs a walking command with a walk and a running
    command with a run."""
    speed_matched_sampling: bool = True
    speed_match_window: float = 0.1

    reward_weight: float = 2.0
    reward_weight_start: float = 0.0
    ramp_start_iter: int = 0
    ramp_end_iter: int = 0
    
    command_name: str | None = "base_velocity"
    command_gate: float = 0.1
    
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

_LAFAN1_WALK_DIR = os.path.join(os.path.dirname(_LAFAN1_RUN_DIR), "lafan1_walk")
LAFAN1_WALK_CLIPS = [
    os.path.join(_LAFAN1_WALK_DIR, f"{name}.npz")
    for name in (
        "walk1_subject1", "walk1_subject2", "walk1_subject5",
        "walk2_subject1", "walk2_subject3", "walk2_subject4",
        "walk3_subject1", "walk3_subject2", "walk3_subject3", "walk3_subject4", "walk3_subject5",
        "walk4_subject1",
    )
]
"""LAFAN1 walking takes retargeted to G1 (30 fps csv in the same directory; npz built locally with
``csv_to_npz.py --input_fps 30``). Root speed: mean 0.2-0.8 m/s, 48-81% of frames in 0.2-1.0 m/s."""

# TEMP (2026-10-02): speed-banded reference set. Walk takes feed the discriminator below 1.0 m/s, run and
# sprint takes from 1.0 m/s up to the 2.5 m/s command ceiling, so at a walking command the speed-matched
# reference is a human walk and at a running command a human run. Replaces the single run1_subject5 clip,
# which has almost no data below 0.6 m/s (speed matching fell back to nearest neighbours there).
AMP_WALK_SPEED_BAND = (0.2, 1.0)
AMP_RUN_SPEED_BAND = (1.0, 2.5)
LAFAN1_BANDED_CLIPS = LAFAN1_WALK_CLIPS + LAFAN1_RUN_CLIPS
LAFAN1_BANDED_SPEED_RANGES = [AMP_WALK_SPEED_BAND] * len(LAFAN1_WALK_CLIPS) + [AMP_RUN_SPEED_BAND] * len(
    LAFAN1_RUN_CLIPS
)

# TEMP (2026-10-06): low-CoT reference set. Top two takes per band by inverse-dynamics mechanical CoT
# (sum max(tau*qdot, 0) / mgd at matched speed; ranking stable across contact thresholds), plus every
# kimodo clip. Kimodo csv was wxyz and is now stored xyzw like LAFAN1 (originals in kimodo/orig_wxyz/).
_KIMODO_DIR = os.path.join(os.path.dirname(_LAFAN1_RUN_DIR), "kimodo")
LAFAN1_WALK_LOWCOT_CLIPS = [os.path.join(_LAFAN1_WALK_DIR, f"{n}.npz") for n in ("walk3_subject4", "walk1_subject1")]
LAFAN1_RUN_LOWCOT_CLIPS = [os.path.join(_LAFAN1_RUN_DIR, f"{n}.npz") for n in ("run2_subject4", "run1_subject5")]
KIMODO_CLIPS = [os.path.join(_KIMODO_DIR, f"kimodo_{i}.npz") for i in range(1, 5)]
"""Kimodo-generated G1 clips (10 s each, 30 fps csv; npz built with ``csv_to_npz.py --input_fps 30``)."""
AMP_KIMODO_SPEED_BAND = (0.2, 2.5)
LOWCOT_CLIPS = LAFAN1_WALK_LOWCOT_CLIPS + LAFAN1_RUN_LOWCOT_CLIPS + KIMODO_CLIPS
LOWCOT_SPEED_RANGES = (
    [AMP_WALK_SPEED_BAND] * len(LAFAN1_WALK_LOWCOT_CLIPS)
    + [AMP_RUN_SPEED_BAND] * len(LAFAN1_RUN_LOWCOT_CLIPS)
    + [AMP_KIMODO_SPEED_BAND] * len(KIMODO_CLIPS)
)

LAFAN1_RUN_SINGLE_CLIP = [os.path.join(_LAFAN1_RUN_DIR, "run1_subject5.npz")]
"""Single-clip option: the take with the fewest idle frames (3%) and the highest share inside the command
range (89% within 0.2-2.5 m/s). Not the default: with speed-matched sampling the discriminator needs
reference data at every speed the policy visits, and this clip has almost none below 0.6 m/s (5%, mostly
accelerations), whereas the six clips together have 23% there."""


@configclass
class RunningAmpPPORunnerCfg(RunningPPORunnerCfg):

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
        # TEMP (2026-10-06): low-CoT set -- top-2 walk (0.2-1.0) + top-2 run (1.0-2.5) + kimodo (0.2-2.5).
        amp_cfg=RslRlAmpCfg(motion_files=LOWCOT_CLIPS, clip_speed_ranges=LOWCOT_SPEED_RANGES),
        # TEMP (2026-10-02): all walk clips gated to 0.2-1.0 m/s + run/sprint clips gated to 1.0-2.5 m/s.
        # amp_cfg=RslRlAmpCfg(motion_files=LAFAN1_BANDED_CLIPS, clip_speed_ranges=LAFAN1_BANDED_SPEED_RANGES),
        # Single clip (user decision 2026-09-23): run1_subject5, the cleanest take. Low-speed coverage is
        # thin (5% below 0.6 m/s), so speed-matched sampling falls back to nearest neighbours there.
        # amp_cfg=RslRlAmpCfg(motion_files=LAFAN1_RUN_SINGLE_CLIP),
    )
