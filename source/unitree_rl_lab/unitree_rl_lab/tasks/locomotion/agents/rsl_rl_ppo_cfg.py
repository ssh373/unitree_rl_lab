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
    speed_range: tuple[float, float] = (0.2, 2.5)  # 2026-10-15: back to 0.2, idle frames out (see amp_edit_noRSI)
    """Default per-transition speed gate. Lower bound 0.0 since 2026-10-13 so standing frames count as
    reference (AMP paper samples the whole dataset); per-clip bands below override it."""
    clip_speed_ranges: list[tuple[float, float]] | None = None
    """Optional per-clip speed gate, one ``(lo, hi)`` per entry of ``motion_files``; ``None`` applies
    ``speed_range`` to every clip. Lets the reference set be banded by gait (walk takes at low speed, run
    takes at high speed) so speed-matched sampling pairs a walking command with a walk and a running
    command with a run."""
    speed_matched_sampling: bool = True
    speed_match_window: float = 0.1

    reward_weight: float = 2.0  # 2026-10-15: 3.0 made style ~1.4/s vs tracking 1.5/s; curriculum stalled at 2.2
    """Style reward weight in task-reward units. 2.0 (= track_lin_vel_xy) until 2026-10-13; 3.0 moves the
    style term towards the AMP paper's equal blend with the task reward without reaching the 09-21
    regime (effective 10) where the policy stood still for style reward."""
    task_reward_scale: float = 1.0
    """Multiplier on the task reward before the style term is added (AMP paper w_G; TUM uses 0.7)."""
    reward_weight_start: float = 0.0
    ramp_start_iter: int = 0
    ramp_end_iter: int = 0
    
    command_name: str | None = "base_velocity"
    command_gate: float = 0.1  # 2026-10-15: no style reward below a 0.1 m/s command (was 0.0)
    """Commands below this get no style reward. 0.0 since 2026-10-13: standing frames are in the reference
    set, so standing environments are scored against standing references (speed matching pairs them)."""
    
    hidden_dims: tuple[int, ...] = (1024, 512)
    activation: str = "relu"
    normalize_input: bool = True
    grad_penalty_coef: float = 10.0
    logit_reg_coef: float = 0.05
    """L2 on the discriminator's logit layer (authors' Isaac Gym implementation: 0.05)."""
    weight_decay: float = 1.0e-4
    """Discriminator optimizer weight decay (authors' implementation: 1e-4)."""
    learning_rate: float = 1.0e-4
    num_updates_per_iter: int = 20
    """Discriminator minibatches per PPO update; 20 = num_learning_epochs x num_mini_batches, i.e. one
    discriminator step per policy minibatch as in the AMP paper (was 10)."""
    batch_size: int = 16384
    """Per minibatch, each side (was 4096)."""
    replay_buffer_size: int = 100_000
    """Policy transitions kept across rollouts for the discriminator (AMP paper replay buffer)."""
    use_symmetry_augmentation: bool = True


# Lagrangian constraint solver (2026-10-14). *What* is constrained and by how much lives in the task's
# ConstraintsCfg (env cfg, next to the rewards); only the multiplier dynamics are set here. Costs reach
# the penalty divided by their bound, so lambda is comparable across terms: lambda = 1 charges one
# reward-weight unit per "one bound's worth" of cost (the baseline's penalty weights span 0.05-10).
# 2026-10-14 run 17-49-26: tracking as ungated constraints, lambda_max 10 -> policy learned to die in 3 steps.
# Run 18-31-38: additive steps of 0.01 on all 16 terms reached 10-200x the baseline weights within 100 iters of
# the gate opening (penalty 7.6/step) -> same collapse. Hence baseline-equivalent lambda_init + multiplicative steps.
CONSTRAINT_UPDATE_RULE = "multiplicative"  # lambda *= exp(lr * clip(J-1)); see the extension docstring (2026-10-14)
CONSTRAINT_LAMBDA_LR = 1.0e-2  # 1% per iteration per unit of clipped relative violation: 100 violated iters -> x2.7
CONSTRAINT_VIOLATION_CLIP = 1.0  # relative violation clipped to +-1 before the dual step
CONSTRAINT_LAMBDA_MAX_RATIO = 3.0  # per-term cap = 3 x lambda_init; 10x cancelled the task reward (run 10-15_20-14-53)
CONSTRAINT_LAMBDA_MIN_RATIO = 0.1  # floor = 0.1 x lambda_init so a satisfied term can still come back
CONSTRAINT_COST_CLIP = 100.0  # per-step normalized cost clipped at 100 bounds (glitch guard; max seen in training ~47)
CONSTRAINT_GATE = "standing"  # or "curriculum" (velocity range at its limit) / "none"
CONSTRAINT_LAMBDA_MAX = 2.0  # absolute cap, only for terms that start at 0 (none in the current task)


@configclass
class RslRlConstraintCfg:
    """Lagrangian solver settings, consumed by
    :class:`unitree_rl_lab.algorithms.constraints.extension.LagrangianPenaltyExtension`."""

    update_rule: str = CONSTRAINT_UPDATE_RULE
    lambda_lr: float = CONSTRAINT_LAMBDA_LR
    lambda_max: float = CONSTRAINT_LAMBDA_MAX
    lambda_max_ratio: float | None = CONSTRAINT_LAMBDA_MAX_RATIO
    lambda_min_ratio: float | None = CONSTRAINT_LAMBDA_MIN_RATIO
    violation_clip: float | None = CONSTRAINT_VIOLATION_CLIP
    cost_clip: float | None = CONSTRAINT_COST_CLIP
    gate: str = CONSTRAINT_GATE
    """Clip the relative violation to +-this before the dual step (None: no clip); see the extension."""
    start_after_standing: bool = True
    """Hold the gated multipliers at their lambda_init until the policy stands (same test as the reward gate)."""
    ep_len_threshold: float = 500.0


@configclass
class RslRlExtendedPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO algorithm configuration with composable extensions (see ``unitree_rl_lab.algorithms``)."""

    class_name: str = "ExtendedPPO"
    amp_cfg: RslRlAmpCfg | None = None
    constraint_cfg: RslRlConstraintCfg | None = None


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
# Walk band from 0.0 (2026-10-13): the idle stretches of the walk takes -- a person standing about -- are
# the standing reference the task needs at zero command. Run takes stay gated from 1.0 m/s.
AMP_WALK_SPEED_BAND = (0.2, 1.0)  # 2026-10-15: lower bound back to 0.2 (idle frames out)
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


@configclass
class RunningConstrainedPPORunnerCfg(RunningPPORunnerCfg):
    """Constrained running task: task rewards + energy reward, every other penalty as a constraint
    (see running_constrained_env_cfg.py); this cfg only sets the multiplier dynamics."""

    class_name: str = "ExtendedOnPolicyRunner"
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
        constraint_cfg=RslRlConstraintCfg(),
    )


@configclass
class RunningSafetyCrlPPORunnerCfg(RunningConstrainedPPORunnerCfg):
    """Safety-CRL (running_safety_crl_env_cfg.py): three safety constraints on top of the PACE objective.
    With so few terms the penalty cannot outweigh the task reward, so the per-term cap is 10x the start value."""

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        self.algorithm.constraint_cfg.lambda_max_ratio = 10.0

