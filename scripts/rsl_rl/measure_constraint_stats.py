# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure a checkpoint's per-step statistics for every reward term, to set constraint thresholds.

Constrained-RL thresholds (Kim et al., "Not Only Rewards But Also Constraints", T-RO 2024) are
per-step quantities: an *average* constraint bounds ``E[f]`` and a *probabilistic* constraint bounds
``P(f > limit)``. This script rolls a trained policy out under the training command distribution
(full command ranges, 10% standing envs, 10 s resampling -- the play cfg) and reports, for every
reward term's **unweighted** function value and for a few raw diagnostics, the mean and the
percentiles. Two reads of the table:

* mean constraint: ``bound = mean`` (or a fraction of it) keeps the baseline's level;
* probabilistic constraint with the paper's 2.5% allowance: ``limit = p97.5`` makes the baseline
  exactly feasible, anything tighter asks for better-than-baseline behaviour.

Term values are computed by calling each term's function with the reward manager's resolved
parameters, so the numbers are exactly what the reward manager sees before weighting. Steps on which
an env was reset are dropped. Every row is reported twice: over all steps, and over steps whose
planar speed command is at least ``--moving_cmd`` (gait terms are meaningless while standing).

Example:
    python scripts/rsl_rl/measure_constraint_stats.py --task Unitree-G1-29dof-Running --headless \
        --load_run 2026-09-15_10-44-43_baseline --checkpoint model_29999.pt \
        --num_envs 1024 --steps 1500 --out logs/constraint_stats_baseline_29999.csv
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Per-step reward-term statistics of an RSL-RL checkpoint.")
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-Running", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments.")
parser.add_argument("--warmup_steps", type=int, default=100, help="Steps discarded after the initial reset.")
parser.add_argument("--steps", type=int, default=1500, help="Steps accumulated after the warm-up.")
parser.add_argument(
    "--terrain",
    type=str,
    default="plane",
    choices=["plane", "generator"],
    help="Flat ground by default; 'generator' reuses the training terrain.",
)
parser.add_argument("--moving_cmd", type=float, default=0.1, help="Planar command speed that counts as moving.")
parser.add_argument(
    "--fixed_speed",
    type=float,
    default=None,
    help="Hold one forward speed command instead of the training distribution.",
)
parser.add_argument("--out", type=str, default=None, help="Optional CSV path for the result table.")
parser.add_argument(
    "--stochastic",
    action="store_true",
    default=False,
    help="Sample actions from the policy's Gaussian (its trained noise std) instead of the deterministic mean."
    " The constraint extension measures costs on exploration rollouts during training, so bounds meant for it"
    " must be measured the same way; e.g. action_rate carries a noise floor of ~2*std^2*num_joints.",
)
parser.add_argument(
    "--train_cfg",
    action="store_true",
    default=False,
    help="Use the training env cfg (observation noise, pushes, action delay/noise, 4096-env scene defaults) with the"
    " command ranges opened to their limits, instead of the play cfg. Combine with --terrain generator to measure"
    " under the conditions the training-time metrics are logged in.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import csv
import gymnasium as gym
import math
import os
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.algorithms.registry import get_runner_class
from unitree_rl_lab.tasks.locomotion.mdp.constraints import track_ang_vel_z_error, track_lin_vel_xy_error
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

PERCENTILES = (50.0, 90.0, 95.0, 97.5, 99.0)

# Safety-limit definitions mirrored from the sim2real protection (deploy/robots/g1_29dof/config/config.yaml):
# commanded torque is clamped at TORQUE_CLAMP_RATIO x the motor peak (= the training effort limit), and the
# upper body has a 15 rad/s joint-velocity kill threshold. Lower-body joints use the asset velocity limit.
TORQUE_CLAMP_RATIO = 0.97
UPPER_BODY_JOINT_REGEX = r"waist|shoulder|elbow|wrist"
UPPER_BODY_VEL_LIMIT = 15.0


def _freeze_env_cfg(env_cfg, terrain: str):
    """Pin the command distribution and terrain so the measurement is stationary."""
    for term_name in list(vars(env_cfg.curriculum).keys()):
        setattr(env_cfg.curriculum, term_name, None)
    if terrain == "plane":
        env_cfg.scene.terrain.terrain_type = "plane"
        env_cfg.scene.terrain.terrain_generator = None
    if args_cli.fixed_speed is not None:
        cmd = env_cfg.commands.base_velocity
        cmd.resampling_time_range = (1.0e9, 1.0e9)
        cmd.rel_standing_envs = 0.0
        cmd.ranges.lin_vel_x = (args_cli.fixed_speed, args_cli.fixed_speed)
        cmd.ranges.lin_vel_y = (0.0, 0.0)
        cmd.ranges.ang_vel_z = (0.0, 0.0)
    return env_cfg


def _safety_limits(robot) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-joint (torque_limit, velocity_limit) the safety constraints are measured against."""
    import re

    torque_limit = robot.data.joint_effort_limits.clamp_min(1e-6) * TORQUE_CLAMP_RATIO
    vel_limit = robot.data.joint_vel_limits.clamp_min(1e-6).clone()
    upper = torch.tensor([bool(re.search(UPPER_BODY_JOINT_REGEX, n)) for n in robot.joint_names], device=vel_limit.device)
    vel_limit[:, upper] = UPPER_BODY_VEL_LIMIT
    return torque_limit, vel_limit


def _raw_diagnostics(base_env, robot, contact_sensor, feet_ids, body_weight_n) -> dict[str, torch.Tensor]:
    """Continuous quantities behind the probabilistic constraints, in physical units."""
    torque_limit, safety_vel_limit = _safety_limits(robot)
    # Commanded (pre-clipping) torque: the simulator clips applied_torque at the effort limit, so only the
    # computed torque shows how far past the limit the controller asked.
    cmd_ratio = robot.data.computed_torque.abs() / torque_limit
    vel_ratio_safety = robot.data.joint_vel.abs() / safety_vel_limit
    gravity_b = robot.data.projected_gravity_b
    tilt_deg = torch.rad2deg(torch.acos((-gravity_b[:, 2]).clamp(-1.0, 1.0)))
    vel_limit = robot.data.joint_vel_limits.clamp_min(1e-6)
    effort_limit = robot.data.joint_effort_limits.clamp_min(1e-6)
    soft_lo = robot.data.soft_joint_pos_limits[..., 0]
    soft_hi = robot.data.soft_joint_pos_limits[..., 1]
    q = robot.data.joint_pos
    pos_excess = (soft_lo - q).clamp_min(0.0) + (q - soft_hi).clamp_min(0.0)
    foot_force = contact_sensor.data.net_forces_w_history[:, :, feet_ids, :].norm(dim=-1)  # (N, H, feet)
    return {
        "cost/track_lin_vel_xy_error": track_lin_vel_xy_error(base_env, "base_velocity"),
        "cost/track_ang_vel_z_error": track_ang_vel_z_error(base_env, "base_velocity"),
        "raw/tilt_deg": tilt_deg,
        "raw/base_height_m": robot.data.root_pos_w[:, 2],
        "raw/lin_vel_z_abs": robot.data.root_lin_vel_b[:, 2].abs(),
        "raw/joint_vel_ratio_max": (robot.data.joint_vel.abs() / vel_limit).amax(dim=-1),
        "raw/joint_torque_ratio_max": (robot.data.applied_torque.abs() / effort_limit).amax(dim=-1),
        "raw/joint_pos_excess_max_rad": pos_excess.amax(dim=-1),
        "raw/foot_force_peak_bw": foot_force.amax(dim=1).amax(dim=1) / body_weight_n,
        # safety limits (sim2real definitions): max ratio over joints, violation indicator, summed excess
        "safety/cmd_torque_ratio_max": cmd_ratio.amax(dim=-1),
        "safety/cmd_torque_over_limit": (cmd_ratio > 1.0).any(dim=-1).float(),
        "safety/cmd_torque_excess_sum": (cmd_ratio - 1.0).clamp_min(0.0).sum(dim=-1),
        "safety/joint_vel_ratio_max": vel_ratio_safety.amax(dim=-1),
        "safety/joint_vel_over_limit": (vel_ratio_safety > 1.0).any(dim=-1).float(),
        "safety/joint_vel_excess_sum": (vel_ratio_safety - 1.0).clamp_min(0.0).sum(dim=-1),
        "raw/feet_in_contact": (contact_sensor.data.current_contact_time[:, feet_ids] > 0.0).sum(dim=1).float(),
    }


def _stats(values: torch.Tensor) -> dict[str, float]:
    if values.numel() == 0:
        return {}
    q = torch.quantile(values, torch.tensor([p / 100.0 for p in PERCENTILES], device=values.device))
    out = {
        "count": float(values.numel()),
        "mean": values.mean().item(),
        "std": values.std().item() if values.numel() > 1 else 0.0,
        "frac_positive": (values > 0.0).float().mean().item(),
        "max": values.max().item(),
    }
    for p, v in zip(PERCENTILES, q.tolist()):
        out[f"p{p:g}"] = v
    return out


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="env_cfg_entry_point" if args_cli.train_cfg else "play_env_cfg_entry_point",
    )
    if args_cli.train_cfg:
        # The play cfg already opens the command ranges; the training cfg starts them narrow for the curriculum.
        env_cfg.commands.base_velocity.ranges = env_cfg.commands.base_velocity.limit_ranges
    env_cfg = _freeze_env_cfg(env_cfg, args_cli.terrain)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint and os.path.sep in args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_class = get_runner_class(getattr(agent_cfg, "class_name", "OnPolicyRunner"))
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    if args_cli.stochastic:
        actor = runner.alg.policy
        std = actor.action_std if hasattr(actor, "action_std") else None

        def policy(obs):
            return actor.act(obs).detach()

        print(f"[INFO] stochastic policy: sampling with the trained noise std (mean {float(actor.std.mean()) if hasattr(actor, 'std') else 'n/a'})")
    else:
        policy = runner.get_inference_policy(device=env.unwrapped.device)

    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    reward_manager = base_env.reward_manager
    term_names = list(reward_manager.active_terms)
    term_cfgs = {name: reward_manager.get_term_cfg(name) for name in term_names}

    cmd_cfg = env_cfg.commands.base_velocity
    contact_sensor = base_env.scene.sensors[cmd_cfg.contact_sensor_name]
    feet_ids, _ = contact_sensor.find_bodies(cmd_cfg.feet_body_names, preserve_order=True)
    body_weight_n = robot.data.default_mass.sum(dim=1).to(base_env.device) * 9.81

    samples: dict[str, list[torch.Tensor]] = {}
    moving_masks: list[torch.Tensor] = []

    with torch.inference_mode():
        obs, _ = env.reset()
        for step in range(args_cli.warmup_steps + args_cli.steps):
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if step < args_cli.warmup_steps:
                continue
            keep = ~dones.bool()
            command = base_env.command_manager.get_command("base_velocity")
            moving = torch.linalg.norm(command[:, :2], dim=-1) >= args_cli.moving_cmd
            moving_masks.append(moving[keep].cpu())

            values = {}
            for name in term_names:
                cfg = term_cfgs[name]
                values[f"reward/{name}"] = cfg.func(base_env, **cfg.params)
            values.update(_raw_diagnostics(base_env, robot, contact_sensor, feet_ids, body_weight_n))
            for key, value in values.items():
                samples.setdefault(key, []).append(value.view(-1)[keep].float().cpu())

    moving_all = torch.cat(moving_masks)
    rows = []
    header = f"{'quantity':38s} {'split':7s} {'mean':>10s} {'p50':>10s} {'p90':>10s} {'p97.5':>10s} {'p99':>10s} {'max':>10s} {'frac>0':>7s}"
    print(header)
    print("-" * len(header))
    for key, chunks in samples.items():
        all_values = torch.cat(chunks)
        for split, values in (("all", all_values), ("moving", all_values[moving_all])):
            s = _stats(values)
            if not s:
                continue
            rows.append({"quantity": key, "split": split, **s})
            print(
                f"{key:38s} {split:7s} {s['mean']:10.4g} {s['p50']:10.4g} {s['p90']:10.4g} "
                f"{s['p97.5']:10.4g} {s['p99']:10.4g} {s['max']:10.4g} {s['frac_positive']:7.3f}"
            )

    weights = {name: term_cfgs[name].weight for name in term_names}
    print("\nreward weights (for reference):", weights)
    print(f"moving fraction of kept steps: {moving_all.float().mean().item():.3f}")

    if args_cli.out:
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
        fieldnames = ["quantity", "split", "count", "mean", "std", "frac_positive", "max"] + [
            f"p{p:g}" for p in PERCENTILES
        ]
        with open(args_cli.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[INFO] wrote {args_cli.out}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
