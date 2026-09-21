# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Score a trained checkpoint on a fixed energy-efficiency protocol.

The protocol is deliberately independent of the reward definition so that policies
trained with different objectives -- pure reward shaping, constrained RL, mimic, or any
mix -- stay directly comparable. For each commanded forward speed the script holds the
command constant, discards a warm-up transient, and then integrates actuator cost over a
fixed window.

Reported per speed (ten columns, the analysis set agreed on 2026-09-21):
    cmd_mps, speed_mps   commanded and achieved planar speed
    pos_power_w          E_mech rate: per-motor positive mechanical power, sum max(tau * qdot, 0) [W]
    cot_pos              CoT_mech = int E_mech dt / (m g d)   [dimensionless]
    torque_sq_raw        E_motor rate: sum tau^2 [Nm^2], motor-heating proxy without k_el
    cot_motor            CoT_motor = int sum tau^2 dt / (m g d)   [Nm s]; relative comparisons on this robot only
    cot_elec             electrical CoT, (sum tau^2 k_el + E_mech) / (m g d) with the k_el estimate from the
                         robot asset; the one number that adds heat and work on a common scale
    track_err            |commanded - achieved| planar velocity
    flight_ratio         fraction of steps with no foot in contact; a walk has none, so a policy that got
                         cheaper by dropping the flight phase shows up here
    survival             fraction of envs that never terminated during the window

Braking work is not counted anywhere: the G1 has no regenerative path, so negative tau * qdot is
assumed dissipated, the same convention as the joint_power reward terms.

Example:
    python scripts/rsl_rl/eval_energy.py --task Unitree-G1-29dof-Running --headless \
        --load_run 2026-09-08_17-13-58_try6 --checkpoint model_25500.pt \
        --out logs/energy_baseline.csv
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate the energy efficiency of an RSL-RL checkpoint.")
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-Running", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=256, help="Number of environments to average over.")
parser.add_argument(
    "--speeds",
    type=str,
    default="0.5,1.0,1.5,2.0,2.5",
    help="Comma separated forward speed commands (m/s) to evaluate.",
)
parser.add_argument("--warmup_steps", type=int, default=150, help="Steps discarded after reset (transient).")
parser.add_argument("--measure_steps", type=int, default=500, help="Steps accumulated per speed.")
parser.add_argument(
    "--terrain",
    type=str,
    default="plane",
    choices=["plane", "generator"],
    help="Flat ground keeps runs comparable; 'generator' reuses the training terrain.",
)
parser.add_argument("--out", type=str, default=None, help="Optional CSV path for the result table.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch


import isaaclab_tasks  # noqa: F401
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_K_EL
from unitree_rl_lab.tasks.locomotion.mdp.joint_coefficients import joint_coefficients
from unitree_rl_lab.algorithms.registry import get_runner_class
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

GRAVITY = 9.81


def _freeze_env_cfg(env_cfg, terrain: str):
    """Strip every source of run-to-run variation that is not the policy itself."""
    # A curriculum would keep moving the command ranges and terrain level under us.
    for term_name in list(vars(env_cfg.curriculum).keys()):
        setattr(env_cfg.curriculum, term_name, None)

    if terrain == "plane":
        env_cfg.scene.terrain.terrain_type = "plane"
        env_cfg.scene.terrain.terrain_generator = None

    # Hold the command for the whole window instead of resampling mid-measurement.
    command_cfg = env_cfg.commands.base_velocity
    command_cfg.resampling_time_range = (1.0e9, 1.0e9)
    command_cfg.rel_standing_envs = 0.0
    command_cfg.heading_command = False
    command_cfg.debug_vis = False
    return env_cfg


def _set_command_speed(env, speed: float):
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    ranges.lin_vel_x = (speed, speed)
    ranges.lin_vel_y = (0.0, 0.0)
    ranges.ang_vel_z = (0.0, 0.0)


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
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
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    device = base_env.device
    dt = base_env.step_dt

    weight_n = robot.data.default_mass.sum(dim=1).to(device) * GRAVITY
    k_el = joint_coefficients(robot, UNITREE_G1_29DOF_K_EL)

    # The feet are the same bodies the training gait metrics use.
    cmd_cfg = env_cfg.commands.base_velocity
    contact_sensor = base_env.scene.sensors[cmd_cfg.contact_sensor_name]
    feet_ids, _ = contact_sensor.find_bodies(cmd_cfg.feet_body_names, preserve_order=True)

    window = args_cli.warmup_steps + args_cli.measure_steps
    max_steps = int(base_env.max_episode_length)
    if window > max_steps:
        raise ValueError(
            f"warmup_steps + measure_steps ({window}) exceeds the episode length ({max_steps} steps). "
            "A time-out reset mid-window would corrupt the energy integral; shorten the window or "
            "raise episode_length_s."
        )

    speeds = [float(s) for s in args_cli.speeds.split(",")]
    rows = []

    for speed in speeds:
        _set_command_speed(base_env, speed)

        num_envs = base_env.num_envs
        pos_energy_j = torch.zeros(num_envs, device=device)
        joule_j = torch.zeros(num_envs, device=device)
        torque_sq_raw_dt = torch.zeros(num_envs, device=device)
        distance_m = torch.zeros(num_envs, device=device)
        speed_sum = torch.zeros(num_envs, device=device)
        track_err_sum = torch.zeros(num_envs, device=device)
        flight_sum = torch.zeros(num_envs, device=device)
        valid = torch.ones(num_envs, dtype=torch.bool, device=device)

        # The reset has to happen inside inference mode: the runner already produced
        # inference tensors, and writing root poses to them from normal mode raises.
        with torch.inference_mode():
            obs, _ = env.reset()

            for step in range(args_cli.warmup_steps + args_cli.measure_steps):
                actions = policy(obs)
                step_out = env.step(actions)
                obs, dones = step_out[0], step_out[2]

                if step < args_cli.warmup_steps:
                    continue

                # An env that terminated inside the window is dropped rather than averaged
                # in, so a policy is never rewarded for falling into a low-power heap.
                valid &= ~dones.bool()

                torque = robot.data.applied_torque
                pos_power = torch.clamp_min(torque * robot.data.joint_vel, 0.0).sum(dim=-1)
                torque_sq_raw = torch.square(torque).sum(dim=-1)
                joule_power = (torch.square(torque) * k_el).sum(dim=-1)
                planar_speed = torch.linalg.norm(robot.data.root_lin_vel_w[:, :2], dim=-1)
                command_xy = base_env.command_manager.get_command("base_velocity")[:, :2]
                is_contact = contact_sensor.data.current_contact_time[:, feet_ids] > 0.0

                pos_energy_j += pos_power * dt
                joule_j += joule_power * dt
                torque_sq_raw_dt += torque_sq_raw * dt
                distance_m += planar_speed * dt
                speed_sum += planar_speed
                track_err_sum += torch.linalg.norm(command_xy - robot.data.root_lin_vel_b[:, :2], dim=-1)
                flight_sum += (~is_contact).all(dim=1).float()

        n = float(args_cli.measure_steps)
        survival = valid.float().mean().item()
        if valid.sum() == 0:
            print(f"[WARN] every env terminated at {speed:.2f} m/s; skipping.")
            continue

        def m(tensor):
            return tensor[valid].mean().item()

        mgd = weight_n * distance_m.clamp_min(1e-6)
        row = {
            "cmd_mps": speed,
            "speed_mps": m(speed_sum) / n,
            "pos_power_w": m(pos_energy_j) / (n * dt),
            "cot_pos": m(pos_energy_j / mgd),
            "torque_sq_raw": m(torque_sq_raw_dt) / (n * dt),
            "cot_motor": m(torque_sq_raw_dt / mgd),
            "cot_elec": m((joule_j + pos_energy_j) / mgd),
            "track_err": m(track_err_sum) / n,
            "flight_ratio": m(flight_sum) / n,
            "survival": survival,
        }
        rows.append(row)
        print(
            f"cmd {row['cmd_mps']:4.2f} | act {row['speed_mps']:5.2f} m/s | "
            f"P+ {row['pos_power_w']:6.1f} W | CoT_mech {row['cot_pos']:5.3f} | "
            f"tau^2 {row['torque_sq_raw']:7.0f} | CoT_motor {row['cot_motor']:6.1f} | CoT_el {row['cot_elec']:5.3f} | "
            f"err {row['track_err']:4.2f} | flight {row['flight_ratio']:.2f} | alive {row['survival']:.2f}"
        )

    if rows:
        print("\n=== summary ===")
        header = list(rows[0].keys())
        print(",".join(header))
        for row in rows:
            print(",".join(f"{row[k]:.4f}" for k in header))
        if args_cli.out:
            os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
            with open(args_cli.out, "w") as f:
                f.write(",".join(header) + "\n")
                for row in rows:
                    f.write(",".join(f"{row[k]:.6f}" for k in header) + "\n")
            print(f"\n[INFO] wrote {args_cli.out}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
