# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Score a trained checkpoint on a fixed foot-impact protocol.

Reward-independent companion to ``eval_energy.py``: same protocol (flat ground, fixed command
speed, warm-up discarded, fixed measurement window, envs that terminate inside the window are
dropped) so policies trained with different objectives stay directly comparable.

The contact sensor is sampled every physics substep (sim.dt = 5 ms) and its history is widened
to one policy step (decimation = 4 substeps), so each policy step sees the net contact force on
each foot at all four substeps. Definitions, per foot:

    in contact         |F| > 1 N at any substep of the policy step (the sensor's own threshold)
    touchdown          the first in-contact step after at least --min_air_ms (default 40 ms =
                       2 policy steps) without contact. Shorter gaps are contact flicker (the sole
                       bouncing or rolling during one stance) and do not open a new stance.
    impact window      the touchdown step plus the following steps, --impact_window_ms long
                       (default 40 ms = 2 policy steps)
    J_impact [N s]     int |F| dt over the impact window: sum over substeps of |F| * sim.dt.
                       This is the landing impulse -- the momentum the ground has to absorb
                       to stop the falling leg -- not the whole-stance impulse (which is fixed
                       at ~ m g / cadence by momentum balance and says nothing about impact).
    F_peak [BW]        max |F| over the impact window, divided by body weight m g
    v_foot_td [m/s]    vertical velocity of the foot link at the end of the policy step before
                       touchdown (negative = falling); J_impact ~ m_eff * |v_foot_td|
    F_stance [BW]      max |F| from one touchdown to the next of the same foot (captures the
                       active push-off peak as well as the landing peak)
    contact_time [s]   number of in-contact policy steps between two touchdowns, times step_dt

Reported per speed:
    cmd_mps, speed_mps      commanded and achieved planar speed
    touchdowns_per_s        both feet together
    impact_impulse_ns       mean J_impact per touchdown
    impact_impulse_per_m    sum of J_impact over the window / distance travelled [N s / m]
    peak_grf_bw             mean F_peak per touchdown
    foot_vz_td              mean v_foot_td per touchdown
    stance_peak_bw          mean F_stance per stance
    contact_time_s          mean stance duration
    flight_ratio            fraction of steps with no foot in contact
    track_err, survival     as in eval_energy.py

Example:
    python scripts/rsl_rl/eval_impact.py --task Unitree-G1-29dof-Running --headless \
        --load_run 2026-09-15_10-44-43_baseline --checkpoint model_29999.pt \
        --out logs/impact_baseline_29999.csv
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate foot impact of an RSL-RL checkpoint.")
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
parser.add_argument("--impact_window_ms", type=float, default=40.0, help="Impact window after touchdown [ms].")
parser.add_argument("--min_air_ms", type=float, default=40.0, help="Shortest air gap that separates two stances [ms].")
parser.add_argument("--contact_force_n", type=float, default=1.0, help="|F| above this counts as contact [N].")
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
import math
import os
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.algorithms.registry import get_runner_class
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

GRAVITY = 9.81


def _freeze_env_cfg(env_cfg, terrain: str):
    """Strip every source of run-to-run variation that is not the policy itself (as eval_energy.py)."""
    for term_name in list(vars(env_cfg.curriculum).keys()):
        setattr(env_cfg.curriculum, term_name, None)

    if terrain == "plane":
        env_cfg.scene.terrain.terrain_type = "plane"
        env_cfg.scene.terrain.terrain_generator = None

    command_cfg = env_cfg.commands.base_velocity
    command_cfg.resampling_time_range = (1.0e9, 1.0e9)
    command_cfg.rel_standing_envs = 0.0
    command_cfg.heading_command = False
    command_cfg.debug_vis = False

    # One policy step of force history, sampled every physics substep, so the impulse integral
    # covers every substep. Observations do not read the history, so inference is unchanged.
    env_cfg.scene.contact_forces.history_length = int(env_cfg.decimation)
    env_cfg.scene.contact_forces.update_period = env_cfg.sim.dt
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
    sim_dt = base_env.physics_dt
    window_steps = max(1, int(math.ceil(args_cli.impact_window_ms * 1e-3 / dt - 1e-6)))
    min_air_steps = max(1, int(math.ceil(args_cli.min_air_ms * 1e-3 / dt - 1e-6)))
    print(
        f"[INFO] step_dt {dt*1e3:.0f} ms, sim_dt {sim_dt*1e3:.0f} ms, impact window {window_steps} steps, "
        f"min air gap {min_air_steps} steps, contact threshold {args_cli.contact_force_n} N"
    )

    weight_n = robot.data.default_mass.sum(dim=1).to(device) * GRAVITY

    cmd_cfg = env_cfg.commands.base_velocity
    contact_sensor = base_env.scene.sensors[cmd_cfg.contact_sensor_name]
    feet_ids, feet_names = contact_sensor.find_bodies(cmd_cfg.feet_body_names, preserve_order=True)
    robot_feet_ids, _ = robot.find_bodies(cmd_cfg.feet_body_names, preserve_order=True)
    history = contact_sensor.data.net_forces_w_history.shape[1]
    if history != int(env_cfg.decimation):
        raise RuntimeError(f"contact history {history} != decimation {env_cfg.decimation}")
    print(f"[INFO] feet: {feet_names}, mass {robot.data.default_mass.sum(dim=1)[0].item():.2f} kg")

    window = args_cli.warmup_steps + args_cli.measure_steps
    max_steps = int(base_env.max_episode_length)
    if window > max_steps:
        raise ValueError(f"warmup_steps + measure_steps ({window}) exceeds the episode length ({max_steps}).")

    speeds = [float(s) for s in args_cli.speeds.split(",")]
    rows = []
    num_envs = base_env.num_envs
    num_feet = len(feet_ids)

    for speed in speeds:
        _set_command_speed(base_env, speed)

        z = lambda *shape: torch.zeros(num_envs, *shape, device=device)  # noqa: E731
        distance_m, speed_sum, track_err_sum, flight_sum = z(), z(), z(), z()
        n_td, n_closed, n_stance = z(), z(), z()
        impulse_sum, peak_sum, vz_td_sum, stance_peak_sum, contact_steps_sum = z(), z(), z(), z(), z()
        win = torch.zeros(num_envs, num_feet, dtype=torch.long, device=device)
        air_steps = torch.full((num_envs, num_feet), 10**6, dtype=torch.long, device=device)
        prev_contact = torch.zeros(num_envs, num_feet, dtype=torch.bool, device=device)
        have_td = torch.zeros(num_envs, num_feet, dtype=torch.bool, device=device)
        ev_peak = z(num_feet)
        stance_max = z(num_feet)
        stance_steps = z(num_feet)
        prev_foot_vz = z(num_feet)
        valid = torch.ones(num_envs, dtype=torch.bool, device=device)

        with torch.inference_mode():
            obs, _ = env.reset()

            for step in range(window):
                actions = policy(obs)
                step_out = env.step(actions)
                obs, dones = step_out[0], step_out[2]

                foot_vz = robot.data.body_lin_vel_w[:, robot_feet_ids, 2]
                fnorm = contact_sensor.data.net_forces_w_history[:, :, feet_ids, :].norm(dim=-1)  # (N, H, feet)
                step_peak = fnorm.max(dim=1)[0]
                contact = step_peak > args_cli.contact_force_n
                # debounced touchdown: contact begins after a real air gap
                touchdown = contact & ~prev_contact & (air_steps >= min_air_steps)
                air_steps = torch.where(contact, torch.zeros_like(air_steps), air_steps + 1)

                if step < args_cli.warmup_steps:
                    prev_foot_vz = foot_vz.clone()
                    prev_contact = contact
                    # warm up the stance tracker too, so the first stance in the window is complete
                    stance_max = torch.where(touchdown, torch.zeros_like(stance_max), stance_max)
                    stance_steps = torch.where(touchdown, torch.zeros_like(stance_steps), stance_steps)
                    have_td |= touchdown
                    stance_max = torch.where(contact, torch.maximum(stance_max, step_peak), stance_max)
                    stance_steps += contact.float()
                    continue

                valid &= ~dones.bool()
                step_impulse = fnorm.sum(dim=1) * sim_dt

                # -- close the previous stance of this foot at the new touchdown
                closing_stance = touchdown & have_td
                stance_peak_sum += (stance_max * closing_stance).sum(dim=1)
                contact_steps_sum += (stance_steps * closing_stance).sum(dim=1)
                n_stance += closing_stance.sum(dim=1).float()
                stance_max = torch.where(touchdown, torch.zeros_like(stance_max), stance_max)
                stance_steps = torch.where(touchdown, torch.zeros_like(stance_steps), stance_steps)
                have_td |= touchdown
                stance_max = torch.where(contact, torch.maximum(stance_max, step_peak), stance_max)
                stance_steps += contact.float()

                # -- touchdown: open the impact window, note the pre-impact foot velocity
                win = torch.where(touchdown, torch.full_like(win, window_steps), win)
                ev_peak = torch.where(touchdown, torch.zeros_like(ev_peak), ev_peak)
                n_td += touchdown.sum(dim=1).float()
                vz_td_sum += (prev_foot_vz * touchdown).sum(dim=1)

                # -- inside the window: integrate |F| dt, track the peak
                active = win > 0
                impulse_sum += (step_impulse * active).sum(dim=1)
                ev_peak = torch.where(active, torch.maximum(ev_peak, step_peak), ev_peak)
                win = (win - active.long()).clamp_min(0)
                closing = active & (win == 0)
                peak_sum += (ev_peak * closing).sum(dim=1)
                n_closed += closing.sum(dim=1).float()

                planar_speed = torch.linalg.norm(robot.data.root_lin_vel_w[:, :2], dim=-1)
                command_xy = base_env.command_manager.get_command("base_velocity")[:, :2]
                distance_m += planar_speed * dt
                speed_sum += planar_speed
                track_err_sum += torch.linalg.norm(command_xy - robot.data.root_lin_vel_b[:, :2], dim=-1)
                flight_sum += (~contact).all(dim=1).float()
                prev_foot_vz = foot_vz.clone()
                prev_contact = contact

        n = float(args_cli.measure_steps)
        survival = valid.float().mean().item()
        if valid.sum() == 0:
            print(f"[WARN] every env terminated at {speed:.2f} m/s; skipping.")
            continue

        def m(tensor):
            return tensor[valid].mean().item()

        row = {
            "cmd_mps": speed,
            "speed_mps": m(speed_sum) / n,
            "touchdowns_per_s": m(n_td) / (n * dt),
            "impact_impulse_ns": m(impulse_sum / n_closed.clamp_min(1.0)),
            "impact_impulse_per_m": m(impulse_sum / distance_m.clamp_min(1e-6)),
            "peak_grf_bw": m(peak_sum / n_closed.clamp_min(1.0) / weight_n),
            "foot_vz_td": m(vz_td_sum / n_td.clamp_min(1.0)),
            "stance_peak_bw": m(stance_peak_sum / n_stance.clamp_min(1.0) / weight_n),
            "contact_time_s": m(contact_steps_sum / n_stance.clamp_min(1.0)) * dt,
            "flight_ratio": m(flight_sum) / n,
            "track_err": m(track_err_sum) / n,
            "survival": survival,
        }
        rows.append(row)
        print(
            f"cmd {row['cmd_mps']:4.2f} | act {row['speed_mps']:5.2f} m/s | td/s {row['touchdowns_per_s']:4.2f} | "
            f"J_imp {row['impact_impulse_ns']:5.2f} Ns | J/m {row['impact_impulse_per_m']:5.2f} Ns/m | "
            f"Fpk {row['peak_grf_bw']:4.2f} BW | vz_td {row['foot_vz_td']:5.2f} | Fst {row['stance_peak_bw']:4.2f} BW | "
            f"tc {row['contact_time_s']:.3f} s | flight {row['flight_ratio']:.2f} | err {row['track_err']:4.2f} | "
            f"alive {row['survival']:.2f}"
        )

    if args_cli.out and rows:
        import csv

        os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
        with open(args_cli.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[INFO] wrote {args_cli.out}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
