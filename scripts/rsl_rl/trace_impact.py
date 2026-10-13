# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record the foot contact force around every touchdown at substep (5 ms) resolution.

Companion to ``eval_impact.py`` (same protocol: flat ground, fixed command speed, warm-up discarded,
envs that terminate inside the window dropped). Instead of per-speed summaries it stores, per
touchdown, the vertical contact force F_z and the force norm |F| of the landing foot from
--pre_ms before to --post_ms after first contact, so the shape of the impact (first peak timing,
settling into the support phase) and the impulse over any window can be examined offline with
``plot_impact_trace.py``.

Touchdown here is resolved to the substep: the first substep with |F| > --contact_force_n after at
least --min_air_ms without contact (same debounce as eval_impact.py, but on the 5 ms grid instead of
the 20 ms policy step). t = 0 ms is that first contact substep; the contact began somewhere in the
5 ms before it.

Output: an .npz with, per speed key ``s<cmd>`` (e.g. ``s2.0``): ``fz`` and ``fn`` arrays of shape
(num_touchdowns, num_samples) in N, ``env`` (env id per touchdown), plus ``t_ms`` (sample times),
``body_weight_n``, ``sim_dt`` and ``speeds``.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Record touchdown force traces of an RSL-RL checkpoint.")
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-Running", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=256, help="Number of environments.")
parser.add_argument("--speeds", type=str, default="2.0", help="Comma separated forward speed commands (m/s).")
parser.add_argument("--warmup_steps", type=int, default=150, help="Steps discarded after reset (transient).")
parser.add_argument("--measure_steps", type=int, default=500, help="Steps recorded per speed.")
parser.add_argument("--pre_ms", type=float, default=20.0, help="Trace length before first contact [ms].")
parser.add_argument("--post_ms", type=float, default=200.0, help="Trace length after first contact [ms].")
parser.add_argument("--min_air_ms", type=float, default=40.0, help="Shortest air gap that separates two stances [ms].")
parser.add_argument("--contact_force_n", type=float, default=1.0, help="|F| above this counts as contact [N].")
parser.add_argument("--terrain", type=str, default="plane", choices=["plane", "generator"])
parser.add_argument("--out", type=str, required=True, help="Output .npz path.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
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
    """Same freeze as eval_impact.py (not imported: that module launches the app at import time)."""
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
    sim_dt = base_env.physics_dt
    sub = int(env_cfg.decimation)
    pre = int(round(args_cli.pre_ms * 1e-3 / sim_dt))
    post = int(round(args_cli.post_ms * 1e-3 / sim_dt))
    min_air = max(1, int(round(args_cli.min_air_ms * 1e-3 / sim_dt)))
    body_weight_n = robot.data.default_mass.sum(dim=1)[0].item() * GRAVITY

    cmd_cfg = env_cfg.commands.base_velocity
    contact_sensor = base_env.scene.sensors[cmd_cfg.contact_sensor_name]
    feet_ids, feet_names = contact_sensor.find_bodies(cmd_cfg.feet_body_names, preserve_order=True)
    print(f"[INFO] sim_dt {sim_dt*1e3:.0f} ms, trace -{pre} .. +{post} substeps, min air {min_air} substeps, feet {feet_names}")

    out = {
        "t_ms": (np.arange(-pre, post) * sim_dt * 1e3).astype(np.float32),
        "body_weight_n": np.float32(body_weight_n),
        "sim_dt": np.float32(sim_dt),
        "contact_force_n": np.float32(args_cli.contact_force_n),
        "min_air_ms": np.float32(args_cli.min_air_ms),
    }
    speeds = [float(s) for s in args_cli.speeds.split(",")]
    out["speeds"] = np.array(speeds, dtype=np.float32)
    window = args_cli.warmup_steps + args_cli.measure_steps

    for speed in speeds:
        _set_command_speed(base_env, speed)
        num_envs = base_env.num_envs
        valid = torch.ones(num_envs, dtype=torch.bool, device=device)
        fz_rec, fn_rec = [], []

        with torch.inference_mode():
            obs, _ = env.reset()
            for step in range(window):
                actions = policy(obs)
                step_out = env.step(actions)
                obs, dones = step_out[0], step_out[2]
                if step < args_cli.warmup_steps:
                    continue
                valid &= ~dones.bool()
                hist = contact_sensor.data.net_forces_w_history[:, :, feet_ids, :].flip(1)  # oldest -> newest
                fz_rec.append(hist[..., 2].clone())
                fn_rec.append(hist.norm(dim=-1))

        fz = torch.cat(fz_rec, dim=1)  # (N, S, feet)
        fn = torch.cat(fn_rec, dim=1)
        S = fz.shape[1]
        contact = fn > args_cli.contact_force_n
        air_run = torch.full(contact.shape[::2], 10**6, dtype=torch.long, device=device)  # (N, feet)
        prev = torch.zeros_like(contact[:, 0])
        env_list, foot_list, s_list = [], [], []
        for s in range(S):
            c = contact[:, s]
            td = c & ~prev & (air_run >= min_air)
            if td.any():
                e, f = td.nonzero(as_tuple=True)
                env_list.append(e)
                foot_list.append(f)
                s_list.append(torch.full_like(e, s))
            air_run = torch.where(c, torch.zeros_like(air_run), air_run + 1)
            prev = c
        e = torch.cat(env_list)
        f = torch.cat(foot_list)
        s0 = torch.cat(s_list)
        keep = valid[e] & (s0 >= pre) & (s0 + post <= S)
        e, f, s0 = e[keep], f[keep], s0[keep]
        idx = s0[:, None] + torch.arange(-pre, post, device=device)[None, :]  # (E, T)
        tr_fz = fz[e[:, None], idx, f[:, None]]
        tr_fn = fn[e[:, None], idx, f[:, None]]
        key = f"s{speed:g}"
        out[f"{key}_fz"] = tr_fz.cpu().numpy().astype(np.float32)
        out[f"{key}_fn"] = tr_fn.cpu().numpy().astype(np.float32)
        out[f"{key}_env"] = e.cpu().numpy().astype(np.int32)
        out[f"{key}_survival"] = np.float32(valid.float().mean().item())
        peak_bw = tr_fn[:, pre:].max(dim=1)[0] / body_weight_n
        print(
            f"cmd {speed:4.2f} | touchdowns {len(e)} | alive {valid.float().mean().item():.2f} | "
            f"peak {peak_bw.mean().item():.2f} BW | J40 {(tr_fn[:, pre:pre+8].sum(dim=1) * sim_dt).mean().item():.2f} Ns"
        )

    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    np.savez(args_cli.out, **out)
    print(f"[INFO] wrote {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
