# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-joint torque diagnostics for a trained checkpoint.

eval_energy.py reports sum (tau / tau_limit)^2 as one number, which says nothing about
*which* joints carry the load or whether any of them are pinned against their effort limit.
This script records every joint's applied torque and reports the distribution per joint.

Torque is captured at the physics rate (200 Hz), not the policy rate (50 Hz): the recorder
hooks scene.write_data_to_sim(), which runs once per decimation substep, so peaks that fall
between control steps are not missed.

Outputs a per-joint table, a normalized torque trace over a few gait cycles, and the raw
array as .npz for further analysis.

Example:
    python scripts/rsl_rl/inspect_torque.py --headless \
        --load_run 2026-09-08_17-13-58_try6 --checkpoint model_25500.pt --speed 2.0
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Inspect per-joint torque of an RSL-RL checkpoint.")
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-Running")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--speed", type=float, default=2.0, help="Commanded forward speed (m/s).")
parser.add_argument("--warmup_steps", type=int, default=150, help="Policy steps discarded after reset.")
parser.add_argument("--record_steps", type=int, default=250, help="Policy steps recorded (x decimation substeps).")
parser.add_argument("--trace_cycles", type=float, default=2.5, help="Gait cycles to draw in the trace plot.")
parser.add_argument("--top", type=int, default=4, help="Joints to draw in the trace plot, by peak load.")
parser.add_argument("--out_prefix", type=str, default="logs/torque")
parser.add_argument("--disable_fabric", action="store_true", default=False)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402

import unitree_rl_lab.tasks  # noqa: F401, E402
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402

# Same categorical slots as plot_energy.py.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e3e2de"

SATURATION_FRACTION = 0.99


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    for term_name in list(vars(env_cfg.curriculum).keys()):
        setattr(env_cfg.curriculum, term_name, None)
    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = None
    command_cfg = env_cfg.commands.base_velocity
    command_cfg.resampling_time_range = (1.0e9, 1.0e9)
    command_cfg.rel_standing_envs = 0.0
    command_cfg.heading_command = False
    command_cfg.debug_vis = False

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint and os.path.sep in args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    joint_names = list(robot.joint_names)
    effort_limit = robot.data.joint_effort_limits[0].clone()
    decimation = base_env.cfg.decimation
    physics_dt = base_env.cfg.sim.dt

    ranges = base_env.command_manager.get_term("base_velocity").cfg.ranges
    ranges.lin_vel_x = (args_cli.speed, args_cli.speed)
    ranges.lin_vel_y = (0.0, 0.0)
    ranges.ang_vel_z = (0.0, 0.0)

    # Hook the decimation loop: write_data_to_sim() runs the actuator model and refreshes
    # applied_torque once per physics substep, so wrapping it captures the full 200 Hz
    # signal instead of the one sample per control step that a reward term would see.
    torque_log: list[torch.Tensor] = []
    joint_vel_log: list[torch.Tensor] = []
    recording = False
    original_write = base_env.scene.write_data_to_sim

    def write_and_record():
        original_write()
        if recording:
            torque_log.append(robot.data.applied_torque.clone())
            joint_vel_log.append(robot.data.joint_vel.clone())

    base_env.scene.write_data_to_sim = write_and_record

    with torch.inference_mode():
        obs, _ = env.reset()
        for step in range(args_cli.warmup_steps + args_cli.record_steps):
            recording = step >= args_cli.warmup_steps
            obs = env.step(policy(obs))[0]

    base_env.scene.write_data_to_sim = original_write

    # (substeps, envs, joints) -> flatten the sample axes for the per-joint statistics.
    torque = torch.stack(torque_log, dim=0)
    joint_vel = torch.stack(joint_vel_log, dim=0)
    expected = args_cli.record_steps * decimation
    if torque.shape[0] != expected:
        print(f"[WARN] recorded {torque.shape[0]} substeps, expected {expected}.")
    abs_torque = torque.abs()
    flat = abs_torque.reshape(-1, abs_torque.shape[-1])

    mean_abs = flat.mean(dim=0)
    p99 = torch.quantile(flat.float(), 0.99, dim=0)
    peak = flat.max(dim=0).values
    saturated = (flat >= SATURATION_FRACTION * effort_limit).float().mean(dim=0)
    load = p99 / effort_limit

    order = torch.argsort(load, descending=True)
    print(f"\n=== per-joint torque at {args_cli.speed:.2f} m/s ===")
    print(f"{'joint':<28s}{'mean|tau|':>10s}{'p99':>9s}{'peak':>9s}{'limit':>8s}{'p99/lim':>9s}{'saturated':>11s}")
    for index in order.tolist():
        print(
            f"{joint_names[index]:<28s}{mean_abs[index]:10.2f}{p99[index]:9.2f}{peak[index]:9.2f}"
            f"{effort_limit[index]:8.1f}{load[index] * 100:8.0f}%{saturated[index] * 100:10.1f}%"
        )

    # Per-joint power attribution. The reward's power term sums these, so this table says
    # which joints a stronger power weight would actually reach -- and, by comparing a
    # joint's power share against its torque share, which ones burn current without doing
    # work (only the torque term can touch those).
    joint_power = (torque * joint_vel).abs().reshape(-1, torque.shape[-1]).mean(dim=0)
    total_power = joint_power.sum()
    torque_sq = torch.square(flat / effort_limit).mean(dim=0)
    total_torque_sq = torque_sq.sum()
    power_share = joint_power / total_power
    torque_share = torque_sq / total_torque_sq

    power_order = torch.argsort(joint_power, descending=True)
    print(f"\n=== per-joint power attribution at {args_cli.speed:.2f} m/s ===")
    print(f"total mechanical power: {total_power:.1f} W")
    print(f"{'joint':<28s}{'power W':>9s}{'P share':>9s}{'tau2 share':>12s}{'ratio':>8s}")
    for index in power_order.tolist():
        if joint_power[index] < 0.5:
            continue
        ratio = torque_share[index] / power_share[index].clamp_min(1e-9)
        print(
            f"{joint_names[index]:<28s}{joint_power[index]:9.1f}{power_share[index] * 100:8.1f}%"
            f"{torque_share[index] * 100:11.1f}%{ratio:8.2f}"
        )
    print(
        "\nratio > 1 means the joint carries more of the torque-squared cost than of the\n"
        "mechanical power: it holds load without doing work, so only joint_torque reaches it."
    )

    total_saturated = saturated.max().item()
    print(
        f"\nworst joint sits at {load[order[0]] * 100:.0f}% of its effort limit (p99); "
        f"max time-at-limit across joints: {total_saturated * 100:.2f}%"
    )

    npz_path = f"{args_cli.out_prefix}_{args_cli.speed:.1f}mps.npz"
    os.makedirs(os.path.dirname(os.path.abspath(npz_path)), exist_ok=True)
    import numpy as np

    np.savez_compressed(
        npz_path,
        torque=torque.cpu().numpy().astype("float32"),
        joint_vel=joint_vel.cpu().numpy().astype("float32"),
        joint_names=np.array(joint_names),
        effort_limit=effort_limit.cpu().numpy(),
        physics_dt=physics_dt,
        decimation=decimation,
        speed=args_cli.speed,
    )
    print(f"[INFO] wrote {npz_path}   shape (substeps, envs, joints) = {tuple(torque.shape)}")

    # --- trace plot: normalized so one saturation line serves every joint ---
    gait_period = 0.5
    trace_substeps = min(int(args_cli.trace_cycles * gait_period / physics_dt), torque.shape[0])
    time_axis = [i * physics_dt for i in range(trace_substeps)]
    top_ids = order[: args_cli.top].tolist()

    fig, ax = plt.subplots(figsize=(11.0, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=0)

    for slot, joint_index in enumerate(top_ids):
        series = (torque[:trace_substeps, 0, joint_index] / effort_limit[joint_index]).cpu().numpy()
        color = SERIES_COLORS[slot % len(SERIES_COLORS)]
        ax.plot(time_axis, series, color=color, linewidth=2.0, label=joint_names[joint_index])
        ax.annotate(
            joint_names[joint_index],
            (time_axis[-1], series[-1]),
            textcoords="offset points",
            xytext=(6, 0),
            fontsize=8,
            color=INK_SECONDARY,
            va="center",
        )

    for level in (1.0, -1.0):
        ax.axhline(level, color=INK_SECONDARY, linewidth=1.0, linestyle="--", zorder=1)
    ax.annotate(
        "effort limit",
        (time_axis[0], 1.0),
        textcoords="offset points",
        xytext=(2, 4),
        fontsize=8,
        color=INK_SECONDARY,
    )

    ax.set_ylim(-1.25, 1.25)
    ax.set_title(
        f"Torque as a fraction of each joint's effort limit  -  env 0, {args_cli.speed:.1f} m/s",
        color=INK_PRIMARY,
        fontsize=11,
        loc="left",
        pad=10,
    )
    ax.set_xlabel("time  [s]", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("tau / tau_limit", color=INK_SECONDARY, fontsize=9)
    legend = ax.legend(frameon=False, fontsize=9, loc="lower left", ncol=min(len(top_ids), 4))
    for text in legend.get_texts():
        text.set_color(INK_PRIMARY)

    fig.tight_layout()
    png_path = f"{args_cli.out_prefix}_{args_cli.speed:.1f}mps.png"
    fig.savefig(png_path, dpi=160, facecolor=SURFACE)
    print(f"[INFO] wrote {png_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
