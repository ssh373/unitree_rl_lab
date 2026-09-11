# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare the AMP discriminator features of a policy against the reference motions.

The discriminator only works if both sides speak the same language: the ``amp`` observation group
produced by the environment and the features rebuilt from the motion npz files must use the same
units, the same joint order and the same frame conventions. A mismatch does not raise -- the
discriminator would simply separate the two sets on the broken dimension and the style reward
would be noise -- so it has to be checked explicitly, before any AMP training is started.

The two distributions are *not* expected to match: that difference is exactly what the
discriminator is meant to learn. What this script looks for is a mismatch that no amount of
training can fix, such as a block whose scale is off by an order of magnitude (wrong units or a
permuted joint order) or whose sign is flipped (wrong frame convention).

Running the baseline policy rather than an untrained one makes the comparison meaningful: those
are the very transitions the discriminator will be shown on the first iteration of AMP training.

Example:
    python scripts/rsl_rl/check_amp_features.py --headless --num_envs 64 \
        --load_run 2026-09-08_17-13-58_try6 --checkpoint model_25500.pt \
        --motion_files source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/motions/lafan1_run/*.npz
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Compare policy and reference AMP features.")
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-Running-AMP", help="Task with an 'amp' observation group.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to roll out.")
parser.add_argument("--steps", type=int, default=400, help="Rollout steps collected after the warm-up.")
parser.add_argument("--warmup_steps", type=int, default=100, help="Steps discarded so the history buffer fills.")
parser.add_argument("--motion_files", nargs="+", required=True, help="Reference motion npz files.")
parser.add_argument("--speed_range", nargs=2, type=float, default=(0.2, 2.5), metavar=("LO", "HI"),
                    help="Speed gate applied to BOTH the reference clips and the policy rollout.")
parser.add_argument("--random_policy", action="store_true", default=False,
                    help="Skip loading a checkpoint and act randomly (useful before any policy exists).")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.algorithms.amp.data import AMP_TERM_DIMS, AmpMotionData
from unitree_rl_lab.algorithms.registry import get_runner_class

# A block whose spread differs from the reference by more than this is almost certainly a bug
# rather than a difference in behaviour.
SCALE_ALARM = 5.0


def _block_stats(features: torch.Tensor) -> list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Mean, std and max-abs per (term, history sample) block, in observation-group order."""
    rows, offset = [], 0
    for term, dim in AMP_TERM_DIMS:
        for age in ("t-1", "t"):
            block = features[:, offset : offset + dim]
            rows.append((f"{term}({age})", block.mean(), block.std(), block.abs().max()))
            offset += dim
    return rows


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # Flat ground and no pushes: differences should come from the gait, not from the terrain.
    if getattr(env_cfg.scene.terrain, "terrain_generator", None) is not None:
        env_cfg.scene.terrain.terrain_type = "plane"
        env_cfg.scene.terrain.terrain_generator = None
    env_cfg.curriculum = None

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if args_cli.random_policy:
        policy = None
        print("[INFO] Acting randomly (no checkpoint).")
    else:
        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO] Loading checkpoint: {resume_path}")
        runner = get_runner_class(getattr(agent_cfg, "class_name", "OnPolicyRunner"))(
            env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
        )
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

    robot = env.unwrapped.scene["robot"]
    lo, hi = args_cli.speed_range

    collected = []
    obs, _ = env.get_observations(), None
    with torch.inference_mode():
        for step in range(args_cli.warmup_steps + args_cli.steps):
            actions = policy(obs) if policy is not None else torch.randn(
                env.unwrapped.num_envs, env.unwrapped.action_manager.total_action_dim, device=env.unwrapped.device
            )
            obs, _, _, _ = env.step(actions)
            if step < args_cli.warmup_steps:
                continue
            # The same gate the reference data uses, so the two sets describe the same regime.
            speed = torch.linalg.vector_norm(robot.data.root_lin_vel_w[:, :2], dim=-1)
            keep = (speed >= lo) & (speed <= hi)
            if keep.any():
                collected.append(obs["amp"][keep].clone())

    if not collected:
        raise RuntimeError(f"No rollout transition fell inside the speed gate {args_cli.speed_range}.")
    policy_features = torch.cat(collected, dim=0).float()

    reference = AmpMotionData(args_cli.motion_files, speed_range=tuple(args_cli.speed_range))
    print("\n" + reference.describe())
    reference.prepare(robot.data.default_joint_pos[0], robot.data.default_joint_vel[0])
    reference_features = reference.features.float()

    print(f"\npolicy    : {tuple(policy_features.shape)}")
    print(f"reference : {tuple(reference_features.shape)}")
    if policy_features.shape[1] != reference_features.shape[1]:
        raise RuntimeError(
            f"Feature width differs: environment {policy_features.shape[1]} vs loader "
            f"{reference_features.shape[1]}. The loader no longer matches the observation group."
        )

    print(f"\n{'block':<26}{'policy mean':>13}{'ref mean':>11}{'policy std':>12}{'ref std':>10}{'std ratio':>11}")
    alarms = []
    for (name, p_mean, p_std, _), (_, r_mean, r_std, _) in zip(
        _block_stats(policy_features), _block_stats(reference_features)
    ):
        ratio = (p_std / r_std).item() if r_std > 1e-6 else float("inf")
        flag = ""
        if ratio > SCALE_ALARM or ratio < 1.0 / SCALE_ALARM:
            flag = "  <== scale"
            alarms.append(name)
        print(f"  {name:<24}{p_mean:13.3f}{r_mean:11.3f}{p_std:12.3f}{r_std:10.3f}{ratio:11.2f}{flag}")

    print()
    if alarms:
        print(f"FAIL: {len(alarms)} block(s) differ in spread by more than {SCALE_ALARM}x: {', '.join(alarms)}")
        print("      Check joint order, velocity frame, and the default joint position used by the loader.")
    else:
        print(f"OK: every block is within {SCALE_ALARM}x of the reference spread.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
