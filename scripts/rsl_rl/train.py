# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""


import gymnasium as gym
import pathlib
import sys

sys.path.insert(0, f"{pathlib.Path(__file__).parent.parent}")
from list_envs import import_packages  # noqa: F401

sys.path.pop(0)

tasks = []
for task_spec in gym.registry.values():
    if "Unitree" in task_spec.id and "Isaac" not in task_spec.id:
        tasks.append(task_spec.id)

import argparse

import argcomplete

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument(
    "--video_all_envs",
    action="store_true",
    default=False,
    help="Record every environment with a static camera instead of following one robot.",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, choices=tasks, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
argcomplete.autocomplete(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# for distributed training, check minimum supported rsl-rl version
RSL_RL_VERSION = "2.3.1"
installed_version = metadata.version("rsl-rl-lib")
if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import inspect
import os
import shutil
import torch
from datetime import datetime


import isaaclab_tasks  # noqa: F401
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.algorithms.registry import get_runner_class
from unitree_rl_lab.utils.export_deploy_cfg import export_deploy_cfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


class TrackingRecordVideo(gym.wrappers.RecordVideo):
    """Record video while following one asset in x/y with a fixed camera height."""

    def __init__(
        self,
        *args,
        tracked_asset_name: str | None = None,
        tracked_env_index: int = 0,
        camera_xy_offset: tuple[float, float] = (2.5, 2.0),
        camera_height: float = 1.6,
        lookat_height: float = 0.65,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._tracked_asset_name = tracked_asset_name
        self._tracked_env_index = tracked_env_index
        self._camera_xy_offset = camera_xy_offset
        self._camera_height = camera_height
        self._lookat_height = lookat_height

    def _capture_frame(self):
        if self._tracked_asset_name is not None:
            unwrapped_env = self.env.unwrapped
            root_xy = (
                unwrapped_env.scene[self._tracked_asset_name]
                .data.root_pos_w[self._tracked_env_index, :2]
                .detach()
                .cpu()
                .tolist()
            )
            terrain_origin_z = float(unwrapped_env.scene.env_origins[self._tracked_env_index, 2].item())
            camera_eye = (
                root_xy[0] + self._camera_xy_offset[0],
                root_xy[1] + self._camera_xy_offset[1],
                terrain_origin_z + self._camera_height,
            )
            camera_target = (
                root_xy[0],
                root_xy[1],
                terrain_origin_z + self._lookat_height,
            )
            unwrapped_env.sim.set_camera_view(eye=camera_eye, target=camera_target)

        super()._capture_frame()


class WandbRecordVideo(TrackingRecordVideo):
    """Record training videos locally and upload completed clips to W&B."""

    def stop_recording(self):
        video_name = self._video_name
        has_frames = len(self.recorded_frames) > 0
        super().stop_recording()

        if not has_frames or video_name is None:
            return

        import wandb

        if wandb.run is None:
            return

        video_path = os.path.join(self.video_folder, f"{video_name}.mp4")
        wandb.log(
            {"Videos/train": wandb.Video(video_path, format="mp4", caption=video_name)},
            commit=False,
        )


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Detect single-robot recording by capability rather than by task name: any task whose
    # velocity command exposes debug_vis_env_index is one of the running variants and can
    # render one robot with a tracking camera. A hard-coded task string silently dropped
    # this for every new variant of the task (-Reward, -Power, -Torque, ...).
    _command_cfg = getattr(env_cfg.commands, "base_velocity", None)
    record_single_running_env = (
        args_cli.video
        and not args_cli.video_all_envs
        and _command_cfg is not None
        and hasattr(_command_cfg, "debug_vis_env_index")
    )
    if args_cli.video:
        print(
            f"[INFO] Video: {'single robot with tracking camera' if record_single_running_env else 'all environments, static camera'}"
        )

    # The recording wrapper follows the robot in x/y. Keep the built-in viewer static so its
    # asset-root callback does not reintroduce the robot's vertical shake.
    if record_single_running_env:
        env_cfg.viewer.origin_type = "env"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.eye = (2.5, 2.0, 1.6)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.65)
        env_cfg.commands.base_velocity.debug_vis_env_index = 0

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # This way, the Ray Tune workflow can extract experiment name.
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Keep every environment active for physics and learning, but render only environment 0's robot.
    if record_single_running_env and env.unwrapped.num_envs > 1:
        env.unwrapped.scene["robot"].set_visibility(False, env_ids=range(1, env.unwrapped.num_envs))

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        if record_single_running_env:
            video_kwargs.update(
                {
                    "tracked_asset_name": "robot",
                    "tracked_env_index": 0,
                    "camera_xy_offset": (2.5, 2.0),
                    "camera_height": 1.6,
                    "lookat_height": 0.65,
                }
            )
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        video_wrapper = WandbRecordVideo if agent_cfg.logger.lower() == "wandb" else TrackingRecordVideo
        env = video_wrapper(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl; the class is chosen by agent_cfg.class_name (default OnPolicyRunner)
    runner_class = get_runner_class(agent_cfg.class_name)
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    export_deploy_cfg(env.unwrapped, log_dir)
    # copy the environment configuration file to the log directory
    shutil.copy(
        inspect.getfile(env_cfg.__class__),
        os.path.join(log_dir, "params", os.path.basename(inspect.getfile(env_cfg.__class__))),
    )

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
