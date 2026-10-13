"""Event terms specific to the AMP arm.

``reset_from_reference`` is reference state initialisation (RSI): the AMP paper starts every episode
from a state sampled from the reference motions, so the policy begins inside the distribution the
discriminator rewards instead of having to find it from the default pose. Here a fraction
``probability`` of the resets is drawn from the reference frames; the rest keep whatever the earlier
reset terms wrote (default pose plus the usual randomisation), which preserves robustness to
off-reference starts. Place the term after ``reset_base`` / ``reset_robot_joints``: it overwrites their
result for the chosen environments.
"""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_mul, sample_uniform

from unitree_rl_lab.algorithms.amp.data import AmpMotionData

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class reset_from_reference(ManagerTermBase):
    """Reset a random subset of the environments to frames of the AMP reference motions.

    Parameters (``cfg.params``):
        motion_files, speed_range, clip_speed_ranges: the reference set, same arguments as
            :class:`AmpMotionData`, so the frames a reset can start from are exactly the frames the
            discriminator is trained on.
        probability: fraction of resets that use a reference frame (AMP paper: all of them).
        yaw_range: the reference heading is rotated by a uniform yaw in this range (world velocities
            rotate with it, so the state stays consistent).
        xy_range: uniform planar offset added to the environment origin, as ``reset_base`` does.
        height_clearance: added to the clip's root height.
        asset_cfg: the robot.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        params = cfg.params
        self.asset_cfg: SceneEntityCfg = params.get("asset_cfg", SceneEntityCfg("robot"))
        self.asset: Articulation = env.scene[self.asset_cfg.name]
        data = AmpMotionData(
            params["motion_files"],
            speed_range=params.get("speed_range", (0.0, 2.5)),
            device=env.device,
            clip_speed_ranges=params.get("clip_speed_ranges"),
        )
        self.states = data.reset_states()
        num_frames = int(self.states["joint_pos"].shape[0])
        if num_frames == 0:
            raise ValueError("reset_from_reference: the speed gates left no reference frame to reset to.")
        if self.states["joint_pos"].shape[1] != self.asset.num_joints:
            raise ValueError(
                f"reset_from_reference: reference frames have {self.states['joint_pos'].shape[1]} joints, "
                f"the robot has {self.asset.num_joints}."
            )
        print(f"[reset_from_reference] {num_frames} reference frames from {len(data.motion_files)} clip(s)")

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        motion_files: list[str],
        probability: float = 1.0,
        yaw_range: tuple[float, float] = (-math.pi, math.pi),
        xy_range: float = 0.5,
        height_clearance: float = 0.01,
        speed_range: tuple[float, float] = (0.0, 2.5),
        clip_speed_ranges=None,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> None:
        if env_ids.numel() == 0:
            return
        device = env.device
        use = torch.rand(env_ids.shape[0], device=device) < probability
        if not use.any():
            return
        ids = env_ids[use]
        n = ids.shape[0]
        frame = torch.randint(0, self.states["joint_pos"].shape[0], (n,), device=device)

        yaw = sample_uniform(yaw_range[0], yaw_range[1], (n,), device)
        zeros = torch.zeros_like(yaw)
        q_yaw = quat_from_euler_xyz(zeros, zeros, yaw)
        quat = quat_mul(q_yaw, self.states["root_quat"][frame])
        lin_vel = quat_apply(q_yaw, self.states["root_lin_vel_w"][frame])
        ang_vel = quat_apply(q_yaw, self.states["root_ang_vel_w"][frame])

        default_root = self.asset.data.default_root_state[ids]
        pos = default_root[:, 0:3].clone() + env.scene.env_origins[ids]
        pos[:, :2] += sample_uniform(-xy_range, xy_range, (n, 2), device)
        pos[:, 2] = env.scene.env_origins[ids, 2] + self.states["root_pos_z"][frame] + height_clearance

        self.asset.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=ids)
        self.asset.write_root_velocity_to_sim(torch.cat([lin_vel, ang_vel], dim=-1), env_ids=ids)
        self.asset.write_joint_state_to_sim(
            self.states["joint_pos"][frame], self.states["joint_vel"][frame], env_ids=ids
        )
