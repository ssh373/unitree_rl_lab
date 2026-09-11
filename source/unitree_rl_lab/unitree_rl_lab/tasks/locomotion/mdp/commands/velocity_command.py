from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch

from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING


class LoggedUniformLevelVelocityCommand(UniformVelocityCommand):
    """Uniform velocity command with reward-independent TensorBoard diagnostics."""

    def __init__(self, cfg: LoggedUniformLevelVelocityCommandCfg, env):
        super().__init__(cfg, env)

        if self.cfg.debug_vis_env_index is not None and not 0 <= self.cfg.debug_vis_env_index < self.num_envs:
            raise ValueError(
                f"debug_vis_env_index must be in [0, {self.num_envs - 1}], got {self.cfg.debug_vis_env_index}."
            )

        self._tracking_sample_count = torch.zeros(self.num_envs, device=self.device)
        if self.cfg.log_tracking_metrics:
            for metric_name in (
                "velocity/command_abs_x_mps",
                "velocity/actual_abs_x_mps",
                "velocity/error_xy_mps",
            ):
                self.metrics[metric_name] = torch.zeros(self.num_envs, device=self.device)

        self._cot_sample_count = torch.zeros(self.num_envs, device=self.device)
        self._cot_energy_j = torch.zeros(self.num_envs, device=self.device)
        self._cot_distance_m = torch.zeros(self.num_envs, device=self.device)
        if self.cfg.log_energy_metrics:
            # Weight in Newton, used to make the cost of transport dimensionless.
            self._robot_weight_n = self.robot.data.default_mass.sum(dim=1).to(self.device) * 9.81
            for metric_name in (
                "energy/mech_power_w",
                "energy/torque_sq_norm",
                "energy/cost_of_transport",
                "energy/cot_integrated",
            ):
                self.metrics[metric_name] = torch.zeros(self.num_envs, device=self.device)

        self._contact_sensor: ContactSensor | None = None
        if self.cfg.contact_sensor_name is not None:
            if self.cfg.feet_body_names is None:
                raise ValueError("feet_body_names must be set when contact_sensor_name is provided.")
            if self.cfg.gait_period is None:
                raise ValueError("gait_period must be set when gait metrics are enabled.")

            self._contact_sensor = env.scene.sensors[self.cfg.contact_sensor_name]
            body_ids, body_names = self._contact_sensor.find_bodies(self.cfg.feet_body_names, preserve_order=True)
            if len(body_ids) != len(self.cfg.gait_offsets):
                raise ValueError(
                    "The number of gait offsets must match the selected feet: "
                    f"got {len(self.cfg.gait_offsets)} offsets and bodies {body_names}."
                )
            self._feet_body_ids = body_ids
            self._feet_body_names = body_names
            self._gait_offsets = torch.tensor(self.cfg.gait_offsets, device=self.device)
            self._time_since_contact = torch.zeros(self.num_envs, len(body_ids), device=self.device)
            self._has_previous_contact = torch.zeros(
                self.num_envs, len(body_ids), dtype=torch.bool, device=self.device
            )
            self._period_sample_count = torch.zeros(self.num_envs, len(body_ids), device=self.device)

            self.metrics["gait/contact_ratio"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["gait/phase_match_ratio"] = torch.zeros(self.num_envs, device=self.device)
            for body_name in body_names:
                self.metrics[f"gait/{body_name}_period_s"] = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None:
            env_ids = slice(None)

        extras = super().reset(env_ids)
        self._tracking_sample_count[env_ids] = 0.0
        self._cot_sample_count[env_ids] = 0.0
        self._cot_energy_j[env_ids] = 0.0
        self._cot_distance_m[env_ids] = 0.0

        if self._contact_sensor is not None:
            self._time_since_contact[env_ids] = 0.0
            self._has_previous_contact[env_ids] = False
            self._period_sample_count[env_ids] = 0.0

        return extras

    def _update_running_mean(self, metric_name: str, value: torch.Tensor) -> None:
        count = self._tracking_sample_count.clamp_min(1.0)
        metric = self.metrics[metric_name]
        metric += (value - metric) / count

    def _debug_vis_callback(self, event):
        if self.cfg.debug_vis_env_index is None:
            super()._debug_vis_callback(event)
            return

        if not self.robot.is_initialized:
            return

        env_index = self.cfg.debug_vis_env_index
        env_slice = slice(env_index, env_index + 1)
        base_pos_w = self.robot.data.root_pos_w[env_slice].clone()
        base_pos_w[:, 2] += 0.5

        goal_scale, goal_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        current_scale, current_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        self.goal_vel_visualizer.visualize(base_pos_w, goal_quat[env_slice], goal_scale[env_slice])
        self.current_vel_visualizer.visualize(base_pos_w, current_quat[env_slice], current_scale[env_slice])

    def _update_metrics(self):
        super()._update_metrics()

        self._tracking_sample_count += 1.0
        if self.cfg.log_tracking_metrics:
            command_xy = self.vel_command_b[:, :2]
            actual_xy = self.robot.data.root_lin_vel_b[:, :2]
            self._update_running_mean("velocity/command_abs_x_mps", torch.abs(command_xy[:, 0]))
            self._update_running_mean("velocity/actual_abs_x_mps", torch.abs(actual_xy[:, 0]))
            self._update_running_mean("velocity/error_xy_mps", torch.linalg.norm(command_xy - actual_xy, dim=-1))

        if self.cfg.log_energy_metrics:
            torque = self.robot.data.applied_torque
            power = torch.sum(torch.abs(torque * self.robot.data.joint_vel), dim=-1)
            effort_limit = self.robot.data.joint_effort_limits.clamp_min(1e-6)
            self._update_running_mean("energy/mech_power_w", power)
            self._update_running_mean("energy/torque_sq_norm", torch.sum(torch.square(torque / effort_limit), dim=-1))

            # Cost of transport is only meaningful while the robot is actually travelling,
            # so it gets its own sample counter instead of the shared tracking one.
            speed = torch.linalg.norm(self.robot.data.root_lin_vel_w[:, :2], dim=-1)
            moving_fast = speed > self.cfg.cot_min_speed
            zero = torch.zeros_like(speed)
            cot = power / (self._robot_weight_n * speed.clamp_min(self.cfg.cot_min_speed))
            self._cot_sample_count += moving_fast.float()
            old_cot = self.metrics["energy/cost_of_transport"]
            self.metrics["energy/cost_of_transport"] = torch.where(
                moving_fast,
                old_cot + (cot - old_cot) / self._cot_sample_count.clamp_min(1.0),
                old_cot,
            )

            # The physically meaningful estimator: energy actually spent over distance
            # actually covered. The running mean above is a time-weighted average of
            # instantaneous P/(m g v), which over-weights slow moments where 1/v blows up,
            # so it reads systematically high and does not match eval_energy.py. Both are
            # logged: this one is the number to compare across runs and against the
            # offline sweep, the other is kept so existing curves stay continuous.
            step_dt = self._env.step_dt
            self._cot_energy_j += torch.where(moving_fast, power * step_dt, zero)
            self._cot_distance_m += torch.where(moving_fast, speed * step_dt, zero)
            has_distance = self._cot_distance_m > 1e-3
            self.metrics["energy/cot_integrated"] = torch.where(
                has_distance,
                self._cot_energy_j / (self._robot_weight_n * self._cot_distance_m.clamp_min(1e-3)),
                zero,
            )

        if self._contact_sensor is None:
            return

        contact_time = self._contact_sensor.data.current_contact_time[:, self._feet_body_ids]
        is_contact = contact_time > 0.0
        first_contact = self._contact_sensor.compute_first_contact(self._env.step_dt)[:, self._feet_body_ids]
        moving = torch.linalg.norm(self.vel_command_b, dim=1) > 0.05
        first_contact &= moving.unsqueeze(1)

        self._time_since_contact += self._env.step_dt * moving.unsqueeze(1)
        valid_period = first_contact & self._has_previous_contact
        for foot_index, body_name in enumerate(self._feet_body_names):
            valid = valid_period[:, foot_index]
            new_count = self._period_sample_count[:, foot_index] + valid.float()
            old_period = self.metrics[f"gait/{body_name}_period_s"]
            new_period = old_period + (self._time_since_contact[:, foot_index] - old_period) / new_count.clamp_min(1.0)
            self.metrics[f"gait/{body_name}_period_s"] = torch.where(valid, new_period, old_period)
            self._period_sample_count[:, foot_index] = new_count

        self._time_since_contact = torch.where(first_contact, 0.0, self._time_since_contact)
        self._has_previous_contact |= first_contact

        global_phase = (
            (self._env.episode_length_buf * self._env.step_dt) % self.cfg.gait_period / self.cfg.gait_period
        )
        expected_contact = (
            (global_phase.unsqueeze(1) + self._gait_offsets).remainder(1.0) < self.cfg.gait_stance_ratio
        )
        phase_match = (expected_contact == is_contact).float().mean(dim=1)

        stationary = ~moving
        self._time_since_contact[stationary] = 0.0
        self._has_previous_contact[stationary] = False

        self._update_running_mean("gait/contact_ratio", is_contact.float().mean(dim=1))
        self._update_running_mean("gait/phase_match_ratio", phase_match)


@configclass
class LoggedUniformLevelVelocityCommandCfg(UniformLevelVelocityCommandCfg):
    class_type: type = LoggedUniformLevelVelocityCommand

    log_tracking_metrics: bool = False
    log_energy_metrics: bool = False
    cot_min_speed: float = 0.3
    debug_vis_env_index: int | None = None
    contact_sensor_name: str | None = None
    feet_body_names: str | list[str] | None = None
    gait_period: float | None = None
    gait_offsets: tuple[float, ...] = (0.0, 0.5)
    gait_stance_ratio: float = 0.5
