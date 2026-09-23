from __future__ import annotations

import torch
from typing import TYPE_CHECKING

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from .joint_coefficients import joint_coefficients

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""
Joint penalties.
"""


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def joint_power_l1(
    env: ManagerBasedRLEnv,
    power_ref: float = 300.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Positive mechanical joint power ``sum max(tau * qdot, 0)`` normalized by a reference power.

    Only positive (motoring) work is charged, per motor. Negative work -- the motor braking
    while the joint is driven by gravity or inertia -- is assumed to be dissipated, not
    recovered, so it costs nothing here: the G1 has no regenerative path to the battery, and
    charging |tau * qdot| would penalize the eccentric phases of a gait as if they drew power.
    Divided by ``power_ref`` (in Watt) so the term stays O(1) and its weight is directly
    comparable with the tracking rewards. The ``energy/pos_mech_power_w`` diagnostic logged by
    the velocity command term is this same quantity in Watt; calibrate ``power_ref`` from it.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.clamp_min(qvel * qfrc, 0.0), dim=-1) / power_ref


def joint_torque_normalized_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Squared joint torque normalized by each actuator's effort limit.

    This covers the resistive (``I^2 R``) share of the electrical cost that mechanical
    power misses: holding a load at zero joint velocity draws no ``tau * qdot`` but still
    heats the motor. Normalizing by the effort limit keeps the 25 Nm ankles and the
    139 Nm knees on the same scale, so one weight covers the whole body.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
    limit = asset.data.joint_effort_limits[:, asset_cfg.joint_ids].clamp_min(1e-6)
    return torch.sum(torch.square(torque / limit), dim=-1)


def joint_power_cot_exp(
    env: ManagerBasedRLEnv,
    sigma_lin: float = 2800.0,
    sigma_ang: float = 1400.0,
    v_floor: float = 0.3,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Mechanical power over a generalized speed, squashed to [0, 1): a per-step cost of transport.

    Adaptive energy regularization after Liang et al. (arXiv:2403.20001). Dividing power by
    speed gives every commanded speed the same pressure, where a flat power penalty barely
    touches the slow gaits that are the least efficient (baseline CoT is 1.06 at 0.5 m/s
    against 0.67 at 1.5 m/s, yet the power term there is a third of its 2.5 m/s value).

    ``sigma_lin`` has units of force. The paper's 1000 N is 8.5x the Go1's weight, which
    keeps the exponent near 0.1 where exp is close to linear -- the exponential only guards
    against the v -> 0 blow-up, it does not reshape the cost. 8.5 x G1 weight (327 N) = 2800.

    Two changes from the paper. The speed is the smaller of measured and commanded, so
    under-tracking still costs energy but outrunning the command does not pay. And the term
    is a cost rather than the paper's bonus: in a weighted sum the bonus's constant offset
    would act as a large survival reward.

    Note this is the instantaneous, time-weighted form, not the distance-weighted ratio of
    integrals that eval_energy.py reports; the two differ whenever speed varies.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    # Positive (motoring) work per motor only; negative work is dissipated, not recovered.
    # Same convention as :func:`joint_power_l1`.
    power = torch.sum(torch.clamp_min(qvel * qfrc, 0.0), dim=-1)

    command = env.command_manager.get_command(command_name)
    lin_speed = torch.minimum(
        torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=-1),
        torch.linalg.norm(command[:, :2], dim=-1),
    )
    ang_speed = torch.minimum(torch.abs(asset.data.root_ang_vel_b[:, 2]), torch.abs(command[:, 2]))
    speed_scale = sigma_lin * lin_speed + sigma_ang * ang_speed
    # The floor keeps a standing command from turning into the maximum penalty.
    return 1.0 - torch.exp(-power / speed_scale.clamp_min(sigma_lin * v_floor))


def pace_total_power(
    env: ManagerBasedRLEnv,
    k_el: dict[str, float],
    k_regen: float = 0.0,
    include_potential: bool = False,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Physics-grounded actuator power in Watt, after PACE (arXiv:2509.06342, eqs. 11-14).

    ``P_total = P_el + P_mech (+ P_pot)`` with

    * ``P_el   = sum_j tau_j^2 * k_el_j``           Joule heating, ``k_el = R / (r k_i)^2`` [W/Nm^2]
    * ``P_mech = sum_j max(tau_j qdot_j, 0) + k_regen * sum_j min(tau_j qdot_j, 0)``
      -- the sign test is done *per motor*: positive (motoring) work is charged in full and
      negative (braking) work is assumed dissipated in the motor, so it costs nothing at the
      default ``k_regen = 0``. This departs from the paper, which sums the joints before the sign
      test so that braking in one joint offsets motoring in another over a shared DC bus; the G1
      has no known regenerative path, so no offsetting is granted here (user decision 2026-09-21).
    * ``P_pot  = sum_b m_b g v_bz``                 optional; integrates to ~0 over a gait cycle and
      is already inside tau^T qdot, so it only reshapes vertical motion within the cycle.

    Differences from :func:`joint_power_l1` / :func:`joint_torque_normalized_l2`: this is one
    quantity in one unit, so a single weight covers heat and work; and heat is scaled by each
    motor's own ``R/(r k_i)^2`` instead of
    ``1/tau_limit^2``. ``k_el`` for the G1 is not published -- see the running_pace cfg for the
    torque-speed-curve estimate and its caveat.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    tau = asset.data.applied_torque[:, asset_cfg.joint_ids]
    qdot = asset.data.joint_vel[:, asset_cfg.joint_ids]

    if not hasattr(env, "_pace_k_el"):
        env._pace_k_el = joint_coefficients(asset, k_el, asset_cfg.joint_ids)
    p_el = torch.sum(torch.square(tau) * env._pace_k_el, dim=-1)

    # Per-motor sign test: motoring work in full, braking work only at the regeneration
    # efficiency (0 = dissipated). Same convention as joint_power_l1 / joint_power_cot_exp.
    p_joint = tau * qdot
    p_mech = torch.sum(torch.clamp_min(p_joint, 0.0) + k_regen * torch.clamp_max(p_joint, 0.0), dim=-1)

    total = p_el + p_mech
    if include_potential:
        total = total + torch.sum(asset.data.default_mass.to(env.device) * asset.data.body_lin_vel_w[:, :, 2], dim=-1) * 9.81
    return total


def pace_energy_cost(
    env: ManagerBasedRLEnv,
    k_el: dict[str, float],
    k_regen: float = 0.0,
    include_potential: bool = False,
    speed_exponent: float = 2.0,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """PACE energy reward, eq. 15-16: ``P_total / (|v_cmd|^p + 1)`` in Watt.

    The commanded speed sets the normalization, so the policy cannot lower the term by running
    faster or slower than asked, and the ``+1`` keeps a standing command finite. ``speed_exponent``
    ``p`` is 2 in the paper, which makes the penalty nearly flat in speed on the G1 (P_total grows
    ~4.5x from 0.5 to 2.5 m/s while the factor falls from 0.8 to 0.14), so the 2026-09-22 PACE run
    beat the reward-shaping runs below 1.5 m/s and lost above. ``p = 1`` (1/(|v|+1): 0.67 at
    0.5 m/s, 0.29 at 2.5 m/s) keeps the low-speed pressure and raises the high-speed one ~2.4x
    at equal weight. Weight it negative; the paper uses -16e-5 per Watt for ANYmal/Tytan.
    """
    command = env.command_manager.get_command(command_name)
    speed = torch.linalg.norm(command[:, :2], dim=-1)
    gamma = 1.0 / (torch.pow(speed, speed_exponent) + 1.0)
    return gamma * pace_total_power(env, k_el, k_regen, include_potential, asset_cfg)


def stand_still(
    env: ManagerBasedRLEnv, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    reward = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return reward * (cmd_norm < 0.1)


"""
Robot.
"""


def orientation_l2(
    env: ManagerBasedRLEnv, desired_gravity: list[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward the agent for aligning its gravity with the desired gravity vector using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    desired_gravity = torch.tensor(desired_gravity, device=env.device)
    cos_dist = torch.sum(asset.data.projected_gravity_b * desired_gravity, dim=-1)  # cosine distance
    normalized = 0.5 * cos_dist + 0.5  # map from [-1, 1] to [0, 1]
    return torch.square(normalized)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)


"""
Feet rewards.
"""


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footpos_translated[:, i, :])
        footvel_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footvel_translated[:, i, :])
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def feet_too_near(
    env: ManagerBasedRLEnv, threshold: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def feet_contact_without_cmd(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """
    Reward for feet contact when the command is zero.
    """
    # asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    reward = torch.sum(is_contact, dim=-1).float()
    return reward * _standing_command_mask(env, command_name, command_threshold)


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


"""
Feet Gait rewards.
"""


def _standing_command_mask(env: ManagerBasedRLEnv, command_name: str, command_threshold: float) -> torch.Tensor:
    """1.0 where the velocity command norm is below ``command_threshold`` (stand-still command), else 0.0.

    One rule for the whole task: commands below the threshold mean "stand", at or above mean "run".
    The gait clock observation, the standing rewards and ``_command_run_gate`` all share it, so there
    is no command band where neither the standing nor the running terms are active.
    """
    command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return (command_norm < command_threshold).float()


def _command_run_gate(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Return 1.0 for running commands (norm >= threshold) and 0.0 for stand-still commands."""
    return 1.0 - _standing_command_mask(env, command_name, command_threshold)


def _gait_leg_phases(
    env: ManagerBasedRLEnv, period: float, offset: list[float] | tuple[float, ...]
) -> torch.Tensor:
    """Compute each leg's normalized phase in the interval [0, 1)."""
    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    offsets = torch.as_tensor(offset, dtype=torch.float, device=env.device).unsqueeze(0)
    return (global_phase + offsets) % 1.0


def run_contact_pattern(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float] | tuple[float, ...],
    stance_ratio: float,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Reward the exact running contact state: single support alternating with flight."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0
    leg_phase = _gait_leg_phases(env, period, offset)

    if leg_phase.shape[1] != is_contact.shape[1]:
        raise ValueError("The number of gait offsets must match the number of configured feet.")

    expected_contact = leg_phase < stance_ratio
    exact_match = torch.all(expected_contact == is_contact, dim=1).float()
    return exact_match * _command_run_gate(env, command_name)


def short_contact_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    max_contact_time: float,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Penalize foot contacts that last longer than the running stance target."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]

    # Normalize the excess so the configured reward weight remains meaningful in seconds.
    normalized_excess = torch.clamp(contact_time / max_contact_time - 1.0, min=0.0, max=1.0)
    penalty = torch.sum(torch.square(normalized_excess), dim=1)
    return penalty * _command_run_gate(env, command_name)


def push_off_reward(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float] | tuple[float, ...],
    stance_ratio: float,
    target_vz: float,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    max_target_vz: float | None = None,
    target_speed: float = 2.0,
    late_stance_start: float = 0.6,
    std: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward upward base velocity while the stance foot is preparing for takeoff."""
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0
    leg_phase = _gait_leg_phases(env, period, offset)

    if leg_phase.shape[1] != is_contact.shape[1]:
        raise ValueError("The number of gait offsets must match the number of configured feet.")

    late_stance = torch.logical_and(
        leg_phase >= stance_ratio * late_stance_start,
        leg_phase < stance_ratio,
    )
    active_push_off = torch.any(late_stance & is_contact, dim=1).float()

    target_vz_tensor = torch.full((env.num_envs,), target_vz, device=env.device)
    if max_target_vz is not None:
        command = env.command_manager.get_command(command_name)
        planar_speed = torch.linalg.vector_norm(command[:, :2], dim=1)
        speed_ratio = torch.clamp(planar_speed / target_speed, min=0.0, max=1.0)
        target_vz_tensor += (max_target_vz - target_vz) * speed_ratio

    base_vz = asset.data.root_lin_vel_w[:, 2]
    push_off_quality = torch.exp(-torch.square((base_vz - target_vz_tensor) / std))
    return push_off_quality * active_push_off * _command_run_gate(env, command_name)


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.05
    return reward


"""
Other rewards.
"""


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        reward += torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    return reward
