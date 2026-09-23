from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
    ang_vel_delta: float = 0.0,
) -> torch.Tensor:
    """Widen the velocity command ranges once ``reward_term_name`` clears 80% of its weight.

    Every level moves lin_vel_x and lin_vel_y by 0.1 toward their limit_ranges. With
    ``ang_vel_delta`` > 0 the same gate also moves ang_vel_z by that amount, so the yaw range
    grows on the lin-vel gate instead of on its own tracking reward (the yaw reward on the
    running task plateaus well under 80% and would never open its own gate, see
    ``ang_vel_cmd_levels``). 0.05 takes the running task's yaw range from 0.4 to 1.0 in 12 of
    the 21 levels lin_vel_x needs. Default 0.0 keeps ang_vel_z fixed (velocity task behaviour).
    """
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()
            if ang_vel_delta > 0.0:
                delta_ang = torch.tensor([-ang_vel_delta, ang_vel_delta], device=env.device)
                ranges.ang_vel_z = torch.clamp(
                    torch.tensor(ranges.ang_vel_z, device=env.device) + delta_ang,
                    limit_ranges.ang_vel_z[0],
                    limit_ranges.ang_vel_z[1],
                ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def ang_vel_z_cmd_range(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    """Log-only term: the current upper bound of the ang_vel_z command range.

    Changes nothing. ``lin_vel_cmd_levels`` logs only lin_vel_x, so this exposes the yaw range it
    widens through ``ang_vel_delta`` as ``Curriculum/ang_vel_z_cmd_range``. Kept as a separate term
    so the existing ``Curriculum/lin_vel_cmd_levels`` key stays comparable with older runs. List it
    after ``lin_vel_cmd_levels`` in the CurriculumCfg so it reports the range of the same step.
    """
    ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    return torch.tensor(ranges.ang_vel_z[1], device=env.device)


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)


def reward_weight_linear_schedule(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    start_weight: float,
    end_weight: float,
    start_step: int,
    end_step: int,
) -> torch.Tensor:
    """Linearly interpolate a reward term's weight between two global step counts.

    ``isaaclab.envs.mdp.modify_reward_weight`` switches the weight in one jump, which makes
    a strong cost penalty arrive as a step change in the advantage scale. Ramping instead
    lets the gait settle under the task rewards first and then trades it down for
    efficiency, which is what keeps the running gait alive.

    Note that ``start_step``/``end_step`` count environment steps, i.e. PPO iteration
    multiplied by ``num_steps_per_env``.
    """
    if end_step <= start_step:
        raise ValueError(f"end_step ({end_step}) must be greater than start_step ({start_step}).")

    alpha = (env.common_step_counter - start_step) / (end_step - start_step)
    alpha = min(max(alpha, 0.0), 1.0)

    term_cfg = env.reward_manager.get_term_cfg(term_name)
    term_cfg.weight = start_weight + alpha * (end_weight - start_weight)

    return torch.tensor(term_cfg.weight, device=env.device)


def reward_weight_after_standing(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    end_weight: float,
    ep_len_threshold: float = 500.0,
    ramp_iters: int = 500,
    steps_per_iter: int = 24,
) -> torch.Tensor:
    """Keep a reward term at zero weight until the policy can stand, then ramp it in.

    Standing is detected from the mean length of the last ``num_envs`` finished episodes
    exceeding ``ep_len_threshold`` steps. The gate latches: once open it never closes, so a
    later dip in episode length cannot toggle the penalty on and off. After opening, the weight
    ramps linearly from 0 to ``end_weight`` over ``ramp_iters`` PPO iterations
    (``steps_per_iter`` environment steps each) so the value function sees a gradual change in
    return scale rather than a step.

    Why not a fixed-iteration ramp: the time to first stand varies from 0.3k to 11k iterations
    between otherwise identical runs, and the energy penalties themselves delay it (the falling
    phase earns ~0.1 of reward per step, so a 0.05 penalty is half the signal), so a schedule
    tied to iteration count either fires before standing or long after. Tied to standing, the
    pre-standing phase is bit-identical to a run without the term.

    All terms using this function share one episode-length tracker and one latch on the env, so
    several penalties open together.
    """
    state = getattr(env, "_standing_gate_state", None)
    if state is None:
        state = {
            "lengths": torch.zeros(env.num_envs, device=env.device),
            "ptr": 0,
            "count": 0,
            "last_step": -1,
            "opened_step": None,
        }
        env._standing_gate_state = state

    # Update the episode-length history once per environment step, even if several terms call
    # this function for the same reset batch. episode_length_buf still holds the finished
    # lengths here: ManagerBasedRLEnv._reset_idx runs the curriculum before zeroing it.
    # rsl_rl's ``learn(init_at_random_ep_len=True)`` (used by scripts/rsl_rl/train.py) fills
    # episode_length_buf with uniform 0..max_episode_length offsets at iteration 0, so every
    # episode that ends within the first max_episode_length steps reports an inflated length
    # (mean ~500 regardless of the policy). Skip those, and only judge once the tracker holds
    # num_envs genuine episodes -- otherwise the very first reset batch decides the latch
    # (the 09-22 s04/s06 runs opened at iteration ~2 with episodes 10 steps long).
    if state["last_step"] != env.common_step_counter:
        state["last_step"] = env.common_step_counter
        if env.common_step_counter > env.max_episode_length:
            ids = torch.as_tensor(env_ids, device=env.device)
            lengths = env.episode_length_buf[ids].float()
            n = len(lengths)
            if n > 0:
                idx = (state["ptr"] + torch.arange(n, device=env.device)) % env.num_envs
                state["lengths"][idx] = lengths
                state["ptr"] = int((state["ptr"] + n) % env.num_envs)
                state["count"] = min(state["count"] + n, env.num_envs)
        if state["opened_step"] is None and state["count"] >= env.num_envs:
            mean_len = state["lengths"][: state["count"]].mean().item()
            if mean_len > ep_len_threshold:
                state["opened_step"] = env.common_step_counter

    if state["opened_step"] is None:
        alpha = 0.0
    else:
        ramp_steps = max(ramp_iters * steps_per_iter, 1)
        alpha = min((env.common_step_counter - state["opened_step"]) / ramp_steps, 1.0)

    term_cfg = env.reward_manager.get_term_cfg(term_name)
    term_cfg.weight = alpha * end_weight
    return torch.tensor(term_cfg.weight, device=env.device)
