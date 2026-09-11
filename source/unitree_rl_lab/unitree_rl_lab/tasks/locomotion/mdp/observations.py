from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(
    env: ManagerBasedRLEnv,
    period: float,
    command_name: str | None = None,
    command_threshold: float = 0.0,
) -> torch.Tensor:
    """Gait clock as (sin, cos) of the episode-time phase.

    When ``command_name`` is given and ``command_threshold`` is positive, the clock output is masked to
    (0, 0) while the velocity command norm is below the threshold. The underlying clock keeps running on
    episode time, so the phase seen once a command resumes is the same one the reward terms use. The
    deployment observation of the same name (``deploy/include/isaaclab/envs/mdp/observations/observations.h``)
    applies the identical rule from the exported ``command_threshold`` parameter.
    """
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)

    if command_name is not None and command_threshold > 0.0:
        command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        phase[command_norm < command_threshold] = 0.0
    return phase
