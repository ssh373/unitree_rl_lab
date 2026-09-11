# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Left-right symmetry augmentation for the Unitree G1 29-DoF locomotion tasks."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["compute_g1_29dof_symmetric_states"]


@torch.no_grad()
def compute_g1_29dof_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Append the left-right mirrored G1 observations and actions to a training batch.

    The first half of each returned tensor is the original batch and the second half is its
    sagittal-plane reflection. Only left-right symmetry is used: front-back symmetry is not valid
    for the G1 running task's morphology and command distribution.
    """
    base_env = env.unwrapped

    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        for group_name in obs.keys():
            if group_name in base_env.observation_manager.active_terms:
                obs_aug[group_name][batch_size:] = _mirror_observation_group(
                    base_env, group_name, obs[group_name]
                )
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = actions.repeat(2, 1)
        action_names = _get_action_joint_names(base_env)
        mirror_indices, mirror_signs = _joint_mirror_map(action_names, actions.device)
        actions_aug[batch_size:] = _mirror_joint_data(actions, mirror_indices, mirror_signs)
    else:
        actions_aug = None

    return obs_aug, actions_aug


def _mirror_observation_group(env, group_name: str, obs: torch.Tensor) -> torch.Tensor:
    """Mirror one concatenated observation group, including every history sample."""
    mirrored = obs.clone()
    term_names = env.observation_manager.active_terms[group_name]
    term_dims = env.observation_manager.group_obs_term_dim[group_name]
    joint_names = env.scene["robot"].joint_names
    action_names = _get_action_joint_names(env)

    offset = 0
    for term_name, term_shape in zip(term_names, term_dims):
        width = math.prod(term_shape)
        term = obs[:, offset : offset + width]

        if term_name == "base_lin_vel":
            transformed = _mirror_vector_history(term, (1.0, -1.0, 1.0))
        elif term_name == "base_ang_vel":
            transformed = _mirror_vector_history(term, (-1.0, 1.0, -1.0))
        elif term_name == "projected_gravity":
            transformed = _mirror_vector_history(term, (1.0, -1.0, 1.0))
        elif term_name == "velocity_commands":
            transformed = _mirror_vector_history(term, (1.0, -1.0, -1.0))
        elif term_name in ("joint_pos_rel", "joint_vel_rel"):
            mirror_indices, mirror_signs = _joint_mirror_map(joint_names, obs.device)
            transformed = _mirror_joint_history(term, mirror_indices, mirror_signs)
        elif term_name == "last_action":
            mirror_indices, mirror_signs = _joint_mirror_map(action_names, obs.device)
            transformed = _mirror_joint_history(term, mirror_indices, mirror_signs)
        elif term_name == "gait_phase":
            # Swapping the legs corresponds to a half-cycle phase shift: (sin, cos) -> (-sin, -cos).
            transformed = -term
        else:
            raise ValueError(
                f"Unsupported observation term '{group_name}/{term_name}' in G1 symmetry augmentation. "
                "Add its left-right transformation before enabling the term."
            )

        mirrored[:, offset : offset + width] = transformed
        offset += width

    if offset != obs.shape[-1]:
        raise ValueError(
            f"Observation dimensions for group '{group_name}' do not add up: transformed {offset}, "
            f"received {obs.shape[-1]}."
        )
    return mirrored


def _mirror_vector_history(data: torch.Tensor, signs: tuple[float, ...]) -> torch.Tensor:
    """Apply component signs to a vector observation with an optional flattened history."""
    component_count = len(signs)
    if data.shape[-1] % component_count != 0:
        raise ValueError(f"Cannot reshape observation of width {data.shape[-1]} into {component_count}-D vectors.")
    history = data.reshape(data.shape[0], -1, component_count)
    sign_tensor = torch.tensor(signs, dtype=data.dtype, device=data.device)
    return (history * sign_tensor).reshape_as(data)


def _mirror_joint_history(
    data: torch.Tensor, mirror_indices: torch.Tensor, mirror_signs: torch.Tensor
) -> torch.Tensor:
    """Mirror joint data with an optional flattened history dimension."""
    joint_count = mirror_indices.numel()
    if data.shape[-1] % joint_count != 0:
        raise ValueError(f"Cannot reshape observation of width {data.shape[-1]} into {joint_count} joints.")
    history = data.reshape(data.shape[0], -1, joint_count)
    mirrored = _mirror_joint_data(history, mirror_indices, mirror_signs)
    return mirrored.reshape_as(data)


def _mirror_joint_data(
    data: torch.Tensor, mirror_indices: torch.Tensor, mirror_signs: torch.Tensor
) -> torch.Tensor:
    """Swap left/right joints and apply signs required by sagittal-plane reflection."""
    return data.index_select(-1, mirror_indices) * mirror_signs


def _joint_mirror_map(joint_names: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Create output-to-input indices and signs for a named G1 joint vector."""
    name_to_index = {name: index for index, name in enumerate(joint_names)}
    mirror_indices: list[int] = []
    mirror_signs: list[float] = []

    for name in joint_names:
        if name.startswith("left_"):
            mirrored_name = "right_" + name[len("left_") :]
        elif name.startswith("right_"):
            mirrored_name = "left_" + name[len("right_") :]
        else:
            mirrored_name = name

        if mirrored_name not in name_to_index:
            raise ValueError(f"G1 symmetry counterpart for joint '{name}' was not found: '{mirrored_name}'.")

        mirror_indices.append(name_to_index[mirrored_name])
        mirror_signs.append(-1.0 if "_roll_" in name or "_yaw_" in name else 1.0)

    return (
        torch.tensor(mirror_indices, dtype=torch.long, device=device),
        torch.tensor(mirror_signs, dtype=torch.float, device=device),
    )


def _get_action_joint_names(env) -> list[str]:
    """Return the resolved joint order used by the single joint-position action term."""
    active_terms = env.action_manager.active_terms
    if len(active_terms) != 1:
        raise ValueError(f"G1 symmetry expects one joint action term, but found {active_terms}.")
    action_term = env.action_manager.get_term(active_terms[0])
    return list(action_term.IO_descriptor.joint_names)
