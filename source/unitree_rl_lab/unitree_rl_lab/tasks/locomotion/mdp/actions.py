from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.utils import DelayBuffer, configclass


class RandomizedDelayedJointPositionAction(JointPositionAction):
    """Joint position action with per-environment command delay and calibration error."""

    cfg: RandomizedDelayedJointPositionActionCfg

    def __init__(self, cfg: RandomizedDelayedJointPositionActionCfg, env):
        super().__init__(cfg, env)

        if cfg.min_delay < 0 or cfg.max_delay < cfg.min_delay:
            raise ValueError(
                f"Invalid action delay range ({cfg.min_delay}, {cfg.max_delay}). "
                "Expected 0 <= min_delay <= max_delay."
            )

        self._delay_buffer = DelayBuffer(cfg.max_delay, self.num_envs, self.device)
        self._action_bias = torch.zeros_like(self.raw_actions)
        self._randomize(torch.arange(self.num_envs, device=self.device))

    def _randomize(self, env_ids: torch.Tensor) -> None:
        delays = torch.randint(
            self.cfg.min_delay,
            self.cfg.max_delay + 1,
            (len(env_ids),),
            device=self.device,
            dtype=torch.int32,
        )
        self._delay_buffer.set_time_lag(delays, env_ids)
        self._action_bias[env_ids].uniform_(*self.cfg.action_bias_range)

    def process_actions(self, actions: torch.Tensor) -> None:
        delayed_actions = self._delay_buffer.compute(actions)
        randomized_actions = delayed_actions + self._action_bias
        randomized_actions = randomized_actions + torch.empty_like(
            randomized_actions
        ).uniform_(*self.cfg.action_noise_range)
        super().process_actions(randomized_actions)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        super().reset(env_ids)
        self._delay_buffer.reset(env_ids)

        resolved_env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        self._randomize(resolved_env_ids)


@configclass
class RandomizedDelayedJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for randomized delayed joint-position commands."""

    class_type: type = RandomizedDelayedJointPositionAction

    min_delay: int = 0
    """Minimum command delay in policy steps."""

    max_delay: int = 1
    """Maximum command delay in policy steps."""

    action_bias_range: tuple[float, float] = (-0.02, 0.02)
    """Per-episode normalized action bias range."""

    action_noise_range: tuple[float, float] = (-0.01, 0.01)
    """Per-step normalized action noise range."""
