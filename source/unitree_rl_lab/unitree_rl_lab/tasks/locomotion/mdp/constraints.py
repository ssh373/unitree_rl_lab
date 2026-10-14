"""Constraint terms and manager for constrained RL (Kim et al., "Not Only Rewards But Also Constraints",
T-RO 2024), declared in the environment configuration next to the rewards.

A constraint term is a reward-style function ``f(env, **params) -> (num_envs,)`` plus how it is bounded:

* ``kind="mean"``: ``E[f] <= bound`` -- an *average* constraint on a continuous quantity.
* ``kind="prob"``: ``P(f > limit) <= bound`` -- a *probabilistic* constraint; the per-step cost is the
  indicator ``f > limit`` and ``bound`` is the allowed violation rate (the paper uses 0.025).

:class:`ConstraintManager` resolves the terms exactly like Isaac Lab's ``RewardManager`` (scene-entity
parameters, callable classes) and returns the raw values and the per-step costs as ``(num_envs, K)``
tensors. It does not touch the reward: the Lagrangian penalty and the multipliers live in
:class:`unitree_rl_lab.algorithms.constraints.extension.LagrangianPenaltyExtension`, which builds this
manager from ``env.cfg.constraints``. Costs are normalized by their bound before they reach the
penalty, so every constraint reads "<= 1" and one multiplier scale covers terms whose raw units differ
by orders of magnitude (joint acceleration vs. a contact count).
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING, Literal

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerBase, ManagerTermBase, ManagerTermBaseCfg, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import yaw_quat

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@configclass
class ConstraintTermCfg(ManagerTermBaseCfg):
    """One constraint: a reward-style function and its bound."""

    kind: Literal["mean", "prob"] = "mean"
    """``"mean"``: E[f] <= bound. ``"prob"``: P(f > limit) <= bound."""
    bound: float = MISSING
    """Allowed mean of f (``mean``) or allowed violation probability (``prob``). Must be positive."""
    limit: float = 0.0
    """``prob`` only: the step counts as a violation when ``f > limit``."""
    lambda_init: float = 0.0
    """Initial multiplier, in reward-weight units applied to the bound-normalized cost."""
    gated: bool = True
    """Hold the multiplier at ``lambda_init`` until the standing gate opens (see the extension)."""
    lambda_max: float | None = None
    """Per-term cap on the multiplier; ``None`` uses the solver's ``lambda_max``."""


class ConstraintManager(ManagerBase):
    """Evaluates the constraint terms of ``env.cfg.constraints`` every step."""

    _env: ManagerBasedRLEnv

    def __init__(self, cfg: object, env: ManagerBasedRLEnv):
        self._term_names: list[str] = []
        self._term_cfgs: list[ConstraintTermCfg] = []
        self._class_term_cfgs: list[ConstraintTermCfg] = []
        super().__init__(cfg, env)
        num_terms = len(self._term_names)
        self._raw = torch.zeros((self.num_envs, num_terms), dtype=torch.float, device=self.device)
        self.is_prob = torch.tensor([c.kind == "prob" for c in self._term_cfgs], dtype=torch.bool, device=self.device)
        self.bounds = torch.tensor([float(c.bound) for c in self._term_cfgs], dtype=torch.float, device=self.device)
        self.limits = torch.tensor([float(c.limit) for c in self._term_cfgs], dtype=torch.float, device=self.device)
        self.lambda_init = torch.tensor([float(c.lambda_init) for c in self._term_cfgs], dtype=torch.float, device=self.device)
        self.gated = torch.tensor([bool(c.gated) for c in self._term_cfgs], dtype=torch.bool, device=self.device)
        self.lambda_max = [c.lambda_max for c in self._term_cfgs]

    def __str__(self) -> str:
        lines = [f"<ConstraintManager> contains {len(self._term_names)} active terms."]
        for name, cfg in zip(self._term_names, self._term_cfgs):
            if cfg.kind == "prob":
                rule = f"P(f > {cfg.limit:g}) <= {cfg.bound:g}"
            else:
                rule = f"E[f] <= {cfg.bound:g}"
            lines.append(f"  {name:28s} {rule:30s} lambda_init={cfg.lambda_init:g} gated={cfg.gated}")
        return "\n".join(lines)

    # -- properties --------------------------------------------------------------------------

    @property
    def active_terms(self) -> list[str]:
        return self._term_names

    @property
    def num_terms(self) -> int:
        return len(self._term_names)

    def get_term_cfg(self, term_name: str) -> ConstraintTermCfg:
        if term_name not in self._term_names:
            raise ValueError(f"Constraint term '{term_name}' not found.")
        return self._term_cfgs[self._term_names.index(term_name)]

    # -- operations --------------------------------------------------------------------------

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        for term_cfg in self._class_term_cfgs:
            term_cfg.func.reset(env_ids=env_ids)
        return {}

    def compute(self) -> torch.Tensor:
        """Raw term values ``f`` for the current state, shape ``(num_envs, K)``."""
        for idx, term_cfg in enumerate(self._term_cfgs):
            self._raw[:, idx] = term_cfg.func(self._env, **term_cfg.params)
        return self._raw

    def cost(self, raw: torch.Tensor) -> torch.Tensor:
        """Per-step cost: ``f`` for mean terms, the indicator ``f > limit`` for prob terms."""
        return torch.where(self.is_prob, (raw > self.limits).float(), raw)

    def normalized_cost(self, raw: torch.Tensor) -> torch.Tensor:
        """Cost divided by its bound, so every constraint reads ``E[.] <= 1``."""
        return self.cost(raw) / self.bounds

    def get_active_iterable_terms(self, env_idx: int) -> Sequence[tuple[str, Sequence[float]]]:
        return [(name, [self._raw[env_idx, idx].item()]) for idx, name in enumerate(self._term_names)]

    # -- implementation ------------------------------------------------------------------------

    def _prepare_terms(self):
        cfg_items = self.cfg.items() if isinstance(self.cfg, dict) else self.cfg.__dict__.items()
        for term_name, term_cfg in cfg_items:
            if term_cfg is None:
                continue
            if not isinstance(term_cfg, ConstraintTermCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type ConstraintTermCfg. Received: '{type(term_cfg)}'."
                )
            if term_cfg.kind not in ("mean", "prob"):
                raise ValueError(f"Constraint '{term_name}': kind must be 'mean' or 'prob', got {term_cfg.kind!r}.")
            if not isinstance(term_cfg.bound, (float, int)) or term_cfg.bound <= 0.0:
                raise ValueError(f"Constraint '{term_name}': bound must be a positive number, got {term_cfg.bound!r}.")
            self._resolve_common_term_cfg(term_name, term_cfg, min_argc=1)
            self._term_names.append(term_name)
            self._term_cfgs.append(term_cfg)
            if isinstance(term_cfg.func, ManagerTermBase):
                self._class_term_cfgs.append(term_cfg)


"""
Cost functions for constraints that have no penalty-shaped reward counterpart.
"""


def track_lin_vel_xy_error(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Planar velocity tracking error ``|v_cmd - v|`` in the yaw-aligned base frame, in m/s.

    The same quantity the baseline's ``track_lin_vel_xy_yaw_frame_exp`` squashes through its exp kernel,
    kept linear so a bound on its mean reads as a speed error.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    return torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2], dim=-1)


def track_ang_vel_z_error(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Yaw-rate tracking error ``|w_cmd - w_z|`` in rad/s (linear form of ``track_ang_vel_z_exp``)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.abs(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
