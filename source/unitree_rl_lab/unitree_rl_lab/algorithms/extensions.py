"""Composable additions to rsl_rl's PPO.

rsl_rl's :class:`~rsl_rl.algorithms.ppo.PPO` has exactly two places where an auxiliary method can
plug in without rewriting the update loop: the reward it records for a transition
(:meth:`process_env_step`) and what happens after a policy update (:meth:`update`). AMP needs
only those two -- its style reward is added to the task reward and its discriminator trains on
its own optimizer -- and so does a penalty-style constraint (the Lagrangian cost penalty in
:mod:`unitree_rl_lab.algorithms.constraints`). Modelling each as a
:class:`PPOExtension` lets several of them run side by side as a list instead of a chain of
subclasses, and keeps PPO's own loss computation untouched so upstream changes to rsl_rl keep
working.

Extensions that need to change the advantage itself (a Lagrangian cost critic, for instance) do
not fit this shape and need a real subclass.
"""

from __future__ import annotations

from itertools import chain
from typing import Any

import torch

from rsl_rl.algorithms import PPO


class PPOExtension:
    """Hook interface. Every method has a no-op default so an extension implements only what it uses."""

    name: str = "extension"

    def on_env_step(self, obs, rewards: torch.Tensor, dones: torch.Tensor, extras: dict) -> torch.Tensor:
        """Return the rewards to record for this step. Called before PPO stores the transition."""
        return rewards

    def after_update(self, storage) -> dict[str, float]:
        """Run once per PPO update, with the (not yet cleared) rollout storage. Returns log values."""
        return {}

    def train(self, mode: bool = True) -> None:
        pass

    def parameters(self):
        """Trainable parameters, for multi-GPU gradient reduction. Default: none."""
        return iter(())

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        pass


class ExtendedPPO(PPO):
    """PPO plus a list of :class:`PPOExtension`.

    ``extensions`` is a list of already constructed extension objects. The class also accepts
    ``amp_cfg`` and ``constraint_cfg`` for convenience -- dictionaries from the agent configuration --
    and builds an :class:`~unitree_rl_lab.algorithms.amp.extension.AMPExtension` /
    :class:`~unitree_rl_lab.algorithms.constraints.extension.LagrangianPenaltyExtension` from them,
    mirroring how PPO accepts ``rnd_cfg``.
    """

    def __init__(
        self,
        *args,
        extensions: list[PPOExtension] | None = None,
        amp_cfg: dict | None = None,
        constraint_cfg: dict | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.extensions: list[PPOExtension] = list(extensions or [])
        if amp_cfg is not None:
            from unitree_rl_lab.algorithms.amp.extension import AMPExtension

            self.extensions.append(AMPExtension(device=self.device, **amp_cfg))
        if constraint_cfg is not None:
            from unitree_rl_lab.algorithms.constraints.extension import LagrangianPenaltyExtension

            self.extensions.append(LagrangianPenaltyExtension(device=self.device, **constraint_cfg))

    # -- rollout ------------------------------------------------------------------------------

    def process_env_step(self, obs, rewards, dones, extras):
        for ext in self.extensions:
            rewards = ext.on_env_step(obs, rewards, dones, extras)
        return super().process_env_step(obs, rewards, dones, extras)

    # -- update -------------------------------------------------------------------------------

    def update(self):
        # Extensions see the rollout before PPO clears the storage at the end of its update.
        ext_logs = {}
        for ext in self.extensions:
            ext_logs.update(ext.after_update(self.storage))
        loss_dict = super().update()
        loss_dict.update(ext_logs)
        return loss_dict

    # -- multi-GPU: include extension parameters, as PPO does for RND ---------------------------

    def broadcast_parameters(self):
        model_params = [self.policy.state_dict()]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        model_params += [ext.state_dict() for ext in self.extensions]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.policy.load_state_dict(model_params[0])
        idx = 1
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[idx])
            idx += 1
        for ext in self.extensions:
            ext.load_state_dict(model_params[idx])
            idx += 1

    def reduce_parameters(self):
        params = self.policy.parameters()
        if self.rnd:
            params = chain(params, self.rnd.parameters())
        for ext in self.extensions:
            params = chain(params, ext.parameters())
        params = list(params)
        grads = [p.grad.view(-1) for p in params if p.grad is not None]
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        for p in params:
            if p.grad is not None:
                numel = p.numel()
                p.grad.data.copy_(all_grads[offset : offset + numel].view_as(p.grad.data))
                offset += numel
