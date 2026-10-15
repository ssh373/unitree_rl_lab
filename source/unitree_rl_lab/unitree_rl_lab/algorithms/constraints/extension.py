"""Multi-constraint Lagrangian penalty as a :class:`~unitree_rl_lab.algorithms.extensions.PPOExtension`.

The constrained problem is ``max_pi J_r(pi)  s.t.  J_{c_k}(pi) <= 1`` for every constraint ``k`` of the
environment's :class:`~unitree_rl_lab.tasks.locomotion.mdp.constraints.ConstraintManager`, whose costs
are already divided by their bound. Its Lagrangian relaxation is solved RCPO-style (Tessler et al.,
2019): PPO runs unchanged on the penalized reward

    r' = r - step_dt * sum_k lambda_k * c_k

and after every PPO update each multiplier takes a dual-ascent step on its rollout-mean violation,
either additive or (default) multiplicative:

    additive:        lambda_k <- clip(lambda_k + lr * v_k, 0, cap_k)
    multiplicative:  lambda_k <- min(lambda_k * exp(lr * v_k), cap_k),   v_k = clip(J_{c_k} - 1, -clip, clip)

A multiplier therefore rises while its constraint is violated and decays back once it is satisfied, so
the penalty weights are set by the bounds instead of by hand. ``lambda`` is in reward-weight units (it is
multiplied by ``step_dt`` the way Isaac Lab's RewardManager multiplies every weight) and acts on the
bound-normalized cost, so ``lambda = 1`` charges one reward unit per "one bound's worth" of cost.

Why multiplicative (2026-10-14): the bound-normalized costs of different terms sit on scales that differ
by orders of magnitude, and so do the multipliers that match the baseline's hand-tuned weights
(``lambda = weight * bound`` is 0.005-0.13 for the G1 running penalties). An additive step moves every
violated term by the same absolute amount, which is 10-200x the baseline weight within a few iterations;
a relative step keeps the baseline's weight ratios and lets only the violated terms grow, by at most
``exp(lr)`` per iteration. With the multiplicative rule a term initialized at 0 stays at 0, so every
term needs a positive ``lambda_init`` (``running_constrained_env_cfg.py`` derives them from the baseline
weights). The per-term cap defaults to ``lambda_max_ratio * lambda_init``: a multiplier sitting at its cap
says the bound is infeasible for this policy.

Compared with IPO (the paper's solver) there is no cost critic and no barrier: the per-state information
still reaches the policy because the cost enters the per-step reward; only the *weight* comes from a
rollout average. The known failure mode is oscillation of ``lambda`` around the bound; keep ``lambda_lr``
small, and add integral/derivative terms (PID Lagrangian, Stooke et al. 2020) here if it shows.

Two details copied from the reward-side energy terms:

* **Standing gate.** Terms with ``gated=True`` keep ``lambda_init`` until the policy can stand, detected
  like ``mdp.reward_weight_after_standing`` (mean length of the last ``num_envs`` finished episodes above
  ``ep_len_threshold``, episodes inside the first ``max_episode_length`` steps ignored because
  ``init_at_random_ep_len`` inflates them; latches). Terms that *define* the task -- command tracking --
  use ``gated=False`` with a non-zero ``lambda_init`` so the policy has a reason to move from step 0.
* **Reset steps.** An environment reset in this step reports the reset state's cost, not the transition's;
  it is dropped from the penalty and from the cost estimate.

The manager is built from ``env.cfg.constraints`` when the extension is constructed (the runner injects
the environment) and attached as ``env.constraint_manager`` so evaluation scripts can read it too.
"""

from __future__ import annotations

from typing import Any

import torch

from unitree_rl_lab.algorithms.extensions import PPOExtension


class LagrangianPenaltyExtension(PPOExtension):
    name = "constraint"

    def __init__(
        self,
        *,
        lambda_lr: float = 1.0e-2,
        lambda_max: float = 10.0,
        lambda_max_ratio: float | None = None,
        update_rule: str = "additive",
        violation_clip: float | None = None,
        start_after_standing: bool = True,
        ep_len_threshold: float = 500.0,
        device: str = "cpu",
        _env=None,
    ):
        """
        Args:
            lambda_lr: Dual step per PPO iteration per unit of relative violation ``J_c - 1``.
            lambda_max: Default cap on every multiplier; a term's ``lambda_max`` overrides it.
            lambda_max_ratio: If set, a term's cap is ``ratio * lambda_init`` (when its ``lambda_init`` > 0 and
                it has no explicit ``lambda_max``); ``lambda_max`` then only applies to terms starting at 0.
            update_rule: ``"additive"`` or ``"multiplicative"`` dual step (see the module docstring).
            violation_clip: If set, the relative violation is clipped to ``[-clip, clip]`` before the dual
                step, so a multiplier moves by at most ``lr * clip`` per iteration. Costs are unbounded
                above (a random policy's yaw error is 40x its bound) and without the clip the early
                violations push a multiplier to its cap within a few iterations.
            start_after_standing / ep_len_threshold: Standing gate for the ``gated`` terms.
            device: Torch device of the rewards.
            _env: The environment; injected by the runner. Not part of the user-facing configuration.
        """
        if _env is None:
            raise RuntimeError("LagrangianPenaltyExtension needs the environment (injected by ExtendedOnPolicyRunner).")
        if update_rule not in ("additive", "multiplicative"):
            raise ValueError(f"update_rule must be 'additive' or 'multiplicative', got {update_rule!r}.")
        self.update_rule = update_rule
        self.lambda_lr = float(lambda_lr)
        self.violation_clip = None if violation_clip is None else float(violation_clip)
        self.start_after_standing = start_after_standing
        self.ep_len_threshold = float(ep_len_threshold)
        self.device = device
        self._env = _env

        base_env = _env.unwrapped
        manager = getattr(base_env, "constraint_manager", None)
        if manager is None:
            constraints_cfg = getattr(base_env.cfg, "constraints", None)
            if constraints_cfg is None:
                raise RuntimeError(
                    "The environment configuration has no 'constraints' section; add a ConstraintsCfg of"
                    " ConstraintTermCfg terms (see unitree_rl_lab.tasks.locomotion.mdp.constraints)."
                )
            from unitree_rl_lab.tasks.locomotion.mdp.constraints import ConstraintManager

            manager = ConstraintManager(constraints_cfg, base_env)
            base_env.constraint_manager = manager
            print(manager)
        self.manager = manager
        self.term_names: list[str] = list(manager.active_terms)
        num_terms = len(self.term_names)

        self.lmbda = manager.lambda_init.detach().clone().float().to(device)
        caps = []
        for init, explicit in zip(self.lmbda.tolist(), manager.lambda_max):
            if explicit is not None:
                caps.append(float(explicit))
            elif lambda_max_ratio is not None and init > 0.0:
                caps.append(float(lambda_max_ratio) * init)
            else:
                caps.append(float(lambda_max))
        self.lambda_max = torch.tensor(caps, dtype=torch.float, device=device)
        if update_rule == "multiplicative" and bool((self.lmbda <= 0.0).any()):
            zero = [n for n, v in zip(self.term_names, self.lmbda.tolist()) if v <= 0.0]
            print(f"[LagrangianPenaltyExtension] multiplicative update: terms starting at 0 stay at 0: {zero}")
        self.gated = manager.gated.to(device)
        self.is_prob = manager.is_prob.to(device)

        self.iteration = 0
        self.gate_open = not start_after_standing
        # Per-rollout accumulators (normalized cost, raw value) and the penalty.
        self._cost_sum = torch.zeros(num_terms, device=device)
        self._raw_sum = torch.zeros(num_terms, device=device)
        self._count = 0
        self._penalty_sum = torch.zeros((), device=device)
        self._penalty_count = 0
        # Standing-gate tracker (finished-episode lengths, ring buffer of num_envs entries).
        self._cur_len: torch.Tensor | None = None
        self._len_buf: torch.Tensor | None = None
        self._len_ptr = 0
        self._len_count = 0

    # -- rollout ------------------------------------------------------------------------------

    def _track_standing(self, dones: torch.Tensor) -> None:
        if self.gate_open:
            return
        env = self._env.unwrapped
        num_envs = dones.shape[0]
        if self._cur_len is None:
            self._cur_len = torch.zeros(num_envs, device=dones.device)
            self._len_buf = torch.zeros(num_envs, device=dones.device)
        self._cur_len += 1.0
        finished = dones.view(-1).bool()
        if finished.any():
            # init_at_random_ep_len offsets every episode in the first max_episode_length steps.
            if env.common_step_counter > env.max_episode_length:
                lengths = self._cur_len[finished]
                for value in lengths:
                    self._len_buf[self._len_ptr] = value
                    self._len_ptr = (self._len_ptr + 1) % num_envs
                self._len_count = min(self._len_count + lengths.numel(), num_envs)
            self._cur_len[finished] = 0.0
        if self._len_count >= num_envs and self._len_buf.mean().item() > self.ep_len_threshold:
            self.gate_open = True

    def on_env_step(self, obs, rewards, dones, extras):
        env = self._env.unwrapped
        with torch.no_grad():
            raw = self.manager.compute().to(rewards.device)  # (N, K)
            cost = self.manager.normalized_cost(raw).to(rewards.device)
            # A reset env reports the reset state's cost, not the transition's: drop it.
            keep = (1.0 - dones.float().to(rewards.device)).view(-1, 1)
            raw = raw * keep
            cost = cost * keep
        self._cost_sum += cost.sum(dim=0)
        self._raw_sum += raw.sum(dim=0)
        self._count += cost.shape[0]
        self._track_standing(dones)
        # Isaac Lab's RewardManager multiplies every weight by step_dt; lambda is in those units.
        penalty = env.step_dt * (cost * self.lmbda).sum(dim=-1)
        self._penalty_sum += penalty.sum()
        self._penalty_count += penalty.numel()
        return rewards - penalty.view_as(rewards)

    # -- dual update ---------------------------------------------------------------------------

    def after_update(self, storage) -> dict[str, float]:
        cost_sum = self._cost_sum.clone()
        raw_sum = self._raw_sum.clone()
        count = torch.tensor(float(self._count), device=cost_sum.device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # Every rank steps the multipliers with the same global cost mean.
            for t in (cost_sum, raw_sum, count):
                torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
        count = count.clamp_min(1.0)
        cost_mean = cost_sum / count  # normalized: 1.0 == at the bound
        raw_mean = raw_sum / count
        violation = cost_mean - 1.0

        logs: dict[str, float] = {
            "constraint/gate_open": float(self.gate_open),
            "constraint/penalty_mean": (self._penalty_sum / max(self._penalty_count, 1)).item(),
            "constraint/max_violation": violation.max().item(),
        }
        for k, name in enumerate(self.term_names):
            logs[f"constraint/{name}/lambda"] = self.lmbda[k].item()  # the multiplier this rollout used
            logs[f"constraint/{name}/violation"] = violation[k].item()
            if self.is_prob[k]:
                logs[f"constraint/{name}/viol_prob"] = (cost_mean[k] * self.manager.bounds[k]).item()
            logs[f"constraint/{name}/raw_mean"] = raw_mean[k].item()
            logs[f"constraint/{name}/lambda_frac_of_cap"] = (self.lmbda[k] / self.lambda_max[k]).item()

        active = (~self.gated) | self.gate_open
        step = violation if self.violation_clip is None else violation.clamp(-self.violation_clip, self.violation_clip)
        if self.update_rule == "multiplicative":
            new_lmbda = self.lmbda * torch.exp(self.lambda_lr * step)
        else:
            new_lmbda = (self.lmbda + self.lambda_lr * step).clamp(min=0.0)
        new_lmbda = torch.minimum(new_lmbda, self.lambda_max)
        self.lmbda = torch.where(active, new_lmbda, self.lmbda)

        self.iteration += 1
        self._cost_sum.zero_()
        self._raw_sum.zero_()
        self._count = 0
        self._penalty_sum.zero_()
        self._penalty_count = 0
        return logs

    # -- bookkeeping ---------------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {
            "lambda": {name: self.lmbda[k].item() for k, name in enumerate(self.term_names)},
            "iteration": self.iteration,
            "gate_open": self.gate_open,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not state:
            return
        saved = state.get("lambda", {})
        for k, name in enumerate(self.term_names):
            if name in saved:
                self.lmbda[k] = float(saved[name])
        self.iteration = int(state.get("iteration", self.iteration))
        self.gate_open = bool(state.get("gate_open", self.gate_open))
