"""AMP as a :class:`~unitree_rl_lab.algorithms.extensions.PPOExtension`.

Per environment step the discriminator scores the ``amp`` observation group -- the
``(s_{t-1}, s_t)`` pair that the step just produced -- and ``reward_weight * step_dt * style_reward`` is
added to the task reward. After every PPO update the discriminator is trained on transitions from
the rollout that was just collected against transitions from the reference motions.

Reference transitions for discriminator training are drawn **speed-matched** to the policy
transitions they are compared with (:meth:`AmpMotionData.sample_matched`). The running takes are
mostly 1-2 m/s while the task commands down to 0.1 m/s; sampled uniformly, the discriminator learns
"fast means human" from ``base_lin_vel`` alone and the style reward turns into a speed bonus that
fights command tracking. Matched by speed, it has to score how the body moves at a given speed.

Two masks keep the style reward honest:

* **Reset steps.** When an environment resets, its observation history is refilled with the reset
  state, so the pair no longer describes a movement. Its style reward is zeroed and the pair is
  excluded from discriminator training.
* **Command gate.** Below ``command_gate`` m/s of commanded planar speed the style reward is zero
  *and* the transition is kept out of the discriminator's policy batch. Originally this only
  excluded standing environments (0.1 m/s); it is now set to the reference speed floor, because
  a policy transition slower than any reference transition would let the discriminator separate
  the two sides by speed again -- the very cue speed matching removes. Slow commands are therefore
  learned from the task rewards alone. The gate is on the *command*, not the achieved speed, so the
  policy cannot earn style reward by running faster than asked.

When the PPO symmetry augmentation is on, reference batches are mirrored with the same function
that mirrors the policy batches. Otherwise the discriminator would see mirrored transitions only
on the policy side and could learn "mirrored means fake" from small left-right asymmetries in the
human data.

The reference features need the robot's default joint position, which exists only once the
environment does; they are built on the first environment step (:meth:`_lazy_prepare`). Playing
back or evaluating a checkpoint never steps the algorithm, so it never touches the motion files.
"""

from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDict

from unitree_rl_lab.algorithms.amp.data import AmpMotionData, transition_speed
from unitree_rl_lab.algorithms.amp.discriminator import AmpDiscriminator
from unitree_rl_lab.algorithms.extensions import PPOExtension


class AMPExtension(PPOExtension):
    name = "amp"

    def __init__(
        self,
        motion_files: list[str],
        *,
        obs_group: str = "amp",
        speed_range: tuple[float, float] = (0.2, 2.5),
        speed_matched_sampling: bool = True,
        speed_match_window: float = 0.1,
        reward_weight: float = 2.0,
        reward_weight_start: float = 0.0,
        ramp_start_iter: int = 0,
        ramp_end_iter: int = 0,
        command_name: str | None = "base_velocity",
        command_gate: float = 0.1,
        hidden_dims: tuple[int, ...] | list[int] = (1024, 512),
        activation: str = "relu",
        normalize_input: bool = True,
        grad_penalty_coef: float = 10.0,
        learning_rate: float = 1.0e-4,
        num_updates_per_iter: int = 10,
        batch_size: int = 4096,
        use_symmetry_augmentation: bool = True,
        device: str = "cpu",
        _env=None,
    ):
        """
        Args:
            motion_files: Reference motion npz files (see :mod:`unitree_rl_lab.algorithms.amp.data`).
            obs_group: Name of the observation group holding the transition pair.
            speed_range: Planar-speed gate applied to the reference transitions.
            speed_matched_sampling: Draw each reference transition for discriminator training at the
                speed of the policy transition it is paired with (see
                :meth:`AmpMotionData.sample_matched`), so speed itself is not a discriminative cue.
            speed_match_window: Half-width in m/s of the speed window used for matching.
            reward_weight: Final weight of the style reward, in reward-manager units (scaled by
                ``step_dt`` like every task reward weight; 2.0 matches ``track_lin_vel_xy``).
            reward_weight_start: Weight at ``ramp_start_iter``; ignored when the ramp is disabled.
            ramp_start_iter / ramp_end_iter: PPO iterations over which the weight ramps linearly.
            command_name / command_gate: Zero the style reward when the planar speed command is below
                ``command_gate``. ``command_name=None`` disables the gate.
            hidden_dims, activation, normalize_input, grad_penalty_coef: Discriminator settings.
            learning_rate: Discriminator Adam learning rate (kept below the policy's).
            num_updates_per_iter / batch_size: Discriminator minibatches per PPO update and their size.
            use_symmetry_augmentation: Mirror reference batches when PPO mirrors policy batches.
            device: Torch device.
            _env: The environment; injected by the runner so the default joint state and the command
                manager can be read. Not part of the user-facing configuration.
        """
        self.obs_group = obs_group
        self.speed_matched_sampling = speed_matched_sampling
        self.speed_match_window = speed_match_window
        self.reward_weight_end = reward_weight
        self.reward_weight_start = reward_weight_start
        self.ramp_start_iter = ramp_start_iter
        self.ramp_end_iter = ramp_end_iter
        self.command_name = command_name
        self.command_gate = command_gate
        self.num_updates_per_iter = num_updates_per_iter
        self.batch_size = batch_size
        self.use_symmetry_augmentation = use_symmetry_augmentation
        self.device = device
        self._env = _env
        self._symmetry_fn = None

        self.data = AmpMotionData(motion_files, speed_range=speed_range, device=device)
        # Built in _lazy_prepare once the observation width is known.
        self.discriminator: AmpDiscriminator | None = None
        self._disc_kwargs = dict(
            hidden_dims=hidden_dims,
            activation=activation,
            normalize_input=normalize_input,
            grad_penalty_coef=grad_penalty_coef,
        )
        self.learning_rate = learning_rate
        self.optimizer: torch.optim.Optimizer | None = None
        self._pending_state: dict[str, Any] | None = None

        self.iteration = 0
        self._style_reward_sum = torch.zeros((), device=device)
        self._style_reward_count = 0
        # Command-gate mask of every environment step of the current rollout, in call order, and the
        # mask of the last step of the previous rollout (which produced this rollout's first observation).
        self._gate_masks: list[torch.Tensor] = []
        self._last_gate_mask: torch.Tensor | None = None
        self._gate_warned = False

    # -- setup --------------------------------------------------------------------------------

    def _lazy_prepare(self, amp_obs: torch.Tensor) -> None:
        if self.discriminator is not None:
            return
        # The first environment step happens inside the runner's ``torch.inference_mode()`` rollout
        # block. Anything created there would be an inference tensor, and the discriminator's
        # parameters and normalisation buffers are updated in place later, outside that block.
        with torch.inference_mode(False):
            self._build(amp_obs.shape[-1])

    def _build(self, num_features: int) -> None:
        self.discriminator = AmpDiscriminator(num_features, device=self.device, **self._disc_kwargs)
        self.optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.learning_rate)
        if self._pending_state is not None:
            self.discriminator.load_state_dict(self._pending_state["discriminator"])
            self.optimizer.load_state_dict(self._pending_state["optimizer"])
            self.iteration = self._pending_state.get("iteration", 0)
            self._pending_state = None

        if not self.data.is_prepared:
            if self._env is None:
                raise RuntimeError("AMPExtension needs the environment to read the robot's default joint state.")
            robot = self._env.unwrapped.scene["robot"]
            self.data.prepare(robot.data.default_joint_pos[0], robot.data.default_joint_vel[0])
            if self.data.features.shape[-1] != num_features:
                raise RuntimeError(
                    f"AMP feature width mismatch: environment group '{self.obs_group}' is {num_features} wide,"
                    f" the motion loader produced {self.data.features.shape[-1]}. The loader and the observation"
                    " group have drifted apart."
                )
            print(self.data.describe())

    def set_symmetry_fn(self, fn) -> None:
        """Register the PPO data-augmentation function so reference batches can be mirrored too."""
        self._symmetry_fn = fn

    # -- reward -------------------------------------------------------------------------------

    @property
    def reward_weight(self) -> float:
        if self.ramp_end_iter <= self.ramp_start_iter:
            return self.reward_weight_end
        alpha = (self.iteration - self.ramp_start_iter) / (self.ramp_end_iter - self.ramp_start_iter)
        alpha = min(max(alpha, 0.0), 1.0)
        return self.reward_weight_start + alpha * (self.reward_weight_end - self.reward_weight_start)

    def on_env_step(self, obs, rewards, dones, extras):
        amp_obs = obs[self.obs_group]
        self._lazy_prepare(amp_obs)
        with torch.no_grad():
            style = self.discriminator.style_reward(amp_obs)
            # A reset refills the history with one state: that pair is not a movement.
            style = style * (1.0 - dones.float().to(style.device).view(-1))
            if self.command_name is not None and self._env is not None:
                cmd = self._env.unwrapped.command_manager.get_command(self.command_name)
                gated = torch.linalg.vector_norm(cmd[:, :2], dim=-1) >= self.command_gate
                style = style * gated.float().to(style.device)
            else:
                gated = torch.ones(style.shape[0], dtype=torch.bool, device=style.device)
        self._gate_masks.append(gated.to(style.device))
        self._style_reward_sum += style.sum()
        self._style_reward_count += style.numel()
        # Isaac Lab's RewardManager multiplies every task weight by step_dt (0.02 s), so scale the
        # style term the same way; ``reward_weight`` is then in the same units as the task reward
        # weights (2.0 = parity with ``track_lin_vel_xy``). Before 2026-09-22 the dt was missing,
        # so 0.2 acted like a weight of 10 and the style term outweighed the whole task reward.
        dt = self._env.unwrapped.step_dt if self._env is not None else 1.0
        return rewards + self.reward_weight * dt * style.to(rewards.device).view_as(rewards)

    # -- discriminator training ----------------------------------------------------------------

    def _rollout_transitions(self, storage) -> torch.Tensor:
        """Valid ``amp`` pairs from the rollout: not right after a reset, and command-gated.

        ``storage.observations[i]`` is the observation the policy acted on at step ``i``, i.e. the
        one produced by environment step ``i - 1``. The reset flag and the command gate recorded at
        that earlier step are therefore what decide whether pair ``i`` is usable; pair 0 is decided
        by the last step of the previous rollout.
        """
        obs = storage.observations[self.obs_group]  # (steps, envs, F)
        dones = storage.dones.squeeze(-1).bool()  # (steps, envs)
        valid = torch.ones_like(dones)
        valid[1:] &= ~dones[:-1]

        masks, self._gate_masks = self._gate_masks, []
        if len(masks) == obs.shape[0]:
            # No previous rollout means obs[0] is the initial reset observation, not a movement.
            prev = self._last_gate_mask if self._last_gate_mask is not None else torch.zeros_like(masks[0])
            gate = torch.stack([prev] + masks[:-1]).to(valid.device)
            valid &= gate
            self._last_gate_mask = masks[-1]
        elif not self._gate_warned:
            import warnings

            warnings.warn(
                f"AMPExtension recorded {len(masks)} command-gate masks for a rollout of {obs.shape[0]} steps;"
                " discriminator batches are not command-gated.",
                RuntimeWarning,
            )
            self._gate_warned = True
        if not valid.any():
            return obs[:0]
        return obs[valid]

    def _augment(self, policy_batch: torch.Tensor, reference_batch: torch.Tensor):
        if not (self.use_symmetry_augmentation and self._symmetry_fn is not None and self._env is not None):
            return policy_batch, reference_batch
        pol_td = TensorDict({self.obs_group: policy_batch}, batch_size=[policy_batch.shape[0]])
        ref_td = TensorDict({self.obs_group: reference_batch}, batch_size=[reference_batch.shape[0]])
        pol_aug, _ = self._symmetry_fn(env=self._env, obs=pol_td, actions=None)
        ref_aug, _ = self._symmetry_fn(env=self._env, obs=ref_td, actions=None)
        return pol_aug[self.obs_group], ref_aug[self.obs_group]

    def after_update(self, storage) -> dict[str, float]:
        if self.discriminator is None:
            return {}
        policy_all = self._rollout_transitions(storage)
        logs: dict[str, list[float]] = {}
        num_pairs = storage.observations[self.obs_group].shape[0] * storage.observations[self.obs_group].shape[1]
        gated_fraction = policy_all.shape[0] / max(num_pairs, 1)
        if policy_all.shape[0] == 0:
            # Nothing commanded fast enough this rollout (early curriculum): the discriminator waits.
            self.iteration += 1
            self._style_reward_sum.zero_()
            self._style_reward_count = 0
            return {"amp/reward_weight": self.reward_weight, "amp/gated_fraction": 0.0, "amp/style_reward_mean": 0.0}
        logs["amp/gated_fraction"] = [gated_fraction]
        for _ in range(self.num_updates_per_iter):
            idx = torch.randint(0, policy_all.shape[0], (self.batch_size,), device=policy_all.device)
            policy_batch = policy_all[idx]
            policy_speed = transition_speed(policy_batch)
            if self.speed_matched_sampling:
                reference_batch = self.data.sample_matched(policy_speed, self.speed_match_window)
            else:
                reference_batch = self.data.sample(self.batch_size)
            reference_batch = reference_batch.to(policy_batch.device)
            # How well the two sides agree on speed: with matching this should sit near zero except
            # where the clips have no data at the policy's speed.
            speed_err = (policy_speed - transition_speed(reference_batch)).abs().mean().item()
            logs.setdefault("amp/speed_match_err", []).append(speed_err)
            logs.setdefault("amp/policy_speed_mean", []).append(policy_speed.mean().item())
            policy_batch, reference_batch = self._augment(policy_batch, reference_batch)

            self.discriminator.update_normalization(policy_batch, reference_batch)
            loss, info = self.discriminator.loss(policy_batch, reference_batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            for k, v in info.items():
                logs.setdefault(k, []).append(v)

        out = {k: sum(v) / len(v) for k, v in logs.items()}
        out["amp/reward_weight"] = self.reward_weight  # the weight this rollout was collected with
        self.iteration += 1
        if self._style_reward_count:
            out["amp/style_reward_mean"] = (self._style_reward_sum / self._style_reward_count).item()
        self._style_reward_sum.zero_()
        self._style_reward_count = 0
        return out

    # -- bookkeeping --------------------------------------------------------------------------

    def train(self, mode: bool = True) -> None:
        if self.discriminator is not None:
            self.discriminator.train(mode)

    def parameters(self):
        return iter(()) if self.discriminator is None else self.discriminator.parameters()

    def state_dict(self) -> dict[str, Any]:
        if self.discriminator is None:
            return self._pending_state or {}
        return {
            "discriminator": self.discriminator.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "iteration": self.iteration,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not state:
            return
        if self.discriminator is None:
            # The discriminator's width comes from the first observation; apply once it exists.
            self._pending_state = state
            return
        self.discriminator.load_state_dict(state["discriminator"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.iteration = state.get("iteration", self.iteration)
