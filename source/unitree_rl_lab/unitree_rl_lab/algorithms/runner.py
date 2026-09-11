"""Runner for :class:`~unitree_rl_lab.algorithms.extensions.ExtendedPPO`.

:class:`rsl_rl.runners.OnPolicyRunner` resolves the algorithm class with ``eval(class_name)`` in
its own module namespace, so a class defined outside rsl_rl cannot be selected by name. This
subclass rebuilds :meth:`_construct_algorithm` -- the same resolve steps in the same order -- and
constructs :class:`ExtendedPPO` directly. Everything else (rollout loop, logging, checkpoints)
stays rsl_rl's, with three small additions so that extensions are saved, restored and switched
between train and eval mode alongside the policy.

Selected through ``class_name = "ExtendedOnPolicyRunner"`` in the agent configuration; see
:mod:`unitree_rl_lab.algorithms.registry`.
"""

from __future__ import annotations

import torch
import warnings

from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, resolve_rnd_config, resolve_symmetry_config
from rsl_rl.runners import OnPolicyRunner

from unitree_rl_lab.algorithms.extensions import ExtendedPPO

_POLICY_CLASSES = {"ActorCritic": ActorCritic, "ActorCriticRecurrent": ActorCriticRecurrent}


class ExtendedOnPolicyRunner(OnPolicyRunner):
    def _construct_algorithm(self, obs) -> ExtendedPPO:
        # Same resolve steps as OnPolicyRunner._construct_algorithm.
        self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)
        if self.cfg.get("empirical_normalization") is not None:
            warnings.warn(
                "The `empirical_normalization` parameter is deprecated. Please set `actor_obs_normalization` and "
                "`critic_obs_normalization` as part of the `policy` configuration instead.",
                DeprecationWarning,
            )
            if self.policy_cfg.get("actor_obs_normalization") is None:
                self.policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
            if self.policy_cfg.get("critic_obs_normalization") is None:
                self.policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]

        # Extensions that read the environment (default joint state, command manager) get it here,
        # the way resolve_symmetry_config injects it for the symmetry augmentation.
        amp_cfg = self.alg_cfg.get("amp_cfg")
        if amp_cfg is not None:
            amp_cfg["_env"] = self.env

        policy_class_name = self.policy_cfg.pop("class_name")
        try:
            policy_class = _POLICY_CLASSES[policy_class_name]
        except KeyError:
            raise ValueError(f"Unsupported policy class {policy_class_name!r}; expected one of {list(_POLICY_CLASSES)}.")
        policy = policy_class(obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg).to(self.device)

        alg_class_name = self.alg_cfg.pop("class_name", "ExtendedPPO")
        if alg_class_name != "ExtendedPPO":
            raise ValueError(f"ExtendedOnPolicyRunner builds ExtendedPPO, but algorithm.class_name is {alg_class_name!r}.")
        alg = ExtendedPPO(policy, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg)

        # Let extensions mirror their own batches with the policy's augmentation function.
        if alg.symmetry and alg.symmetry.get("use_data_augmentation"):
            for ext in alg.extensions:
                if hasattr(ext, "set_symmetry_fn"):
                    ext.set_symmetry_fn(alg.symmetry["data_augmentation_func"])

        alg.init_storage("rl", self.env.num_envs, self.num_steps_per_env, obs, [self.env.num_actions])
        return alg

    # -- checkpoints: extension state travels with the policy, under its own key ----------------

    def save(self, path: str, infos=None):
        super().save(path, infos)
        ext_state = {ext.name: ext.state_dict() for ext in self.alg.extensions}
        if any(ext_state.values()):
            saved = torch.load(path, weights_only=False)
            saved["extensions_state_dict"] = ext_state
            torch.save(saved, path)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None):
        infos = super().load(path, load_optimizer=load_optimizer, map_location=map_location)
        loaded = torch.load(path, weights_only=False, map_location=map_location)
        for ext in self.alg.extensions:
            ext.load_state_dict(loaded.get("extensions_state_dict", {}).get(ext.name, {}))
        return infos

    # -- train / eval switches ------------------------------------------------------------------

    def train_mode(self):
        super().train_mode()
        for ext in self.alg.extensions:
            ext.train(True)

    def eval_mode(self):
        super().eval_mode()
        for ext in self.alg.extensions:
            ext.train(False)
