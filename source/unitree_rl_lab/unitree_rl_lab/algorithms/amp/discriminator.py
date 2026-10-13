"""AMP discriminator: tells policy transitions from reference transitions and turns its verdict
into a style reward.

Follows Peng et al., "AMP: Adversarial Motion Priors for Stylized Physics-Based Character
Control" (2021):

* least-squares GAN objective -- reference transitions are regressed to +1, policy transitions
  to -1, which is far better behaved than the saturating cross-entropy loss;
* a gradient penalty on the *reference* inputs, without which the discriminator sharpens until
  its reward carries no gradient for the policy (the AMP paper's single most important trick);
* style reward ``max(0, 1 - 0.25 * (D(s, s') - 1)^2)`` in ``[0, 1]``: 1 when the discriminator is
  convinced the transition is human, 0 once it is sure it is not.

Input normalisation is on by default. The reference statistics (``python -m
unitree_rl_lab.algorithms.amp.data``) put ``joint_vel_rel`` at roughly four times the spread of
``joint_pos_rel`` and ``projected_gravity``, so an unnormalised discriminator would attend to joint
velocities and little else. The same :class:`rsl_rl.networks.EmpiricalNormalization` that RND
uses is fed with both policy and reference batches so that neither side is favoured.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.networks import MLP, EmpiricalNormalization


class AmpDiscriminator(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_dims: tuple[int, ...] | list[int] = (1024, 512),
        activation: str = "relu",
        normalize_input: bool = True,
        grad_penalty_coef: float = 10.0,
        logit_reg_coef: float = 0.0,
        device: str = "cpu",
    ):
        """
        Args:
            num_features: Width of one transition pair (the flattened ``amp`` observation group).
            hidden_dims: Hidden layer sizes. The AMP paper uses (1024, 512).
            activation: Hidden activation.
            normalize_input: Standardise inputs with running statistics before the MLP.
            grad_penalty_coef: Weight of the gradient penalty on reference inputs (paper: 10).
            logit_reg_coef: L2 penalty on the weights of the output (logit) layer, the ``disc_logit_reg``
                of the authors' Isaac Gym implementation (0.05 there). Keeps the logits from growing
                without bound once the two sides are easy to separate. 0 disables it.
            device: Torch device.
        """
        super().__init__()
        self.num_features = num_features
        self.grad_penalty_coef = grad_penalty_coef
        self.logit_reg_coef = logit_reg_coef
        self.device = device

        if normalize_input:
            self.input_normalizer = EmpiricalNormalization(shape=[num_features], until=1.0e8)
        else:
            self.input_normalizer = nn.Identity()
        self.trunk = MLP(num_features, 1, list(hidden_dims), activation=activation)
        self.to(device)

    # -- statistics ---------------------------------------------------------------------------

    def update_normalization(self, *batches: torch.Tensor) -> None:
        """Feed raw transition batches (policy and reference alike) to the running statistics."""
        if isinstance(self.input_normalizer, EmpiricalNormalization):
            for batch in batches:
                self.input_normalizer.update(batch)

    # -- inference ----------------------------------------------------------------------------

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Discriminator output ``D(s, s')`` for raw transitions. Shape (N, 1)."""
        return self.trunk(self.input_normalizer(features))

    def style_reward(self, features: torch.Tensor) -> torch.Tensor:
        """Per-transition style reward in ``[0, 1]``. Shape (N,). Call under ``torch.no_grad``."""
        d = self.forward(features).squeeze(-1)
        return torch.clamp(1.0 - 0.25 * (d - 1.0) ** 2, min=0.0)

    # -- training -----------------------------------------------------------------------------

    def loss(self, policy_features: torch.Tensor, reference_features: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """LSGAN loss plus gradient penalty, with diagnostics for the training log.

        Returns:
            The total loss to back-propagate, and a dictionary with its components, the mean
            discriminator outputs and the classification accuracy on each side. Accuracy that
            saturates at 1.0 within a few iterations is the signature of a feature mismatch
            between the environment and the motion loader.
        """
        reference_input = self.input_normalizer(reference_features).detach().requires_grad_(True)
        policy_input = self.input_normalizer(policy_features).detach()

        d_ref = self.trunk(reference_input)
        d_pol = self.trunk(policy_input)

        loss_ref = 0.5 * torch.mean((d_ref - 1.0) ** 2)
        loss_pol = 0.5 * torch.mean((d_pol + 1.0) ** 2)

        # Gradient penalty on reference samples (AMP paper, eq. 8): keeps D smooth where the
        # policy has to move towards, so the style reward stays informative.
        grad = torch.autograd.grad(
            outputs=d_ref.sum(), inputs=reference_input, create_graph=True, retain_graph=True
        )[0]
        grad_penalty = torch.mean(torch.sum(grad**2, dim=-1))

        logit_reg = torch.sum(torch.square(self.trunk[-1].weight))  # the MLP's last layer is the logit layer
        total = loss_ref + loss_pol + self.grad_penalty_coef * grad_penalty + self.logit_reg_coef * logit_reg
        info = {
            "amp/loss_reference": loss_ref.item(),
            "amp/loss_policy": loss_pol.item(),
            "amp/grad_penalty": grad_penalty.item(),
            "amp/logit_reg": logit_reg.item(),
            "amp/d_reference": d_ref.mean().item(),
            "amp/d_policy": d_pol.mean().item(),
            "amp/acc_reference": (d_ref > 0).float().mean().item(),
            "amp/acc_policy": (d_pol < 0).float().mean().item(),
        }
        return total, info
