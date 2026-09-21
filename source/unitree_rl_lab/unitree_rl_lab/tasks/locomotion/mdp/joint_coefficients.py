"""Expand a {joint-name-regex: value} mapping to a per-joint tensor."""

from __future__ import annotations

import re

import torch

from isaaclab.assets import Articulation


def joint_coefficients(asset: Articulation, mapping: dict[str, float], joint_ids=slice(None)) -> torch.Tensor:
    """Return one coefficient per selected joint, in joint order.

    Patterns are tried in mapping order and the first match wins. Every selected joint must
    match some pattern; a joint without a coefficient raises ``KeyError`` so a typo in the
    mapping cannot silently zero a motor's cost.
    """
    if isinstance(joint_ids, slice):
        names = asset.joint_names[joint_ids]
    else:
        names = [asset.joint_names[i] for i in joint_ids]
    out = torch.zeros(len(names), device=asset.device)
    for j, name in enumerate(names):
        for pattern, value in mapping.items():
            if re.search(pattern, name):
                out[j] = value
                break
        else:
            raise KeyError(f"no coefficient matches joint '{name}'")
    return out
