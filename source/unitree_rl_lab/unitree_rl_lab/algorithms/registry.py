"""Runner class registry.

``train.py``, ``play.py`` and ``eval_energy.py`` pick the rsl_rl runner from the ``class_name``
field of the agent configuration (:attr:`isaaclab_rl.rsl_rl.RslRlOnPolicyRunnerCfg.class_name`,
default ``"OnPolicyRunner"``). Entries are import paths rather than classes: a runner module is
imported the first time a configuration asks for it, so an experimental runner that fails to
import cannot break the tasks that never use it.
"""

from __future__ import annotations

import importlib

RUNNERS: dict[str, str] = {
    "OnPolicyRunner": "rsl_rl.runners:OnPolicyRunner",
    "DistillationRunner": "rsl_rl.runners:DistillationRunner",
}
"""Registered runners as ``name -> "package.module:ClassName"``."""


def register_runner(name: str, import_path: str) -> None:
    """Register the class at ``import_path`` (``"package.module:ClassName"``) under ``name``."""
    if ":" not in import_path:
        raise ValueError(f"import_path must look like 'package.module:ClassName', got {import_path!r}.")
    RUNNERS[name] = import_path


def get_runner_class(name: str) -> type:
    """Resolve a runner class by name, importing its module on first use."""
    try:
        import_path = RUNNERS[name]
    except KeyError:
        raise ValueError(f"Unknown runner class {name!r}. Registered runners: {sorted(RUNNERS)}.") from None
    module_name, class_name = import_path.split(":", 1)
    return getattr(importlib.import_module(module_name), class_name)
