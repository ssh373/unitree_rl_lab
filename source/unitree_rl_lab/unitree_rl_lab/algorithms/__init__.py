"""Algorithm-layer extensions to rsl_rl (runner registry, AMP, ...).

Nothing in here is imported by the task packages under ``unitree_rl_lab.tasks``. The training
scripts import :mod:`unitree_rl_lab.algorithms.registry` and resolve runner classes lazily, so a
task that never asks for an experimental runner never loads its code.
"""
