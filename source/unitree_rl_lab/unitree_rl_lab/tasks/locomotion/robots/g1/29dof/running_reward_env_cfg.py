"""Running task with energy-efficiency reward shaping.

This is the pure-reward baseline in ``running_env_cfg.py`` plus two normalized actuation
cost terms, and nothing else. Everything the two configurations share -- scene, events,
commands, actions, observations, terminations and all 21 baseline reward terms -- is
inherited rather than copied, so the only difference between the two runs is the
intervention being measured.

Measured against the baseline at matched training budget (both at iteration 25500,
256 envs, flat ground, fixed command speed): cost of transport down 12.4-12.5% across
0.5-2.5 m/s, sum (tau/tau_limit)^2 down ~24%, survival 100%, at the cost of roughly
0.014 m/s of extra velocity tracking error at low speed.
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .running_env_cfg import CurriculumCfg as BaseCurriculumCfg
from .running_env_cfg import RewardsCfg as BaseRewardsCfg
from .running_env_cfg import RobotEnvCfg as BaseRobotEnvCfg
from .running_env_cfg import RobotPlayEnvCfg as BaseRobotPlayEnvCfg

# Both cost terms are normalized (power by ENERGY_POWER_REF_W, torque by each actuator's
# effort limit), so the weights below are on the same scale as the task rewards
# (track_lin_vel_xy = 2.0) instead of the usual hard-to-read 1e-5 range. They ramp in
# linearly so the running gait is established under the task rewards first and only then
# gets traded down for efficiency -- a strong cost penalty from step 0 collapses the
# flight phase into a walk before the policy ever learns to run.
# ENERGY_POWER_REF_W is calibrated from scripts/rsl_rl/eval_energy.py on the pure-reward
# baseline (2026-09-08_17-13-58_try6/model_25500): 230 W at 1.0 m/s and 446 W at 2.0 m/s,
# with sum (tau/tau_limit)^2 at 0.95 and 1.45 respectively. At 2.0 m/s the two terms are
# then worth about -0.22 and -0.15, i.e. ~10% of the roughly 3.6 of positive reward --
# enough to bend the gait without drowning out velocity tracking.
PPO_STEPS_PER_ITER = 24
ENERGY_RAMP_START_ITER = 1500
ENERGY_RAMP_END_ITER = 4000
ENERGY_POWER_REF_W = 400.0
ENERGY_POWER_WEIGHT = -0.20
ENERGY_TORQUE_WEIGHT = -0.10


def _ramp(term_name: str, end_weight: float) -> dict:
    """Ramp parameters shared by every variant, so the schedule can never drift."""
    return {
        "term_name": term_name,
        "start_weight": 0.0,
        "end_weight": end_weight,
        "start_step": ENERGY_RAMP_START_ITER * PPO_STEPS_PER_ITER,
        "end_step": ENERGY_RAMP_END_ITER * PPO_STEPS_PER_ITER,
    }


@configclass
class RewardsCfg(BaseRewardsCfg):
    """Baseline reward terms plus the two actuation-cost penalties."""

    # Both start at zero weight -- RewardManager skips zero-weight terms outright, so
    # before the ramp starts this is bit-identical to the baseline objective.
    # joint_power is the mechanical work; joint_torque is the I^2 R share that mechanical
    # power misses, and is what reaches joints that hold load without doing work
    # (waist_roll draws 0.7% of total power but sits at 79% of its effort limit).
    joint_power = RewTerm(
        func=mdp.joint_power_l1,
        weight=0.0,
        params={"power_ref": ENERGY_POWER_REF_W},
    )
    joint_torque = RewTerm(func=mdp.joint_torque_normalized_l2, weight=0.0)


@configclass
class CurriculumCfg(BaseCurriculumCfg):
    """Baseline curriculum plus the linear ramp that switches the cost terms on."""

    joint_power_weight = CurrTerm(
        func=mdp.reward_weight_linear_schedule,
        params=_ramp("joint_power", ENERGY_POWER_WEIGHT),
    )
    joint_torque_weight = CurrTerm(
        func=mdp.reward_weight_linear_schedule,
        params=_ramp("joint_torque", ENERGY_TORQUE_WEIGHT),
    )


@configclass
class RobotEnvCfg(BaseRobotEnvCfg):
    rewards: RewardsCfg = RewardsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()


@configclass
class RobotPlayEnvCfg(BaseRobotPlayEnvCfg):
    rewards: RewardsCfg = RewardsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()


"""
Single-term ablations.

The combined configuration above changes two things at once, so it cannot say which of the
two energy models produced the result. These two variants isolate them. Everything else --
the ramp schedule, power_ref, and all 21 baseline reward terms -- is shared with the
combined variant, so the only difference between the three runs is which cost term is on.
"""


@configclass
class PowerOnlyRewardsCfg(BaseRewardsCfg):
    """Mechanical work only: sum |tau * qdot| / power_ref."""

    joint_power = RewTerm(
        func=mdp.joint_power_l1,
        weight=0.0,
        params={"power_ref": ENERGY_POWER_REF_W},
    )


@configclass
class PowerOnlyCurriculumCfg(BaseCurriculumCfg):
    joint_power_weight = CurrTerm(
        func=mdp.reward_weight_linear_schedule,
        params=_ramp("joint_power", ENERGY_POWER_WEIGHT),
    )


@configclass
class TorqueOnlyRewardsCfg(BaseRewardsCfg):
    """Resistive heating only: sum (tau / tau_limit)^2."""

    joint_torque = RewTerm(func=mdp.joint_torque_normalized_l2, weight=0.0)


@configclass
class TorqueOnlyCurriculumCfg(BaseCurriculumCfg):
    joint_torque_weight = CurrTerm(
        func=mdp.reward_weight_linear_schedule,
        params=_ramp("joint_torque", ENERGY_TORQUE_WEIGHT),
    )


@configclass
class PowerOnlyEnvCfg(BaseRobotEnvCfg):
    rewards: PowerOnlyRewardsCfg = PowerOnlyRewardsCfg()
    curriculum: PowerOnlyCurriculumCfg = PowerOnlyCurriculumCfg()


@configclass
class PowerOnlyPlayEnvCfg(BaseRobotPlayEnvCfg):
    rewards: PowerOnlyRewardsCfg = PowerOnlyRewardsCfg()
    curriculum: PowerOnlyCurriculumCfg = PowerOnlyCurriculumCfg()


@configclass
class TorqueOnlyEnvCfg(BaseRobotEnvCfg):
    rewards: TorqueOnlyRewardsCfg = TorqueOnlyRewardsCfg()
    curriculum: TorqueOnlyCurriculumCfg = TorqueOnlyCurriculumCfg()


@configclass
class TorqueOnlyPlayEnvCfg(BaseRobotPlayEnvCfg):
    rewards: TorqueOnlyRewardsCfg = TorqueOnlyRewardsCfg()
    curriculum: TorqueOnlyCurriculumCfg = TorqueOnlyCurriculumCfg()


"""
CoT variant: the combined penalties plus a speed-normalized power cost.

The two terms above penalize power regardless of how fast the robot was asked to go, so the
slowest -- and least efficient -- commands get the least pressure. ``joint_cot`` adds the
adaptive energy regularization of Liang et al. (2024), a per-step cost of transport, on top of
them. It is a separate task so the ablation configurations stay untouched.

Constant weights and no ramp. With the current environment the running gait emerges after
the old ramp had already finished, so the ramp never did what it was designed for; if this
variant fails to stand within ~8k iterations, a gate on episode length is the fix, not a
linear schedule.

Weight: at 2 m/s the baseline's 446 W gives an exponent of 446/(2800*1.98) = 0.080 and a cost
of 0.077, so -1.5 adds -0.12 to the existing -0.37 of energy pressure (13% of the ~3.6 of
positive reward, up from 10%). At 0.5 m/s the same weight roughly doubles the pressure, which
is the point. -3.0 would reproduce the power term's 2 m/s pressure on its own, i.e. the value
to use with ENERGY_POWER_WEIGHT set to 0 for a clean replace-instead-of-add comparison.
"""

ENERGY_COT_SIGMA_LIN = 2800.0  # N; 8.5 x G1 weight, matches the paper's regime
ENERGY_COT_SIGMA_ANG = 1400.0  # N*m/rad; paper ratio sigma_ang = 0.5 * sigma_lin
ENERGY_COT_V_FLOOR = 0.3  # m/s; same floor as the training metric's cot_min_speed
ENERGY_COT_WEIGHT = -1.5


@configclass
class CotRewardsCfg(BaseRewardsCfg):
    """Power + torque at their final weights, plus the speed-normalized power cost."""

    joint_power = RewTerm(
        func=mdp.joint_power_l1,
        weight=ENERGY_POWER_WEIGHT,
        params={"power_ref": ENERGY_POWER_REF_W},
    )
    joint_torque = RewTerm(func=mdp.joint_torque_normalized_l2, weight=ENERGY_TORQUE_WEIGHT)
    joint_cot = RewTerm(
        func=mdp.joint_power_cot_exp,
        weight=ENERGY_COT_WEIGHT,
        params={
            "sigma_lin": ENERGY_COT_SIGMA_LIN,
            "sigma_ang": ENERGY_COT_SIGMA_ANG,
            "v_floor": ENERGY_COT_V_FLOOR,
        },
    )


@configclass
class CotEnvCfg(BaseRobotEnvCfg):
    # Curriculum is inherited from the baseline: terrain and command levels only, no ramp.
    rewards: CotRewardsCfg = CotRewardsCfg()


@configclass
class CotPlayEnvCfg(BaseRobotPlayEnvCfg):
    rewards: CotRewardsCfg = CotRewardsCfg()
