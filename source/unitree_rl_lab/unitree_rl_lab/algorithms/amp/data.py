"""Reference transition pairs for the AMP discriminator.

The discriminator is trained to separate transitions produced by the policy from transitions
found in human running motions. Both sides must be described by *identical* features, so this
module reproduces the ``amp`` observation group of the environment
(:mod:`unitree_rl_lab.tasks.locomotion.mdp.amp_observations`) from the motion npz files.

Feature layout
--------------
The observation manager flattens each term's history separately and only then concatenates the
terms, so the 134-dimensional vector is *not* ``[s_{t-1} | s_t]`` but a sequence of per-term
blocks, oldest sample first::

    [  0:  3] base_lin_vel(t-1)        [  3:  6] base_lin_vel(t)
    [  6:  9] base_ang_vel(t-1)        [  9: 12] base_ang_vel(t)
    [ 12: 15] projected_gravity(t-1)   [ 15: 18] projected_gravity(t)
    [ 18: 47] joint_pos_rel(t-1)       [ 47: 76] joint_pos_rel(t)
    [ 76:105] joint_vel_rel(t-1)       [105:134] joint_vel_rel(t)

Getting this wrong is silent: the discriminator would still train, just on a permutation of the
reference features, and the style reward would be meaningless.

Correspondence with the environment
-----------------------------------
* ``base_lin_vel`` / ``base_ang_vel`` -- the environment reports ``root_com_{lin,ang}_vel_b``,
  i.e. the *centre-of-mass* velocity expressed in the *link* frame. The npz stores
  ``body_lin_vel_w`` / ``body_ang_vel_w`` (centre-of-mass velocities) and ``body_quat_w`` (link
  orientation), so rotating body 0's velocity by the inverse of body 0's quaternion reproduces
  the observation exactly. Body 0 is the root: its height matches the root height column of the
  source csv to within float precision.
* ``joint_pos_rel`` subtracts the robot's *current* default joint position, which an event term
  may randomise per environment. The default is therefore taken from the live articulation at
  training time (:meth:`prepare`) rather than hard-coded here.
* ``joint_vel_rel`` subtracts ``default_joint_vel``, which is zero for this robot but is passed
  in for the same reason.
* The npz stores velocities computed by the simulator during the replay, so they are used
  directly instead of differentiating positions.

Speed gate
----------
The takes are several minutes long and mix running with standing around and with bursts faster
than the task ever commands. Both extremes hurt:

* Idle transitions (~20% of the data) teach the discriminator that standing still is the most
  human-looking thing to do, which is the opposite of what the running task needs.
* Transitions above the command limit (~13%) ask for a stride the velocity-tracking reward will
  never allow, so the two rewards would pull against each other.

Both are dropped per transition rather than by trimming frame ranges, because the idle stretches
are scattered through the takes (16 separate stretches in one sprint clip) rather than sitting at
the ends. A pair is kept only if *both* of its frames pass the gate.
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass

AMP_TERM_DIMS: tuple[tuple[str, int], ...] = (
    ("base_lin_vel", 3),
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("joint_pos_rel", 29),
    ("joint_vel_rel", 29),
)
"""Terms of the ``amp`` observation group and their per-frame widths, in group order."""

AMP_HISTORY_LENGTH = 2
FRAME_DIM = sum(dim for _, dim in AMP_TERM_DIMS)
FEATURE_DIM = FRAME_DIM * AMP_HISTORY_LENGTH

ROOT_BODY_INDEX = 0
GRAVITY_VEC_W = (0.0, 0.0, -1.0)


def quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate ``vec`` by the inverse of ``quat`` (w, x, y, z), matching ``isaaclab.utils.math``.

    Reimplemented here so that the statistics CLI runs without importing Isaac Sim.
    """
    shape = vec.shape
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    xyz = quat[:, 1:]
    t = xyz.cross(vec, dim=-1) * 2
    return (vec - quat[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)


@dataclass
class ClipStats:
    """What a single clip contributed, for logging and for the ``--stats`` report."""

    name: str
    frames: int
    duration_s: float
    speed_mean: float
    speed_max: float
    kept_pairs: int
    total_pairs: int

    @property
    def kept_fraction(self) -> float:
        return self.kept_pairs / max(self.total_pairs, 1)


class AmpMotionData:
    """Reference transition pairs sampled by the AMP discriminator.

    Loading is split in two so that constructing the algorithm never touches the filesystem more
    than it has to: :meth:`__init__` reads the npz files and applies the speed gate, while
    :meth:`prepare` needs the robot's default joint state and therefore runs once the environment
    exists. Evaluation and replay never call :meth:`prepare`, so a missing or moved motion file
    cannot break scoring a trained checkpoint.
    """

    def __init__(
        self,
        motion_files: list[str] | tuple[str, ...],
        speed_range: tuple[float, float] = (0.2, 2.5),
        device: str = "cpu",
    ):
        if not motion_files:
            raise ValueError("AmpMotionData needs at least one motion file.")
        self.motion_files = list(motion_files)
        self.speed_range = speed_range
        self.device = device

        self._clips: list[dict[str, np.ndarray]] = []
        self.clip_stats: list[ClipStats] = []
        self._features: torch.Tensor | None = None

        import os

        missing = [p for p in self.motion_files if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError(
                "AMP reference npz file(s) not found:\n  "
                + "\n  ".join(missing)
                + "\nThe npz files are gitignored and must be rebuilt from the csv next to them:\n"
                "  python scripts/mimic/csv_to_npz.py -f <clip>.csv --input_fps 30 --headless"
            )
        for path in self.motion_files:
            self._clips.append(self._load_clip(path))

    def _load_clip(self, path: str) -> dict[str, np.ndarray]:
        """Read one npz and select the transition pairs that pass the speed gate."""
        with np.load(path) as raw:
            fps = float(raw["fps"][0])
            joint_pos = raw["joint_pos"].astype(np.float32)
            joint_vel = raw["joint_vel"].astype(np.float32)
            root_quat = raw["body_quat_w"][:, ROOT_BODY_INDEX].astype(np.float32)
            root_lin_vel = raw["body_lin_vel_w"][:, ROOT_BODY_INDEX].astype(np.float32)
            root_ang_vel = raw["body_ang_vel_w"][:, ROOT_BODY_INDEX].astype(np.float32)

        # Planar speed decides the gate: vertical motion is part of running, not a measure of it.
        speed = np.linalg.norm(root_lin_vel[:, :2], axis=-1)
        lo, hi = self.speed_range
        in_range = (speed >= lo) & (speed <= hi)
        # A pair is usable only if both of its frames are in range, and pairs never straddle clips.
        keep = in_range[:-1] & in_range[1:]
        indices = np.nonzero(keep)[0]

        name = path.rsplit("/", 1)[-1].replace(".npz", "")
        self.clip_stats.append(
            ClipStats(
                name=name,
                frames=len(joint_pos),
                duration_s=len(joint_pos) / fps,
                speed_mean=float(speed.mean()),
                speed_max=float(speed.max()),
                kept_pairs=int(len(indices)),
                total_pairs=int(len(keep)),
            )
        )
        return {
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "root_quat": root_quat,
            "root_lin_vel": root_lin_vel,
            "root_ang_vel": root_ang_vel,
            "indices": indices,
        }

    def prepare(self, default_joint_pos: torch.Tensor, default_joint_vel: torch.Tensor) -> None:
        """Build the feature tensor using the robot's default joint state.

        Args:
            default_joint_pos: Default joint positions, shape (num_joints,), in the simulator's
                joint order -- the same order the npz was recorded in.
            default_joint_vel: Default joint velocities, shape (num_joints,).
        """
        if self._features is not None:
            return

        default_joint_pos = default_joint_pos.detach().to("cpu", torch.float32).reshape(-1)
        default_joint_vel = default_joint_vel.detach().to("cpu", torch.float32).reshape(-1)

        per_clip = [self._clip_features(clip, default_joint_pos, default_joint_vel) for clip in self._clips]
        features = torch.cat([f for f in per_clip if len(f)], dim=0)
        if not len(features):
            raise ValueError(
                f"The speed gate {self.speed_range} removed every transition in "
                f"{len(self.motion_files)} clip(s). Widen the range or check the motion files."
            )
        self._features = features.to(self.device)
        # The raw arrays are no longer needed once the features exist.
        self._clips = []

    def _clip_features(
        self, clip: dict[str, np.ndarray], default_joint_pos: torch.Tensor, default_joint_vel: torch.Tensor
    ) -> torch.Tensor:
        indices = clip["indices"]
        if not len(indices):
            return torch.empty(0, FEATURE_DIM)

        quat = torch.from_numpy(clip["root_quat"])
        gravity = torch.tensor(GRAVITY_VEC_W).expand(len(quat), 3)
        frame = {
            "base_lin_vel": quat_apply_inverse(quat, torch.from_numpy(clip["root_lin_vel"])),
            "base_ang_vel": quat_apply_inverse(quat, torch.from_numpy(clip["root_ang_vel"])),
            "projected_gravity": quat_apply_inverse(quat, gravity),
            "joint_pos_rel": torch.from_numpy(clip["joint_pos"]) - default_joint_pos,
            "joint_vel_rel": torch.from_numpy(clip["joint_vel"]) - default_joint_vel,
        }

        prev = torch.from_numpy(indices)
        curr = prev + 1
        # Per-term history blocks, oldest sample first -- see the module docstring.
        blocks = []
        for term, _ in AMP_TERM_DIMS:
            blocks.append(frame[term][prev])
            blocks.append(frame[term][curr])
        return torch.cat(blocks, dim=-1)

    @property
    def is_prepared(self) -> bool:
        return self._features is not None

    @property
    def num_transitions(self) -> int:
        return 0 if self._features is None else int(self._features.shape[0])

    @property
    def features(self) -> torch.Tensor:
        if self._features is None:
            raise RuntimeError("AmpMotionData.prepare() must be called before the features are used.")
        return self._features

    def sample(self, batch_size: int) -> torch.Tensor:
        """Draw ``batch_size`` reference transitions uniformly, with replacement.

        Sampling is uniform over transitions rather than over clips, so a longer clip contributes
        proportionally more -- the same weighting the policy's own rollouts have.
        """
        features = self.features
        idx = torch.randint(0, features.shape[0], (batch_size,), device=features.device)
        return features[idx]

    def describe(self) -> str:
        """One line per clip plus a total, for the training log."""
        lines = [f"AMP reference data: {len(self.clip_stats)} clip(s), speed gate {self.speed_range} m/s"]
        for s in self.clip_stats:
            lines.append(
                f"  {s.name:<20} {s.duration_s:7.1f}s  v_mean {s.speed_mean:4.2f}  v_max {s.speed_max:4.2f}"
                f"  kept {s.kept_pairs:6d}/{s.total_pairs:6d} ({s.kept_fraction * 100:4.1f}%)"
            )
        kept = sum(s.kept_pairs for s in self.clip_stats)
        total = sum(s.total_pairs for s in self.clip_stats)
        lines.append(f"  {'TOTAL':<20} {'':7}   {'':14}  kept {kept:6d}/{total:6d} ({kept / max(total, 1) * 100:4.1f}%)")
        return "\n".join(lines)


def _main() -> None:
    """Report what the loader would feed the discriminator, without starting Isaac Sim.

    ``python -m unitree_rl_lab.algorithms.amp.data <clip>.npz ... [--speed-range LO HI]``

    Joint blocks are reported in absolute terms: ``joint_pos_rel`` subtracts the robot's default
    pose, which is only known once the environment exists, and a constant offset does not change
    the spread. Use this to check that the clips cover the commanded speed range and that no
    feature is orders of magnitude away from the others.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Summarise AMP reference motions.")
    parser.add_argument("motion_files", nargs="+", help="Motion npz files produced by csv_to_npz.py.")
    parser.add_argument("--speed-range", nargs=2, type=float, default=(0.2, 2.5), metavar=("LO", "HI"))
    args = parser.parse_args()

    data = AmpMotionData(args.motion_files, speed_range=tuple(args.speed_range))
    print(data.describe())

    # Zero defaults: the spread is what matters here, not the offset.
    num_joints = data._clips[0]["joint_pos"].shape[1]
    data.prepare(torch.zeros(num_joints), torch.zeros(num_joints))
    features = data.features
    print(f"\nfeatures: {tuple(features.shape)}  (expected (N, {FEATURE_DIM}))")

    print(f"\n{'block':<26}{'range':>18}{'mean':>9}{'std':>9}{'|max|':>9}")
    offset = 0
    for term, dim in AMP_TERM_DIMS:
        for age in ("t-1", "t"):
            block = features[:, offset : offset + dim]
            print(
                f"  {term + '(' + age + ')':<24}[{offset:3d}:{offset + dim:3d}]"
                f"{block.mean():9.3f}{block.std():9.3f}{block.abs().max():9.3f}"
            )
            offset += dim

    # The two halves of a pair are one control step apart, so they must be close but not equal.
    print("\nsanity: consecutive frames differ but stay close")
    offset = 0
    for term, dim in AMP_TERM_DIMS:
        prev = features[:, offset : offset + dim]
        curr = features[:, offset + dim : offset + 2 * dim]
        delta = (curr - prev).abs()
        print(f"  {term:<24} |s_t - s_(t-1)|  mean {delta.mean():7.4f}  max {delta.max():8.4f}")
        offset += 2 * dim


if __name__ == "__main__":
    _main()
