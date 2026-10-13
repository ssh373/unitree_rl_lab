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

# Offset of the base_lin_vel(t) block: base_lin_vel is the first term, and its (t-1) sample comes first.
_LIN_VEL_T_OFFSET = AMP_TERM_DIMS[0][1]


def transition_speed(features: torch.Tensor) -> torch.Tensor:
    """Planar speed of the current frame of each transition, from the ``base_lin_vel(t)`` block.

    Used on both the policy and the reference side of the discriminator batch, so speed matching
    compares like with like. The base frame is close to level while running, so this is within a
    few percent of the world-frame planar speed the gate uses.
    """
    return torch.linalg.vector_norm(features[:, _LIN_VEL_T_OFFSET : _LIN_VEL_T_OFFSET + 2], dim=-1)


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
    speed_range: tuple[float, float] = (0.2, 2.5)

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
        clip_speed_ranges: list[tuple[float, float] | None] | None = None,
    ):
        """
        Args:
            motion_files: Motion npz files produced by ``csv_to_npz.py``.
            speed_range: Planar-speed gate applied to every clip that has no entry in
                ``clip_speed_ranges``.
            device: Torch device the feature tensor is moved to in :meth:`prepare`.
            clip_speed_ranges: Optional per-clip gate, one ``(lo, hi)`` per entry of ``motion_files``
                (``None`` entries fall back to ``speed_range``). Used to band the reference set by
                gait, e.g. walking takes below 1 m/s and running takes above it, so that the
                speed-matched sampler draws a walk at walking commands and a run at running
                commands instead of whichever gait the takes happen to hold at that speed.
        """
        if not motion_files:
            raise ValueError("AmpMotionData needs at least one motion file.")
        self.motion_files = list(motion_files)
        self.speed_range = tuple(speed_range)
        self.device = device
        if clip_speed_ranges is None:
            clip_speed_ranges = [None] * len(self.motion_files)
        if len(clip_speed_ranges) != len(self.motion_files):
            raise ValueError(
                f"clip_speed_ranges has {len(clip_speed_ranges)} entries for {len(self.motion_files)} motion files."
            )
        self.clip_speed_ranges: list[tuple[float, float]] = [
            self.speed_range if r is None else (float(r[0]), float(r[1])) for r in clip_speed_ranges
        ]

        self._clips: list[dict[str, np.ndarray]] = []
        self.clip_stats: list[ClipStats] = []
        self._features: torch.Tensor | None = None
        self._speeds: torch.Tensor | None = None

        import os

        missing = [p for p in self.motion_files if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError(
                "AMP reference npz file(s) not found:\n  "
                + "\n  ".join(missing)
                + "\nThe npz files are gitignored and must be rebuilt from the csv next to them:\n"
                "  python scripts/mimic/csv_to_npz.py -f <clip>.csv --input_fps 30 --headless"
            )
        for path, clip_range in zip(self.motion_files, self.clip_speed_ranges):
            self._clips.append(self._load_clip(path, clip_range))

    def _load_clip(self, path: str, speed_range: tuple[float, float]) -> dict[str, np.ndarray]:
        """Read one npz and select the transition pairs that pass the clip's speed gate."""
        with np.load(path) as raw:
            fps = float(raw["fps"][0])
            joint_pos = raw["joint_pos"].astype(np.float32)
            joint_vel = raw["joint_vel"].astype(np.float32)
            root_quat = raw["body_quat_w"][:, ROOT_BODY_INDEX].astype(np.float32)
            root_lin_vel = raw["body_lin_vel_w"][:, ROOT_BODY_INDEX].astype(np.float32)
            root_ang_vel = raw["body_ang_vel_w"][:, ROOT_BODY_INDEX].astype(np.float32)
            root_pos_z = raw["body_pos_w"][:, ROOT_BODY_INDEX, 2].astype(np.float32)

        # Planar speed decides the gate: vertical motion is part of running, not a measure of it.
        speed = np.linalg.norm(root_lin_vel[:, :2], axis=-1)
        lo, hi = speed_range
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
                speed_range=(float(lo), float(hi)),
            )
        )
        return {
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "root_quat": root_quat,
            "root_lin_vel": root_lin_vel,
            "root_ang_vel": root_ang_vel,
            "root_pos_z": root_pos_z,
            "indices": indices,
            "frame_indices": np.nonzero(in_range)[0],
        }

    def reset_states(self) -> dict[str, torch.Tensor]:
        """Every frame that passes its clip's speed gate, as the state a reset can write to the simulator.

        Returns ``root_pos_z``, ``root_quat`` (w, x, y, z, world), ``root_lin_vel_w``, ``root_ang_vel_w``,
        ``joint_pos`` and ``joint_vel`` stacked over clips, on ``self.device``. Used by the reference
        state initialisation event (AMP paper: episodes start from states sampled from the reference
        motions). Must be called before :meth:`prepare`, which drops the raw arrays.
        """
        if not self._clips:
            raise RuntimeError("reset_states() needs the raw clips; call it before prepare().")
        keys = {
            "root_pos_z": "root_pos_z",
            "root_quat": "root_quat",
            "root_lin_vel_w": "root_lin_vel",
            "root_ang_vel_w": "root_ang_vel",
            "joint_pos": "joint_pos",
            "joint_vel": "joint_vel",
        }
        out = {}
        for name, key in keys.items():
            parts = [clip[key][clip["frame_indices"]] for clip in self._clips if len(clip["frame_indices"])]
            out[name] = torch.from_numpy(np.concatenate(parts, axis=0)).to(self.device)
        return out

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
                f"The speed gate(s) {sorted(set(self.clip_speed_ranges))} removed every transition in "
                f"{len(self.motion_files)} clip(s). Widen the range or check the motion files."
            )
        # Sorted by speed so that speed-matched sampling is a pair of searchsorted calls.
        speeds = transition_speed(features)
        order = torch.argsort(speeds)
        self._features = features[order].to(self.device)
        self._speeds = speeds[order].to(self.device)
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

    def sample_matched(self, speeds: torch.Tensor, window: float) -> torch.Tensor:
        """Draw one reference transition per entry of ``speeds``, each within ``±window`` m/s of it.

        Without this the discriminator is shown the clips' own speed distribution (mostly 1-2 m/s)
        against whatever the policy is currently commanded, and it separates the two sides on
        ``base_lin_vel`` alone -- the style reward then simply says "run faster". Matching the
        reference speed to each policy transition removes speed as a cue, so the discriminator has
        to judge *how* the body moves at that speed. Speed tracking stays the task reward's job.

        Where the clips have no transition inside the window (below ~0.3 m/s in the running takes)
        the nearest neighbours by speed are used instead, so coverage gaps degrade gracefully
        rather than raising.
        """
        features = self.features
        ref = self._speeds
        speeds = speeds.to(ref.device)
        lo = torch.searchsorted(ref, speeds - window)
        hi = torch.searchsorted(ref, speeds + window, right=True)
        empty = hi <= lo
        if empty.any():
            pos = torch.searchsorted(ref, speeds[empty])
            lo[empty] = (pos - 1).clamp(min=0)
            hi[empty] = (pos + 1).clamp(max=ref.shape[0])
        u = torch.rand(speeds.shape[0], device=ref.device)
        idx = (lo + (u * (hi - lo).to(u.dtype)).long()).clamp(max=ref.shape[0] - 1)
        return features[idx]

    def speed_coverage(self, edges=(0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5)) -> str:
        """Share of reference transitions per speed bin -- the bins speed matching can draw from."""
        speeds = self._speeds.cpu()
        counts = torch.histogram(speeds, bins=torch.tensor(edges, dtype=speeds.dtype)).hist
        n = max(int(speeds.numel()), 1)
        return "  speed coverage: " + "  ".join(
            f"{lo:.1f}-{hi:.1f}: {int(c):5d} ({c / n * 100:4.1f}%)" for lo, hi, c in zip(edges[:-1], edges[1:], counts)
        )

    def describe(self) -> str:
        """One line per clip plus a total, for the training log."""
        gates = sorted(set(self.clip_speed_ranges))
        gate_str = f"speed gate {gates[0]} m/s" if len(gates) == 1 else f"per-clip speed gates {gates} m/s"
        lines = [f"AMP reference data: {len(self.clip_stats)} clip(s), {gate_str}"]
        for s in self.clip_stats:
            lines.append(
                f"  {s.name:<20} {s.duration_s:7.1f}s  v_mean {s.speed_mean:4.2f}  v_max {s.speed_max:4.2f}"
                f"  gate [{s.speed_range[0]:.1f}, {s.speed_range[1]:.1f}]"
                f"  kept {s.kept_pairs:6d}/{s.total_pairs:6d} ({s.kept_fraction * 100:4.1f}%)"
            )
        kept = sum(s.kept_pairs for s in self.clip_stats)
        total = sum(s.total_pairs for s in self.clip_stats)
        lines.append(f"  {'TOTAL':<20} {'':7}   {'':14}  kept {kept:6d}/{total:6d} ({kept / max(total, 1) * 100:4.1f}%)")
        if self._speeds is not None:
            lines.append(self.speed_coverage())
        return "\n".join(lines)


class AmpReplayBuffer:
    """Circular buffer of policy transitions for discriminator training (AMP paper, Sec. 5).

    Without it the discriminator only ever sees the rollout just collected, overfits to the current
    policy, and the two chase each other. Keeping the last ``capacity`` policy transitions gives it a
    history to separate from the reference set instead of a single snapshot.
    """

    def __init__(self, capacity: int, num_features: int, device: str = "cpu"):
        self.capacity = int(capacity)
        self.buffer = torch.zeros(self.capacity, num_features, device=device)
        self._pos = 0
        self._full = False

    @property
    def size(self) -> int:
        return self.capacity if self._full else self._pos

    def insert(self, transitions: torch.Tensor) -> None:
        n = transitions.shape[0]
        if n == 0:
            return
        if n >= self.capacity:
            self.buffer.copy_(transitions[-self.capacity :])
            self._pos, self._full = 0, True
            return
        end = self._pos + n
        if end <= self.capacity:
            self.buffer[self._pos : end] = transitions
        else:
            first = self.capacity - self._pos
            self.buffer[self._pos :] = transitions[:first]
            self.buffer[: n - first] = transitions[first:]
        self._pos = end % self.capacity
        self._full = self._full or end >= self.capacity

    def sample(self, batch_size: int) -> torch.Tensor:
        if self.size == 0:
            raise RuntimeError("AmpReplayBuffer is empty.")
        idx = torch.randint(0, self.size, (batch_size,), device=self.buffer.device)
        return self.buffer[idx]


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
