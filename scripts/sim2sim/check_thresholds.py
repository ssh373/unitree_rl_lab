#!/usr/bin/env python3
"""Check recorded torques against the per-joint sim2real deployment thresholds.

    python scripts/sim2sim/check_thresholds.py run.csv --deploy-yaml <log>/params/deploy.yaml

Reports, per joint: peak delivered torque, how often the threshold is exceeded, how long
each excursion lasts, and the torque the PD law actually *asked* for. The requested value
matters because MuJoCo clips at its own actuatorfrcrange: a joint sitting exactly on the
MuJoCo rail hides how much more the policy wanted.
"""
import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_torque import (DEFAULT_XML, load_csv, policy_order_labels,  # noqa: E402
                            resample_uniform)

# sim2real per-joint torque thresholds [Nm]
DEFAULT_THRESHOLDS = {
    r".*_hip_pitch_joint": 80.0,
    r".*_hip_yaw_joint": 80.0,
    r".*_hip_roll_joint": 120.0,
    r".*_knee_joint": 120.0,
    r".*_ankle_.*_joint": 50.0,
}


def threshold_for(name, table):
    for pat, v in table.items():
        if re.fullmatch(pat, name):
            return v
    return None


def episodes(mask, dt):
    """Durations [s] of each contiguous True run."""
    if not mask.any():
        return np.array([])
    d = np.diff(mask.astype(np.int8))
    starts = np.flatnonzero(d == 1) + 1
    ends = np.flatnonzero(d == -1) + 1
    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, len(mask)]
    n = min(len(starts), len(ends))
    return (ends[:n] - starts[:n]) * dt


def gains(deploy_yaml, ids):
    """kp/kd from deploy.yaml (stored in SDK order, see articulation.h) -> policy order."""
    y = open(deploy_yaml).read()
    num = lambda k: np.array([float(v) for v in re.search(
        rf"{k}:\s*\[([^\]]*)\]", y, re.S).group(1).replace("\n", " ").split(",")])
    return num("stiffness")[ids], num("damping")[ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--deploy-yaml", required=True)
    ap.add_argument("--mjcf", default=DEFAULT_XML)
    ap.add_argument("--skip", type=float, default=3.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--all-joints", action="store_true",
                    help="also list joints that have no configured threshold")
    args = ap.parse_args()

    labels, mj_lims = policy_order_labels(args.deploy_yaml, args.mjcf)
    if labels is None:
        sys.exit("could not map joint names; check --deploy-yaml / --mjcf")
    ids = [int(v) for v in re.search(r"joint_ids_map:\s*\[([^\]]*)\]",
                                     open(args.deploy_yaml).read(), re.S
                                     ).group(1).replace("\n", " ").split(",")]
    kp, kd = gains(args.deploy_yaml, ids)

    t, qdes, q, dq, tau = load_csv(args.csv)
    keep = t >= t[0] + args.skip
    if args.end is not None:
        keep &= t <= args.end
    t, dt, (qdes, q, dq, tau) = resample_uniform(t[keep], qdes[keep], q[keep], dq[keep], tau[keep])
    dur = t[-1] - t[0]

    print(f"\n{os.path.basename(args.csv)}   window {t[0]:.1f}-{t[-1]:.1f}s ({dur:.1f}s)\n")
    hdr = (f"{'joint':<24}{'thr':>6}{'peak':>8}{'over%':>7}{'n/s':>6}"
           f"{'longest':>9}{'req_peak':>10}{'mj_lim':>8}  verdict")
    print(hdr)
    print("-" * len(hdr))

    worst = []
    for j, name in enumerate(labels):
        thr = threshold_for(name, DEFAULT_THRESHOLDS)
        if thr is None and not args.all_joints:
            continue
        a = np.abs(tau[:, j])
        req = np.abs(kp[j] * (qdes[:, j] - q[:, j]) - kd[j] * dq[:, j])
        if thr is None:
            print(f"{name:<24}{'-':>6}{a.max():8.1f}{'-':>7}{'-':>6}{'-':>9}"
                  f"{req.max():10.1f}{mj_lims[j]:8.0f}")
            continue
        over = a > thr
        eps = episodes(over, dt)
        pct = 100.0 * over.mean()
        verdict = "OK" if not over.any() else ("MARGINAL" if pct < 1.0 else "EXCEEDS")
        if over.any():
            worst.append((pct, name, a.max(), thr))
        print(f"{name:<24}{thr:6.0f}{a.max():8.1f}{pct:7.2f}{len(eps)/dur:6.1f}"
              f"{1000*eps.max() if len(eps) else 0:8.0f}ms{req.max():10.1f}{mj_lims[j]:8.0f}"
              f"  {verdict}")

    print("\nover%   = fraction of time above the threshold")
    print("n/s     = threshold crossings per second (gait is ~2/s per leg)")
    print("req_peak= torque the PD law asked for, before MuJoCo's own clip (mj_lim)")
    if not worst:
        print("\nNo joint exceeded its sim2real threshold in this window.")
    else:
        print()
        for pct, name, pk, thr in sorted(worst, reverse=True):
            print(f"  {name}: peak {pk:.0f} Nm vs threshold {thr:.0f} Nm ({pct:.2f}% of the time)")


if __name__ == "__main__":
    sys.exit(main())
