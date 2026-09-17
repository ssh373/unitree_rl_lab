#!/usr/bin/env python3
"""Compare telemetry CSVs recorded at different injected latencies.

    python scripts/sim2sim/compare_latency.py /tmp/latency_sweep/run_*ms.csv \
        --deploy-yaml <log_dir>/params/deploy.yaml

Prints how each vibration indicator moves as latency grows. A policy that is safe to
deploy keeps its high-frequency torque roughly flat; one that will shake on hardware
shows hf_rms climbing steeply with delay.
"""
import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_torque import (band_stats, load_csv, policy_order_labels,  # noqa: E402
                            resample_uniform, DEFAULT_XML)


def delay_of(path):
    m = re.search(r"(\d+)ms", os.path.basename(path))
    return int(m.group(1)) if m else -1


def summarize(path, args, labels, lims):
    t, qdes, q, dq, tau = load_csv(path)
    keep = (t >= t[0] + args.skip) & (t <= t[0] + args.skip + args.duration)
    if keep.sum() < 2000:
        return None  # run ended early -> the robot fell
    t, dt, (qdes, q, dq, tau) = resample_uniform(t[keep], qdes[keep], q[keep], dq[keep], tau[keep])
    fs = 1.0 / dt
    n = tau.shape[1]

    hf_rms = np.zeros(n)
    for j in range(n):
        _, hf_rms[j], _ = band_stats(tau[:, j], fs, args.hf_cut)
    sat = np.array([np.mean(np.abs(tau[:, j]) > 0.9 * lims[j]) * 100 if not np.isnan(lims[j])
                    else 0.0 for j in range(n)])
    stride = max(1, int(round(args.policy_dt / dt)))
    act_rate = np.abs(np.diff(qdes[::stride], axis=0))
    return dict(
        duration=t[-1] - t[0],
        hf_total=float(np.sqrt((hf_rms ** 2).sum())),
        hf_legs=float(np.sqrt((hf_rms[[labels.index(l) for l in labels
                                       if "hip" in l or "knee" in l or "ankle" in l]] ** 2).sum())),
        hf_worst=labels[int(np.argmax(hf_rms))],
        hf_worst_val=float(hf_rms.max()),
        sat_max=float(sat.max()),
        sat_joint=labels[int(np.argmax(sat))],
        tau_rms=float(np.sqrt((tau ** 2).mean())),
        track_p95=float(np.percentile(np.abs(qdes - q), 95)),
        act_p95=float(np.percentile(act_rate, 95)),
        gyro_proxy=float(np.sqrt((np.diff(dq, axis=0) ** 2).mean())),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--deploy-yaml", default=None)
    ap.add_argument("--mjcf", default=DEFAULT_XML)
    ap.add_argument("--hf-cut", type=float, default=10.0)
    ap.add_argument("--policy-dt", type=float, default=0.02)
    ap.add_argument("--skip", type=float, default=3.0)
    ap.add_argument("--duration", type=float, default=15.0,
                    help="analyse the same number of seconds in every run")
    args = ap.parse_args()

    paths = sorted(args.csvs, key=delay_of)
    labels, lims = policy_order_labels(args.deploy_yaml, args.mjcf)
    if labels is None:
        sys.exit("need --deploy-yaml and the MuJoCo xml to map joint names")

    hdr = (f"{'delay':>7}{'survived':>10}{'hf_total':>10}{'hf_legs':>9}{'worst joint':>24}"
           f"{'hf':>7}{'sat%':>7}{'tau_rms':>9}{'trk_p95':>9}{'act_p95':>9}")
    print(f"\nsame {args.duration:g}s window in every run, {args.hf_cut:g} Hz cut\n")
    print(hdr)
    print("-" * len(hdr))
    base = None
    for p in paths:
        r = summarize(p, args, labels, lims)
        d = delay_of(p)
        if r is None:
            print(f"{d:5d}ms{'FELL':>10}   (run ended early - policy lost balance)")
            continue
        if base is None:
            base = r
        print(f"{d:5d}ms{r['duration']:9.1f}s{r['hf_total']:10.2f}{r['hf_legs']:9.2f}"
              f"{r['hf_worst']:>24}{r['hf_worst_val']:7.2f}{r['sat_max']:7.1f}"
              f"{r['tau_rms']:9.1f}{r['track_p95']:9.3f}{r['act_p95']:9.3f}")

    if base:
        print("\nrelative to 0 ms:")
        for p in paths:
            r = summarize(p, args, labels, lims)
            if r is None:
                continue
            print(f"  {delay_of(p):3d}ms : hf_total x{r['hf_total']/base['hf_total']:.2f}   "
                  f"sat x{(r['sat_max']+1e-9)/(base['sat_max']+1e-9):.2f}   "
                  f"act_rate x{r['act_p95']/base['act_p95']:.2f}")
        print("\nreading: hf_total x1.0-1.3 -> latency-robust; x2+ -> expect shaking on hardware;")
        print("         FELL at a delay the real robot has -> do not deploy this policy.")


if __name__ == "__main__":
    sys.exit(main())
