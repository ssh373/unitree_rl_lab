#!/usr/bin/env python3
"""Analyze a sim2sim / sim2real telemetry CSV produced by TelemetryLogger.

    RL_LOG_CSV=/tmp/run.csv ./g1_ctrl --network lo      # record
    python scripts/sim2sim/analyze_torque.py /tmp/run.csv --deploy-yaml <log_dir>/params/deploy.yaml

Columns: t, qdes_XX, q_XX, dq_XX, tau_XX  (XX = policy joint order).

Prints a per-joint table of vibration indicators and writes plots next to the CSV.
"""
import argparse
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np

DEFAULT_XML = os.path.expanduser("~/unitree_mujoco/unitree_robots/g1/g1_29dof.xml")


def load_csv(path):
    data = np.genfromtxt(path, delimiter=",", names=True)
    names = list(data.dtype.names)
    t = data["t"].astype(np.float64)
    n = sum(1 for c in names if c.startswith("qdes_"))
    stack = lambda p: np.column_stack([data[f"{p}_{i:02d}"] for i in range(n)]).astype(np.float64)
    return t, stack("qdes"), stack("q"), stack("dq"), stack("tau")


def joint_names_and_limits(xml_path):
    """Return (names, |tau|max) in MuJoCo/SDK joint order."""
    if not os.path.exists(xml_path):
        return None, None
    root = ET.parse(xml_path).getroot()
    names, lims = [], []
    for j in root.iter("joint"):
        rng = j.get("actuatorfrcrange")
        if rng is None:
            continue  # floating base
        names.append(j.get("name", f"joint_{len(names)}"))
        lims.append(abs(float(rng.split()[1])))
    return names, np.array(lims)


def policy_order_labels(deploy_yaml, xml_path):
    names, lims = joint_names_and_limits(xml_path)
    ids = None
    if deploy_yaml and os.path.exists(deploy_yaml):
        txt = open(deploy_yaml).read()
        m = re.search(r"joint_ids_map:\s*\[([^\]]*)\]", txt, re.S)
        if m:
            ids = [int(v) for v in m.group(1).replace("\n", " ").split(",")]
    if names is None or ids is None:
        return None, None
    return [names[i] for i in ids], np.array([lims[i] for i in ids])


def auto_window(t, tau, bucket=2.0):
    """Pick the longest stretch where the robot is actually doing something.

    A record usually starts with several seconds of standing; analysing that tells you
    nothing about gait vibration, so find where the torque activity ramps up."""
    act = np.abs(tau).mean(axis=1)
    edges = np.arange(t[0], t[-1], bucket)
    lvl = np.array([act[(t >= e) & (t < e + bucket)].mean() if ((t >= e) & (t < e + bucket)).any()
                    else 0.0 for e in edges])
    thr = 0.5 * (lvl.max() + np.median(lvl))
    active = lvl >= thr
    if not active.any():
        return t[0], t[-1]
    idx = np.flatnonzero(active)
    # longest contiguous run
    splits = np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1)
    run = max(splits, key=len)
    return float(edges[run[0]]), float(min(edges[run[-1]] + bucket, t[-1]))


def resample_uniform(t, *arrays):
    """The 1 kHz FSM thread jitters a little; resample onto a uniform grid for the FFT."""
    dt = float(np.median(np.diff(t)))
    tt = np.arange(t[0], t[-1], dt)
    out = [np.column_stack([np.interp(tt, t, a[:, j]) for j in range(a.shape[1])]) for a in arrays]
    return tt, dt, out


def welch_psd(x, fs, nseg=4096):
    """Averaged periodogram. Single-shot FFTs of a long record leak badly and produce a
    dense comb of spurious lines; averaging short segments gives a readable noise floor."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    nseg = min(nseg, len(x))
    step = nseg // 2
    win = np.hanning(nseg)
    norm = (win ** 2).sum() * fs
    segs = [x[i:i + nseg] for i in range(0, len(x) - nseg + 1, step)]
    if not segs:
        segs = [np.pad(x, (0, nseg - len(x)))]
    acc = np.zeros(nseg // 2 + 1)
    for seg in segs:
        acc += np.abs(np.fft.rfft(seg * win)) ** 2
    psd = acc / (len(segs) * norm)
    freq = np.fft.rfftfreq(nseg, d=1.0 / fs)
    return freq, psd


def band_stats(x, fs, f_lo):
    """(HF power fraction, HF RMS in Nm, dominant HF frequency)."""
    freq, psd = welch_psd(x, fs)
    df = freq[1] - freq[0]
    psd[0] = 0.0
    total = psd.sum() * df
    if total <= 0:
        return 0.0, 0.0, 0.0
    hi = freq > f_lo
    hf_power = psd[hi].sum() * df
    f_peak = float(freq[hi][np.argmax(psd[hi])]) if hi.any() and psd[hi].max() > 0 else 0.0
    return float(hf_power / total), float(np.sqrt(max(hf_power, 0.0))), f_peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--deploy-yaml", default=None, help="params/deploy.yaml of the policy (for joint names)")
    ap.add_argument("--mjcf", default=DEFAULT_XML)
    ap.add_argument("--policy-dt", type=float, default=0.02)
    ap.add_argument("--hf-cut", type=float, default=10.0, help="Hz above which torque content counts as vibration")
    ap.add_argument("--skip", type=float, default=1.0, help="seconds to drop at the start (startup transient)")
    ap.add_argument("--start", type=float, default=None, help="analysis window start [s] (default: auto-detect the active segment)")
    ap.add_argument("--end", type=float, default=None, help="analysis window end [s]")
    ap.add_argument("--plot-window", type=float, default=3.0, help="seconds of time series to draw")
    ap.add_argument("--min-tau", type=float, default=1.0,
                    help="ignore joints whose torque RMS is below this [Nm]; HF%% is a ratio and "
                         "explodes on joints that carry almost no load")
    ap.add_argument("--top", type=int, default=8, help="how many worst joints to plot")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    t, qdes, q, dq, tau = load_csv(args.csv)

    if args.start is None and args.end is None:
        lo, hi = auto_window(t, tau)
        print(f"auto-detected active segment: {lo:.1f}-{hi:.1f}s "
              f"(override with --start/--end)")
    else:
        lo = args.start if args.start is not None else t[0] + args.skip
        hi = args.end if args.end is not None else t[-1]
    keep = (t >= max(lo, t[0] + args.skip)) & (t <= hi)
    if keep.sum() < 1000:
        sys.exit(f"window {lo:.1f}-{hi:.1f}s has too few samples")
    t, qdes, q, dq, tau = t[keep], qdes[keep], q[keep], dq[keep], tau[keep]

    t, dt, (qdes, q, dq, tau) = resample_uniform(t, qdes, q, dq, tau)
    fs = 1.0 / dt
    n = tau.shape[1]

    labels, lims = policy_order_labels(args.deploy_yaml, args.mjcf)
    if labels is None:
        labels = [f"j{i:02d}" for i in range(n)]
        lims = np.full(n, np.nan)

    dtau = np.diff(tau, axis=0) / dt
    rows = []
    for j in range(n):
        ac = tau[:, j] - tau[:, j].mean()
        hf, hf_rms, fpeak = band_stats(tau[:, j], fs, args.hf_cut)
        # sign flips of the AC part per second == chatter rate
        flips = np.count_nonzero(np.diff(np.sign(ac)) != 0) / (t[-1] - t[0])
        sat = np.nan if np.isnan(lims[j]) else 100.0 * np.mean(np.abs(tau[:, j]) > 0.9 * lims[j])
        rows.append(dict(
            joint=labels[j],
            tau_rms=float(np.sqrt(np.mean(tau[:, j] ** 2))),
            tau_peak=float(np.max(np.abs(tau[:, j]))),
            sat_pct=sat,
            hf_frac=100.0 * hf,
            hf_rms=hf_rms,
            f_peak=fpeak,
            flips_hz=flips,
            dtau_p95=float(np.percentile(np.abs(dtau[:, j]), 95)),
            track_rms=float(np.sqrt(np.mean((qdes[:, j] - q[:, j]) ** 2))),
        ))

    # action rate at the policy rate (qdes is ZOH between policy steps)
    stride = max(1, int(round(args.policy_dt / dt)))
    dq_des = np.abs(np.diff(qdes[::stride], axis=0))

    # Rank by ABSOLUTE high-frequency torque, not by the ratio: a wrist carrying 0.05 Nm
    # scores a huge HF%% while being physically irrelevant.
    ranked = [r for r in rows if r["tau_rms"] >= args.min_tau]
    muted = len(rows) - len(ranked)
    ranked.sort(key=lambda r: -r["hf_rms"])

    hdr = (f"{'joint':<26}{'tau_rms':>9}{'tau_pk':>8}{'sat%':>7}"
           f"{'hf_rms':>8}{'HF%':>7}{'fpk_hf':>8}{'flip/s':>8}{'dtau_p95':>10}{'trk_rms':>9}")
    print(f"\nwindow {t[0]:.1f}-{t[-1]:.1f}s ({t[-1]-t[0]:.1f}s)   fs {fs:.0f} Hz   "
          f"HF cut {args.hf_cut:g} Hz   sorted by hf_rms [Nm]")
    print(hdr)
    print("-" * len(hdr))
    for r in ranked:
        sat = "  n/a" if np.isnan(r["sat_pct"]) else f"{r['sat_pct']:6.1f}"
        print(f"{r['joint']:<26}{r['tau_rms']:9.1f}{r['tau_peak']:8.1f}{sat:>7}"
              f"{r['hf_rms']:8.2f}{r['hf_frac']:7.1f}{r['f_peak']:8.1f}{r['flips_hz']:8.1f}"
              f"{r['dtau_p95']:10.0f}{r['track_rms']:9.3f}")
    if muted:
        print(f"({muted} joints below --min-tau {args.min_tau:g} Nm hidden)")

    print(f"\naction rate |dq_des| @ {1/args.policy_dt:.0f} Hz : "
          f"mean {dq_des.mean():.4f} rad  p95 {np.percentile(dq_des, 95):.4f} rad  max {dq_des.max():.4f} rad")
    nyq = 0.5 / args.policy_dt
    near_nyq = [r["joint"] for r in ranked if r["hf_rms"] > 1.0 and r["f_peak"] > 0.6 * nyq]
    sat_bad = [f"{r['joint']} ({r['sat_pct']:.0f}%)" for r in ranked
               if not np.isnan(r["sat_pct"]) and r["sat_pct"] > 5.0]
    if sat_bad:
        print(f"WARNING: torque saturation > 5% of the time on: {', '.join(sat_bad)}")
    if near_nyq:
        print(f"WARNING: torque peaks near the policy Nyquist ({nyq:.0f} Hz) on: {', '.join(near_nyq)}")

    if args.no_plot:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    worst = [labels.index(r["joint"]) for r in ranked[:args.top]]
    fig, axes = plt.subplots(len(worst), 3, figsize=(16, 2.2 * len(worst)), squeeze=False)
    win = t <= t[0] + args.plot_window
    for k, j in enumerate(worst):
        ax = axes[k][0]
        ax.plot(t[win], tau[win, j], lw=0.7)
        if not np.isnan(lims[j]):
            ax.axhline(lims[j], color="r", ls=":", lw=0.8)
            ax.axhline(-lims[j], color="r", ls=":", lw=0.8)
        ax.set_ylabel(labels[j], fontsize=7)
        if k == 0:
            ax.set_title("torque [Nm] (3 s)")

        ax = axes[k][1]
        ax.plot(t[win], qdes[win, j], lw=0.7, label="q_des")
        ax.plot(t[win], q[win, j], lw=0.7, label="q")
        if k == 0:
            ax.set_title("q_des vs q [rad]")
            ax.legend(fontsize=6)

        ax = axes[k][2]
        freq, psd = welch_psd(tau[:, j], fs)
        ax.semilogy(freq[1:], psd[1:] + 1e-12, lw=0.8)
        ax.axvline(nyq, color="r", ls="--", lw=0.8)
        ax.axvline(args.hf_cut, color="g", ls=":", lw=0.8)
        ax.set_xlim(0, min(150, fs / 2))
        if k == 0:
            ax.set_title("torque PSD (red = policy Nyquist)")
    axes[-1][0].set_xlabel("t [s]")
    axes[-1][1].set_xlabel("t [s]")
    axes[-1][2].set_xlabel("f [Hz]")
    fig.tight_layout()
    out = os.path.splitext(args.csv)[0] + "_torque.png"
    fig.savefig(out, dpi=130)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    sys.exit(main())
