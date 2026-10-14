"""Touchdown force traces: first-peak timing, settling into support, and impulse-window sensitivity.

Reads the .npz files written by ``trace_impact.py`` for two policies and writes one figure plus a CSV
summary. All numbers are per touchdown (both feet pooled), envs that terminated inside the window
already excluded by the recorder.

Definitions (t = 0 is the first 5 ms substep with |F| > 1 N after >= 40 ms without contact):
    first peak      first local maximum of |F| after touchdown that exceeds 0.5 BW
    support window  60-120 ms after touchdown (mid-stance for a ~160-190 ms stance); its mean and the
                    within-touchdown std of |F| say whether the force has settled
    J(w)            int |F| dt over the first w ms after touchdown, w in {20, 40, 60, 80}
    stance impulse  int |F| dt until |F| first drops below the contact threshold after the first peak

Example:
    python scripts/rsl_rl/plot_impact_trace.py --speed 2.0 \
        --inputs baseline=logs/impact_trace_baseline_29999.npz "Energy reward=logs/impact_trace_PACEedit_29999.npz" \
        --png logs/report_impact_trace_2ms_1012.png --csv logs/impact_trace_summary_2ms_1012.csv
"""

import argparse
import csv
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

_PREF = ["Noto Sans CJK KR", "NanumGothic", "NanumBarunGothic", "Malgun Gothic", "AppleGothic"]
_avail = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in _PREF if f in _avail), None)
if FONT is None:
    for f in font_manager.findSystemFonts():
        try:
            n = font_manager.FontProperties(fname=f).get_name()
        except Exception:
            continue
        if any(k in n for k in ("Nanum", "Noto Sans CJK")):
            font_manager.fontManager.addfont(f); FONT = n; break
plt.rcParams["font.family"] = [FONT or "DejaVu Sans", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# categorical slots 1 and 2 of the reference palette (validated pair), text in ink tokens
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, MUTED, GRID = "#1a1a19", "#5f5e58", "#8a8982", "#e4e3dd"
WINDOWS_MS = [20, 40, 60, 80]


def first_peak(f, bw, min_bw=0.5):
    """Index of the first local maximum above min_bw*BW, per row; -1 if none."""
    n = f.shape[1]
    up = np.zeros_like(f, dtype=bool)
    up[:, 1:-1] = (f[:, 1:-1] >= f[:, :-2]) & (f[:, 1:-1] > f[:, 2:])
    up[:, 0] = f[:, 0] > f[:, 1]
    up &= f > min_bw * bw
    has = up.any(axis=1)
    idx = np.where(has, up.argmax(axis=1), -1)
    return idx


def analyse(d, speed):
    key = f"s{speed:g}"
    t = d["t_ms"]
    bw = float(d["body_weight_n"])
    dt = float(d["sim_dt"])
    thr = float(d["contact_force_n"])
    fz, fn = d[f"{key}_fz"], d[f"{key}_fn"]
    i0 = int(np.argmin(np.abs(t)))
    post = fn[:, i0:]
    post_z = fz[:, i0:]
    tp = t[i0:]

    pk = first_peak(post, bw)
    ok = pk >= 0
    pk_t = tp[pk[ok]]
    pk_f = post[ok, pk[ok]] / bw
    gmax = post[:, : int(round(100 / (dt * 1e3)))].argmax(axis=1)

    # support window 60-120 ms
    sw = (tp >= 60) & (tp < 120)
    sup_mean = post[:, sw].mean(axis=1) / bw
    sup_std = post[:, sw].std(axis=1) / bw

    # stance end: first substep after the first peak where |F| < threshold
    stance_end = np.full(len(post), post.shape[1])
    for k in np.flatnonzero(ok):
        below = np.flatnonzero(post[k, pk[k]:] < thr)
        if len(below):
            stance_end[k] = pk[k] + below[0]
    cum = np.cumsum(post, axis=1) * dt
    J = {w: cum[:, int(round(w / (dt * 1e3))) - 1] for w in WINDOWS_MS}
    J_stance = np.array([cum[k, min(stance_end[k], post.shape[1]) - 1] for k in range(len(post))])
    Jz = {w: (np.cumsum(np.clip(post_z, 0, None), axis=1) * dt)[:, int(round(w / (dt * 1e3))) - 1] for w in WINDOWS_MS}

    return {
        "n": len(post),
        "t": t, "fz": fz, "fn": fn, "bw": bw, "i0": i0,
        "peak_t_ms": pk_t, "peak_bw": pk_f, "peak_found": ok.mean(),
        "gmax_t_ms": tp[gmax], "gmax_bw": post[np.arange(len(post)), gmax] / bw,
        "sup_mean_bw": sup_mean, "sup_std_bw": sup_std,
        "stance_ms": stance_end * dt * 1e3,
        "J": J, "Jz": Jz, "J_stance": J_stance,
    }


def q(x, p):
    return float(np.percentile(x, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", type=float, default=2.0)
    ap.add_argument("--inputs", nargs="+", required=True, help="label=path.npz ...")
    ap.add_argument("--png", required=True)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--n_traces", type=int, default=60, help="Individual traces drawn per panel.")
    args = ap.parse_args()

    runs = []
    for item in args.inputs:
        label, path = item.split("=", 1)
        runs.append((label, analyse(np.load(path), args.speed)))

    # ---- summary table ----
    rows = []
    for label, r in runs:
        row = {
            "policy": label, "cmd_mps": args.speed, "touchdowns": r["n"],
            "first_peak_ms_median": q(r["peak_t_ms"], 50), "first_peak_ms_p25": q(r["peak_t_ms"], 25),
            "first_peak_ms_p75": q(r["peak_t_ms"], 75), "first_peak_bw_median": q(r["peak_bw"], 50),
            "first_peak_found": r["peak_found"],
            "max100_ms_median": q(r["gmax_t_ms"], 50), "max100_bw_median": q(r["gmax_bw"], 50),
            "support_mean_bw": float(r["sup_mean_bw"].mean()), "support_within_std_bw": float(r["sup_std_bw"].mean()),
            "stance_ms_median": q(r["stance_ms"], 50),
            "J_stance_ns": float(r["J_stance"].mean()),
        }
        for w in WINDOWS_MS:
            row[f"J{w}_ns"] = float(r["J"][w].mean())
            row[f"J{w}_sd"] = float(r["J"][w].std())
            row[f"J{w}_share_of_stance"] = float((r["J"][w] / r["J_stance"]).mean())
            row[f"Jz{w}_ns"] = float(r["Jz"][w].mean())
        rows.append(row)
    for row in rows:
        print(f"== {row['policy']} @ {row['cmd_mps']} m/s, n={row['touchdowns']}")
        print(f"   first peak: median {row['first_peak_ms_median']:.0f} ms (IQR {row['first_peak_ms_p25']:.0f}-{row['first_peak_ms_p75']:.0f}), "
              f"{row['first_peak_bw_median']:.2f} BW; found in {100*row['first_peak_found']:.0f}% | max in 100 ms at {row['max100_ms_median']:.0f} ms, {row['max100_bw_median']:.2f} BW")
        print(f"   support 60-120 ms: mean {row['support_mean_bw']:.2f} BW, within-touchdown std {row['support_within_std_bw']:.2f} BW | stance {row['stance_ms_median']:.0f} ms, J_stance {row['J_stance_ns']:.1f} Ns")
        print("   " + " | ".join(f"J{w} {row[f'J{w}_ns']:.2f}±{row[f'J{w}_sd']:.2f} Ns ({100*row[f'J{w}_share_of_stance']:.0f}% of stance)" for w in WINDOWS_MS))
    if len(rows) == 2:
        a, b = rows
        print(f"== {b['policy']} vs {a['policy']}: " + " | ".join(f"J{w} {100*(b[f'J{w}_ns']/a[f'J{w}_ns']-1):+.0f}%" for w in WINDOWS_MS)
              + f" | J_stance {100*(b['J_stance_ns']/a['J_stance_ns']-1):+.0f}%")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            wri = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wri.writeheader(); wri.writerows(rows)
        print(f"[INFO] wrote {args.csv}")

    # ---- figure ----
    ncol = max(len(runs), 2)
    fig = plt.figure(figsize=(6.5 * ncol, 9), facecolor="white")
    gs = fig.add_gridspec(2, ncol, height_ratios=[1.15, 1], hspace=0.38, wspace=0.22)
    axes_top = [fig.add_subplot(gs[0, i]) for i in range(len(runs))]
    ax_mean = fig.add_subplot(gs[1, : ncol // 2])
    ax_J = fig.add_subplot(gs[1, ncol // 2 :])
    rng = np.random.default_rng(0)
    ymax = max(np.percentile(r["fz"].max(axis=1) / r["bw"], 98) for _, r in runs) * 1.08

    for ax, (label, r), c in zip(axes_top, runs, COLORS):
        t, fzb = r["t"], r["fz"] / r["bw"]
        pick = rng.choice(len(fzb), size=min(args.n_traces, len(fzb)), replace=False)
        for k in pick:
            ax.plot(t, fzb[k], color=c, lw=0.6, alpha=0.10)
        lo, hi = np.percentile(fzb, [25, 75], axis=0)
        ax.fill_between(t, lo, hi, color=c, alpha=0.18, lw=0, label="IQR")
        ax.plot(t, fzb.mean(axis=0), color=c, lw=2.2, label="평균")
        pt, pf = q(r["gmax_t_ms"], 50), q(r["gmax_bw"], 50)
        ax.plot([pt], [pf], marker="o", ms=8, color=c, mec="white", mew=1.5, zorder=5)
        ax.annotate(f"100 ms 내 최대: 중앙값 {pt:.0f} ms, {pf:.2f} BW", (pt, pf), xytext=(pt + 22, min(pf + 0.45, ymax - 0.15)),
                    fontsize=9, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
        for w in WINDOWS_MS:
            ax.axvline(w, color=MUTED, lw=0.8, ls=(0, (3, 3)))
            ax.text(w, ymax * 0.985, f"{w}", ha="center", va="top", fontsize=8, color=INK2)
        ax.axhline(1.0, color=MUTED, lw=0.8, ls=":")
        ax.text(t[-1], 1.0, " 1 BW", va="center", ha="left", fontsize=8, color=INK2, clip_on=False)
        ax.set_title(f"{label}  (n={r['n']} 터치다운)", fontsize=11, color=INK, loc="left")
        ax.set_xlim(t[0], t[-1]); ax.set_ylim(-0.05, ymax)
        ax.set_xlabel("터치다운 이후 시간 [ms]"); ax.set_ylabel("F_z / 체중 [BW]")
        ax.legend(loc="upper right", frameon=False, fontsize=9)

    # mean overlay
    for (label, r), c in zip(runs, COLORS):
        t, m = r["t"], (r["fz"] / r["bw"]).mean(axis=0)
        ax_mean.plot(t, m, color=c, lw=2.2, label=label)
    for w in WINDOWS_MS:
        ax_mean.axvline(w, color=MUTED, lw=0.8, ls=(0, (3, 3)))
    ax_mean.axhline(1.0, color=MUTED, lw=0.8, ls=":")
    ax_mean.set_xlim(runs[0][1]["t"][0], runs[0][1]["t"][-1]); ax_mean.set_ylim(-0.05, None)
    ax_mean.set_title("평균 F_z(t) 비교", fontsize=11, color=INK, loc="left")
    ax_mean.set_xlabel("터치다운 이후 시간 [ms]"); ax_mean.set_ylabel("F_z / 체중 [BW]")
    ax_mean.legend(loc="upper right", frameon=False, fontsize=9)

    # impulse vs window
    label_offsets = [-14, 9, -14, 9]
    for k, ((label, r), c) in enumerate(zip(runs, COLORS)):
        means = [r["J"][w].mean() for w in WINDOWS_MS]
        sds = [r["J"][w].std() for w in WINDOWS_MS]
        ax_J.errorbar(WINDOWS_MS, means, yerr=sds, color=c, lw=2, marker="o", ms=8, mec="white", mew=1.5,
                      capsize=3, label=label)
        for w, m in zip(WINDOWS_MS, means):
            ax_J.annotate(f"{m:.1f}", (w, m), xytext=(0, label_offsets[k]), textcoords="offset points",
                          ha="center", fontsize=8.5, color=INK)
    ax_J.set_xticks(WINDOWS_MS); ax_J.set_xlim(12, 88)
    ax_J.set_title("착지 충격량 ∫|F|dt 와 적분 창 길이 (평균 ± 표준편차)", fontsize=11, color=INK, loc="left")
    ax_J.set_xlabel("적분 창 [ms]"); ax_J.set_ylabel("충격량 [N·s]")
    ax_J.legend(loc="upper left", frameon=False, fontsize=9)

    for ax in axes_top + [ax_mean, ax_J]:
        ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9)
    fig.suptitle(f"터치다운 전후 발 접촉력 — {args.speed:g} m/s, model_29999, 평지 (t=0: |F|>1 N 첫 서브스텝, 5 ms 샘플)",
                 fontsize=12, color=INK, x=0.02, ha="left")
    os.makedirs(os.path.dirname(os.path.abspath(args.png)), exist_ok=True)
    fig.savefig(args.png, dpi=150, bbox_inches="tight")
    print(f"[INFO] wrote {args.png}")


if __name__ == "__main__":
    main()
