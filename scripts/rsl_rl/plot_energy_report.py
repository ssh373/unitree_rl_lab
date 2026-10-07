"""Report figures for the G1 running energy-efficiency comparison (N policies, 5 speeds).

Reads logs/energy_new_<stem>_29999.csv (+ _groups.csv, from ``eval_energy.py --per_group``) for every
policy in POLICIES and writes
    <out>/report_energy_raw_<tag>.png          raw power [W] and CoT, mechanical / Joule / electrical total
    <out>/report_cot_reduction_<tag>.png       saving vs baseline for the three CoT definitions
    <out>/report_group_breakdown_2ms_<tag>.png per joint-group Joule heat and mechanical work at 2.0 m/s

    python scripts/rsl_rl/plot_energy_report.py --csv_dir logs --out logs --tag 1007
"""
import argparse
import csv
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

# ---- Korean-capable font -------------------------------------------------------------------------
_PREF = ["Noto Sans CJK KR", "Noto Sans CJK JP", "Noto Sans CJK SC", "NanumGothic", "NanumBarunGothic",
         "NanumSquareRound", "Malgun Gothic", "AppleGothic"]
_avail = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in _PREF if f in _avail), None)
if FONT is None:
    for f in font_manager.findSystemFonts():
        try:
            n = font_manager.FontProperties(fname=f).get_name()
        except Exception:
            continue
        if any(k in n for k in ("Nanum", "Noto Sans CJK", "Noto Serif CJK")):
            font_manager.fontManager.addfont(f); FONT = n; break
plt.rcParams["font.family"] = [FONT or "DejaVu Sans", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ---- palette (data-viz reference palette) --------------------------------------------------------
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
# (csv stem, report label, colour, marker). The first entry is the baseline every saving is measured against.
POLICIES = [
    ("baseline",  "baseline",                 "#898781", "o"),
    ("PACEedit",  "Energy reward",            "#eb6834", "s"),
    ("AMPedit",   "AMP",                      "#1baf7a", "^"),
    ("AMPpace",   "AMP+Energy reward",        "#2a78d6", "D"),
    ("AMPlowcot", "AMP (low-CoT motions)",    "#eda100", "v"),
]
GROUP_COL = {"hip": "#2a78d6", "knee": "#eb6834", "ankle": "#1baf7a", "waist": "#52514e", "arm": "#b5b4ad"}


def load(csv_dir, stem):
    """Summary rows (one per speed) with the per-group Joule/mech watts attached; None if the CSV is missing."""
    path = os.path.join(csv_dir, f"energy_new_{stem}_29999.csv")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    gpath = os.path.join(csv_dir, f"energy_new_{stem}_29999_groups.csv")
    groups = []
    if os.path.isfile(gpath):
        with open(gpath) as f:
            groups = [{k: (v if k == "group" else float(v)) for k, v in r.items()} for r in csv.DictReader(f)]
    for r in rows:
        g = [x for x in groups if x["cmd_mps"] == r["cmd_mps"]]
        r["groups"] = {x["group"]: {"pos_w": x["pos_power_w"], "joule_w": x["joule_w"]} for x in g}
        # Joule watts: exact group sum when available, else recovered from the CoT columns (<0.4% error).
        r["joule_w"] = sum(x["joule_w"] for x in g) if g else r["pos_power_w"] * (r["cot_elec"] - r["cot_pos"]) / r["cot_pos"]
    return rows


def style(ax, title, ylabel, speeds):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=8.5, length=0)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8, fontweight="bold")
    ax.set_ylabel(ylabel, color=INK2, fontsize=8.5)
    ax.set_xticks(speeds)
    ax.set_xticklabels([f"{s:.1f}" for s in speeds])
    ax.set_xlim(0.3, 2.7)


def repel_labels(ax, pts, fmt, fontsize=7.5, gap=11.5):
    """pts: list of (x, y, colour). Labels start 6 px above their point; per x they are pushed apart
    vertically (display px) so none overlap; a label pushed far from its point gets a thin leader."""
    fig = ax.figure
    fig.canvas.draw()
    tr, inv = ax.transData, ax.transData.inverted()
    by_x = {}
    for x, y, c in pts:
        by_x.setdefault(round(x, 3), []).append([x, y, c])
    for _, grp in by_x.items():
        grp.sort(key=lambda p: p[1])
        disp = [tr.transform((p[0], p[1]))[1] + 6 for p in grp]
        for i in range(1, len(disp)):
            if disp[i] - disp[i - 1] < gap:
                disp[i] = disp[i - 1] + gap
        shift = (np.mean(disp) - np.mean([tr.transform((p[0], p[1]))[1] + 6 for p in grp]))
        disp = [d - shift for d in disp]
        for p, d in zip(grp, disp):
            xd, _ = tr.transform((p[0], p[1]))
            ly = inv.transform((xd, d))[1]
            far = abs(d - 6 - tr.transform((p[0], p[1]))[1]) > 9
            ax.annotate(fmt(p[1]), (p[0], p[1]), xytext=(p[0], ly), textcoords="data", ha="center",
                        va="bottom", fontsize=fontsize, color=INK2,
                        arrowprops=dict(arrowstyle="-", color=GRID, lw=0.6) if far else None,
                        bbox=dict(boxstyle="round,pad=0.12", fc=SURFACE, ec="none", alpha=0.85))


class Report:
    def __init__(self, csv_dir):
        self.policies = []
        self.data = {}
        for stem, label, c, m in POLICIES:
            rows = load(csv_dir, stem)
            if rows is None:
                print(f"skip {label}: no energy_new_{stem}_29999.csv in {csv_dir}")
                continue
            self.policies.append((stem, label, c, m))
            self.data[stem] = rows
        self.base = self.policies[0][0]
        self.speeds = [r["cmd_mps"] for r in self.data[self.base]]

    def series(self, stem, col):
        rows = self.data[stem]
        if col == "cot_heat":
            return np.array([r["cot_elec"] - r["cot_pos"] for r in rows])
        if col == "elec_w":
            return np.array([r["pos_power_w"] + r["joule_w"] for r in rows])
        return np.array([r[col] for r in rows])

    def fig_raw(self, out):
        panels = [
            ("pos_power_w", r"기계적 일  $\sum \max(\tau\,\dot{q},\,0)$", "파워 [W]", lambda v: f"{v:.0f}"),
            ("joule_w",     r"권선 Joule 열  $\sum k_{el}\,\tau^2$", "파워 [W]", lambda v: f"{v:.0f}"),
            ("elec_w",      "전기적 합  (기계적 일 + Joule 열)", "파워 [W]", lambda v: f"{v:.0f}"),
            ("cot_pos",     "기계적 CoT", "CoT [-]", lambda v: f"{v:.2f}"),
            ("cot_heat",    "Joule 열 CoT", "CoT [-]", lambda v: f"{v:.2f}"),
            ("cot_elec",    "전기적 CoT  (기계적 + Joule)", "CoT [-]", lambda v: f"{v:.2f}"),
        ]
        fig, axes = plt.subplots(2, 3, figsize=(15, 8.8), facecolor=SURFACE)
        for ax, (col, title, yl, fmt) in zip(axes.flat, panels):
            style(ax, title, yl, self.speeds)
            pts, ymax = [], 0
            for stem, label, c, m in self.policies:
                y = self.series(stem, col)
                ymax = max(ymax, y.max())
                ax.plot(self.speeds, y, color=c, lw=2, marker=m, ms=6, mec=SURFACE, mew=1.2, label=label, zorder=3)
                pts += [(x, v, c) for x, v in zip(self.speeds, y)]
            ax.set_ylim(0, ymax * 1.18)
            repel_labels(ax, pts, fmt)
        for ax in axes[1]:
            ax.set_xlabel("명령 전진 속도 [m/s]", color=INK2, fontsize=9)
        h, l = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper right", ncol=len(self.policies), frameon=False, fontsize=9.5,
                   bbox_to_anchor=(0.99, 0.985))
        fig.suptitle("G1 29dof 달리기 — 정책별 에너지 원시값 (평지, 256 env, 10 s 평균, iter 29999)",
                     x=0.01, ha="left", fontsize=13, color=INK, fontweight="bold", y=0.985)
        fig.text(0.01, 0.005, "위 행: 평균 파워 [W] · 아래 행: CoT = ∫P dt / (m g d), m = 33.34 kg, d = 몸통 경로 길이 · "
                 "Joule 열의 k_el은 T–N 곡선 추정값(1/3 스케일 가정)이라 Joule·전기적 값은 정책 간 상대 비교용",
                 fontsize=8, color=INK2)
        fig.tight_layout(rect=(0, 0.02, 1, 0.95))
        fig.savefig(out, dpi=200)
        plt.close(fig)

    def fig_reduction(self, out):
        """Energy saving vs baseline as positive % (higher = better), one line per policy, per CoT definition.
        The 5-speed mean is a dashed reference line in the series colour, labelled at the right edge."""
        panels = [("cot_pos", "기계적 CoT 절감률"), ("cot_heat", "Joule 열 CoT 절감률"),
                  ("cot_elec", "전기적 CoT 절감률  (일 + 열)")]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), facecolor=SURFACE, sharey=True)
        base = {col: self.series(self.base, col) for col, _ in panels}
        for ax, (col, title) in zip(axes, panels):
            style(ax, title, "baseline 대비 절감 [%]  (높을수록 좋음)", self.speeds)
            ax.set_xlim(0.3, 3.0)
            pts = []
            means = []
            for stem, label, c, m in self.policies[1:]:
                sav = (1 - self.series(stem, col) / base[col]) * 100
                ax.plot(self.speeds, sav, color=c, lw=2.2, marker=m, ms=7, mec=SURFACE, mew=1.2, label=label, zorder=3)
                ax.hlines(sav.mean(), 0.5, 2.5, color=c, lw=1, ls=(0, (3, 3)), alpha=0.7, zorder=2)
                means.append((sav.mean(), c))
                pts += [(x, v, c) for x, v in zip(self.speeds, sav)]
            # mean labels, pushed apart so two close means stay readable
            means.sort()
            ys = [mv for mv, _ in means]
            for i in range(1, len(ys)):
                if ys[i] - ys[i - 1] < 4.5:
                    ys[i] = ys[i - 1] + 4.5
            for (mv, c), y in zip(means, ys):
                ax.text(2.6, y, f"평균 {mv:.0f}%", ha="left", va="center", fontsize=8.5, color=c, fontweight="bold")
            ax.set_ylim(0, 85)
            ax.set_yticks(range(0, 81, 20))
            ax.set_yticklabels([f"{t}%" for t in range(0, 81, 20)])
            ax.set_xlabel("명령 전진 속도 [m/s]", color=INK2, fontsize=9)
            repel_labels(ax, pts, lambda v: f"{v:.0f}", fontsize=8)
        axes[1].set_ylabel(""); axes[2].set_ylabel("")
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper right", ncol=len(self.policies) - 1, frameon=False, fontsize=9.5,
                   bbox_to_anchor=(0.99, 0.985))
        fig.suptitle("baseline 대비 CoT 절감률 — 높을수록 같은 거리를 적은 에너지로 이동",
                     x=0.01, ha="left", fontsize=13, color=INK, fontweight="bold", y=0.985)
        fig.text(0.01, 0.005, "절감률 = (1 − 정책 CoT / baseline CoT) × 100, 같은 명령 속도끼리 비교 · 점선 = 5개 속도의 산술평균",
                 fontsize=8, color=INK2)
        fig.tight_layout(rect=(0, 0.03, 1, 0.93))
        fig.savefig(out, dpi=200)
        plt.close(fig)

    def fig_groups(self, out, speed=2.0):
        order = ["hip", "knee", "ankle", "waist", "arm"]
        names = {"hip": "hip (6)", "knee": "knee (2)", "ankle": "ankle (4)", "waist": "waist (3)", "arm": "arm (14)"}
        pols = [p for p in self.policies if next(x for x in self.data[p[0]] if x["cmd_mps"] == speed)["groups"]]

        def grp(stem, part):
            r = next(x for x in self.data[stem] if x["cmd_mps"] == speed)["groups"]
            g = {k: r[k][part] for k in r}
            g["arm"] = g.pop("shoulder", 0) + g.pop("elbow", 0) + g.pop("wrist", 0)
            return g

        n = len(pols)
        fig, axes = plt.subplots(1, 2, figsize=(5.2 + 2.1 * n, 5.8), facecolor=SURFACE, sharey=True)
        ymax = 0
        for ax, (part, title) in zip(axes, [("joule_w", r"권선 Joule 열  $\sum k_{el}\,\tau^2$"),
                                            ("pos_w", r"기계적 일  $\sum \max(\tau\,\dot{q},\,0)$")]):
            style(ax, title, "파워 [W]", self.speeds)
            ax.set_xlim(-0.6, n - 0.4)
            for i, (stem, label, c, m) in enumerate(pols):
                g = grp(stem, part)
                bottom = 0
                for k in order:
                    v = g[k]
                    ax.bar(i, v, bottom=bottom, width=0.62, color=GROUP_COL[k], edgecolor=SURFACE, lw=1.2, zorder=3,
                           label=names[k] if i == 0 else None)
                    if v >= 18:
                        ax.text(i, bottom + v / 2, f"{v:.0f}", ha="center", va="center", fontsize=8.5,
                                color="white" if k in ("hip", "knee", "waist") else INK)
                    bottom += v
                ymax = max(ymax, bottom)
                ax.text(i, bottom + 6, f"{bottom:.0f} W", ha="center", va="bottom", fontsize=9.5, color=INK,
                        fontweight="bold")
            ax.set_xticks(range(n))
            ax.set_xticklabels([p[1].replace("+", "\n+").replace(" (", "\n(") for p in pols], fontsize=8.5, color=INK)
        for ax in axes:
            ax.set_ylim(0, ymax * 1.15)
        axes[1].set_ylabel("")
        axes[0].legend(loc="upper right", frameon=False, fontsize=8.5, title="관절 그룹 (관절 수)", title_fontsize=8.5)
        fig.suptitle(f"관절 그룹별 분해 @ {speed:.1f} m/s — 막대 합 = 전신 파워", x=0.01, ha="left", fontsize=12.5,
                     color=INK, fontweight="bold", y=0.99)
        fig.text(0.01, 0.005, "eval_energy.py --per_group · arm = shoulder + elbow + wrist", fontsize=8, color=INK2)
        fig.tight_layout(rect=(0, 0.03, 1, 0.94))
        fig.savefig(out, dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", default="logs")
    ap.add_argument("--out", default="logs")
    ap.add_argument("--tag", default="1007")
    args = ap.parse_args()
    rep = Report(args.csv_dir)
    print("policies:", [p[1] for p in rep.policies], "font:", FONT)
    rep.fig_raw(f"{args.out}/report_energy_raw_{args.tag}.png")
    rep.fig_reduction(f"{args.out}/report_cot_reduction_{args.tag}.png")
    rep.fig_groups(f"{args.out}/report_group_breakdown_2ms_{args.tag}.png")
    print("saved")
