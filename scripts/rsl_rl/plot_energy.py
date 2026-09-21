# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Plot and compare energy-efficiency sweeps produced by eval_energy.py.

Each CSV is one policy. The first one is treated as the baseline, so the cost-of-transport
panel can annotate every other policy with its change against it.

Example:
    python scripts/rsl_rl/plot_energy.py \
        logs/energy_baseline_try6_25500.csv=baseline \
        logs/energy_reward_tuned.csv="reward tuned" \
        --out logs/energy_compare.png
"""

from __future__ import annotations

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots, in the fixed order documented in the data-viz reference palette.
# The first three are the all-pairs validated set; marker shape carries identity as well
# as hue so the series never rely on color alone.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SERIES_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e3e2de"

# (csv column, panel title, y-axis label, lower-is-better)
PANELS = [
    ("cot_pos", "CoT_mech (positive motor work)", "int sum max(tau q', 0) dt / (m g d)", True),
    ("cot_motor", "CoT_motor (sum tau^2)", "int sum tau^2 dt / (m g d)   [Nm s]", True),
    ("cot_elec", "Electrical CoT (k_el estimate)", "(sum tau^2 k_el + E_mech) / (m g d)", True),
    ("track_err", "Velocity tracking error", "|cmd - actual|   [m/s]", True),
]

SURVIVAL_FLOOR = 0.98


def _load(path: str) -> list[dict[str, float]]:
    with open(path) as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def _parse_series(specs: list[str]) -> list[tuple[str, list[dict[str, float]]]]:
    series = []
    for spec in specs:
        path, _, label = spec.partition("=")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"no such CSV: {path}")
        label = label or os.path.splitext(os.path.basename(path))[0]
        series.append((label, _load(path)))
    return series


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csvs", nargs="+", help="PATH[=LABEL] per policy; the first one is the baseline.")
    parser.add_argument("--out", type=str, default="logs/energy_compare.png")
    parser.add_argument("--title", type=str, default="G1 29dof running - energy efficiency")
    args = parser.parse_args()

    series = _parse_series(args.csvs)
    if len(series) > 3:
        print(
            f"[WARN] {len(series)} series: only the first three hues are validated for all-pairs "
            "colorblind separation. Marker shape and direct labels still separate them, but "
            "consider faceting into separate figures if the colors get hard to tell apart."
        )

    baseline_label, baseline_rows = series[0]
    baseline_cot = {row["cmd_mps"]: row["cot_pos"] for row in baseline_rows}

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), facecolor=SURFACE)
    fig.patch.set_facecolor(SURFACE)

    low_survival = []

    for panel_index, (column, panel_title, y_label, _) in enumerate(PANELS):
        ax = axes.flat[panel_index]
        ax.set_facecolor(SURFACE)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRID)
        ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=0)
        # Headroom so the baseline-delta labels never collide with the axis ticks.
        ax.margins(y=0.18)

        for series_index, (label, rows) in enumerate(series):
            color = SERIES_COLORS[series_index % len(SERIES_COLORS)]
            marker = SERIES_MARKERS[series_index % len(SERIES_MARKERS)]
            x = [row["cmd_mps"] for row in rows]
            y = [row[column] for row in rows]

            ax.plot(x, y, color=color, linewidth=2.0, zorder=3, label=label)
            # A window in which some env fell is not comparable with one where none did,
            # so those points are drawn hollow instead of silently averaged in.
            for xi, yi, row in zip(x, y, rows):
                ok = row["survival"] >= SURVIVAL_FLOOR
                ax.plot(
                    xi,
                    yi,
                    marker=marker,
                    markersize=8,
                    color=color,
                    markerfacecolor=color if ok else SURFACE,
                    markeredgecolor=color,
                    markeredgewidth=1.8,
                    zorder=4,
                )
                if not ok and panel_index == 0:
                    low_survival.append(f"{label} @ {xi:.1f} m/s: {row['survival'] * 100:.0f}% survived")

        if panel_index == 0 and len(series) > 1:
            # The number the reader actually wants: change against the baseline.
            for series_index, (label, rows) in enumerate(series[1:], start=1):
                for row in rows:
                    base = baseline_cot.get(row["cmd_mps"])
                    if base is None:
                        continue
                    delta = (row["cot_pos"] - base) / base * 100.0
                    ax.annotate(
                        f"{delta:+.0f}%",
                        (row["cmd_mps"], row["cot_pos"]),
                        textcoords="offset points",
                        xytext=(0, -15 if delta < 0 else 9),
                        ha="center",
                        fontsize=8,
                        color=INK_SECONDARY,
                    )

        ax.set_title(panel_title, color=INK_PRIMARY, fontsize=11, loc="left", pad=8)
        ax.set_xlabel("commanded forward speed  [m/s]", color=INK_SECONDARY, fontsize=9)
        ax.set_ylabel(y_label, color=INK_SECONDARY, fontsize=9)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.985),
        frameon=False,
        fontsize=9,
        ncol=min(len(series), 4),
    )
    for text in legend.get_texts():
        text.set_color(INK_PRIMARY)

    fig.suptitle(args.title, color=INK_PRIMARY, fontsize=13, x=0.012, ha="left", y=0.975)
    footnote = (
        f"lower is better in all four panels   |   baseline: {baseline_label}   |   "
        "hollow marker = some env terminated during the window"
    )
    if low_survival:
        footnote += "\n" + "; ".join(low_survival)
    else:
        footnote += "\nall runs: 100% survival"
    fig.text(0.012, 0.015, footnote, color=INK_SECONDARY, fontsize=8, ha="left")

    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=160, facecolor=SURFACE)
    print(f"[INFO] wrote {args.out}")

    # A table view so the figure is never the only way to read the numbers.
    columns = [
        "cmd_mps", "speed_mps", "pos_power_w", "cot_pos", "torque_sq_raw", "cot_motor", "cot_elec",
        "track_err", "flight_ratio", "survival",
    ]
    # Older CSVs use the pre-2026-09-21 column set; keep them readable instead of crashing.
    columns = [c for c in columns if all(c in row for _, rows in series for row in rows)]
    for label, rows in series:
        print(f"\n--- {label}")
        print("  ".join(f"{c:>10s}" for c in columns))
        for row in rows:
            print("  ".join(f"{row[c]:10.3f}" for c in columns))


if __name__ == "__main__":
    main()
