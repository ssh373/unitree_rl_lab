# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Split the energy sweep into its two physical halves and plot each against the baseline.

``eval_energy.py`` reports one combined electrical cost. This script separates the two terms
that go into it and draws three panels:

    1. Joule heat          sum k_el * tau^2            [W]   -- the "torque squared times a
                                                               constant" half; k_el is the
                                                               per-motor-family estimate in
                                                               ``UNITREE_G1_29DOF_K_EL``
    2. Mechanical work     sum max(tau * qdot, 0)      [W]   -- the useful half (braking is
                                                               dissipated, not recovered)
    3. Cost of transport   both halves over m g d            -- solid: total (heat + work),
                                                               dashed: the mechanical half alone,
                                                               so the gap is the heat half

The first CSV is the baseline; every other series is annotated with its percent change
against it at each commanded speed.

Joule power in watts is not a column of the sweep CSV. Where ``<csv>_groups.csv`` exists
(``eval_energy.py --per_group``) it is summed exactly from the per-group ``joule_w``;
otherwise it is recovered from ``pos_power_w * (cot_elec - cot_pos) / cot_pos``, which agrees
with the exact sum to better than 0.2% on every run checked. ``cot_elec - cot_pos`` is exact
either way -- the CoT columns are per-env means of the same denominator.

Example:
    python scripts/rsl_rl/plot_energy_components.py \
        logs/energy_new_baseline_29999.csv=baseline \
        logs/energy_new_PACEedit_29999.csv="PACE edit" \
        logs/energy_new_AMPedit_29999.csv="AMP" \
        logs/energy_new_AMPpace_29999.csv="AMP + PACE" \
        --out logs/energy_components.png
"""

from __future__ import annotations

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots in the fixed order of the data-viz reference palette; marker shape carries
# identity alongside hue so no series is told apart by color alone.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SERIES_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e3e2de"

SURVIVAL_FLOOR = 0.98


def _load(path: str) -> list[dict[str, float]]:
    with open(path) as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def _exact_joule_w(path: str) -> dict[float, float]:
    """Whole-body Joule power per commanded speed from the per-group CSV, if one was written."""
    group_path = os.path.splitext(path)[0] + "_groups.csv"
    if not os.path.isfile(group_path):
        return {}
    totals: dict[float, float] = {}
    with open(group_path) as f:
        for row in csv.DictReader(f):
            speed = float(row["cmd_mps"])
            totals[speed] = totals.get(speed, 0.0) + float(row["joule_w"])
    return totals


def _derive(rows: list[dict[str, float]], exact: dict[float, float]) -> None:
    """Add the four derived columns the panels plot."""
    for row in rows:
        row["cot_mech"] = row["cot_pos"]
        row["cot_joule"] = row["cot_elec"] - row["cot_pos"]
        row["mech_w"] = row["pos_power_w"]
        speed = row["cmd_mps"]
        if speed in exact:
            row["joule_w"] = exact[speed]
            row["joule_exact"] = 1.0
        else:
            row["joule_w"] = row["pos_power_w"] * row["cot_joule"] / max(row["cot_pos"], 1e-9)
            row["joule_exact"] = 0.0


def _parse_series(specs: list[str]) -> list[tuple[str, list[dict[str, float]]]]:
    series = []
    for spec in specs:
        path, _, label = spec.partition("=")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"no such CSV: {path}")
        rows = _load(path)
        _derive(rows, _exact_joule_w(path))
        series.append((label or os.path.splitext(os.path.basename(path))[0], rows))
    return series


# (column, panel title, y label, annotate percent change against the baseline)
PANELS = [
    ("joule_w", "Joule heat   $\\sum k_{el}\\,\\tau^2$", "heat [W]", True),
    ("mech_w", "Mechanical work   $\\sum \\max(\\tau\\,\\dot{q},\\,0)$", "positive mechanical power [W]", True),
    ("cot_elec", "Cost of transport   (heat + work) / $mgd$", "CoT  [-]", True),
]


def _style(ax):
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=0)
    ax.margins(x=0.10, y=0.22)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csvs", nargs="+", help="PATH[=LABEL] per policy; the first one is the baseline.")
    parser.add_argument("--out", type=str, default="logs/energy_components.png")
    parser.add_argument("--title", type=str, default="G1 29dof running - heat, work and cost of transport")
    args = parser.parse_args()

    series = _parse_series(args.csvs)
    baseline_label, baseline_rows = series[0]

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.4), facecolor=SURFACE)
    low_survival = []

    for panel_index, (column, panel_title, y_label, annotate) in enumerate(PANELS):
        ax = axes[panel_index]
        _style(ax)
        base_by_speed = {row["cmd_mps"]: row[column] for row in baseline_rows}

        for series_index, (label, rows) in enumerate(series):
            color = SERIES_COLORS[series_index % len(SERIES_COLORS)]
            marker = SERIES_MARKERS[series_index % len(SERIES_MARKERS)]
            x = [row["cmd_mps"] for row in rows]
            y = [row[column] for row in rows]
            ax.plot(x, y, color=color, linewidth=2.0, zorder=3, label=label)

            # The mechanical half drawn under the total, so the gap between the two lines is
            # the Joule half -- "each one's CoT" without a fourth panel.
            if column == "cot_elec":
                ax.plot(
                    x,
                    [row["cot_mech"] for row in rows],
                    color=color,
                    linewidth=1.3,
                    linestyle=(0, (4, 2)),
                    alpha=0.85,
                    zorder=2,
                )

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

            if annotate and series_index > 0:
                for row in rows:
                    base = base_by_speed.get(row["cmd_mps"])
                    if not base:
                        continue
                    delta = (row[column] - base) / base * 100.0
                    # Stagger by series so two policies that land on nearly the same value do
                    # not print their deltas on top of each other.
                    step = 11 * (series_index - 1)
                    dy = -(14 + step) if delta < 0 else (10 + step)
                    # Horizontal stagger separates series that land on the same value, but at
                    # the first and last speed it would push the label off the axis.
                    edge = row["cmd_mps"] in (x[0], x[-1])
                    dx = 0 if edge else 17 * (series_index - 2)
                    ax.annotate(
                        f"{delta:+.0f}%",
                        (row["cmd_mps"], row[column]),
                        textcoords="offset points",
                        xytext=(dx, dy),
                        ha="center",
                        fontsize=7.5,
                        color=INK_SECONDARY,
                    )

        ax.set_title(panel_title, color=INK_PRIMARY, fontsize=11, loc="left", pad=8)
        ax.set_xlabel("commanded forward speed  [m/s]", color=INK_SECONDARY, fontsize=9)
        ax.set_ylabel(y_label, color=INK_SECONDARY, fontsize=9)

    axes[2].annotate(
        "solid: total      dashed: mechanical half",
        (0.98, 0.02),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=8,
        color=INK_SECONDARY,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles, labels, loc="upper right", bbox_to_anchor=(0.995, 0.99), frameon=False,
        fontsize=9, ncol=min(len(series), 4),
    )
    for text in legend.get_texts():
        text.set_color(INK_PRIMARY)

    fig.suptitle(args.title, color=INK_PRIMARY, fontsize=13, x=0.008, ha="left", y=0.975)
    derived = [label for label, rows in series if not rows[0]["joule_exact"]]
    footnote = (
        f"lower is better in all three panels   |   percentages are against {baseline_label}   |   "
        "hollow marker = some env terminated during the window"
    )
    footnote += "\n" + ("; ".join(low_survival) if low_survival else "all runs: 100% survival")
    # A policy can get cheap by dropping the flight phase, so the gait is reported next to the cost.
    flight = []
    for label, rows in series:
        at_two = min(rows, key=lambda r: abs(r["cmd_mps"] - 2.0))
        flight.append(f"{label} {at_two['flight_ratio'] * 100:.0f}%")
    footnote += "   |   flight ratio @ 2.0 m/s: " + ", ".join(flight)
    if derived:
        footnote += f"   |   Joule watts recovered from the CoT columns (<0.2% error) for: {', '.join(derived)}"
    fig.text(0.008, 0.015, footnote, color=INK_SECONDARY, fontsize=8, ha="left")

    fig.tight_layout(rect=(0, 0.07, 1, 0.92))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=160, facecolor=SURFACE)
    print(f"[INFO] wrote {args.out}")

    # The table view, so the figure is never the only way to read the numbers.
    columns = ["cmd_mps", "speed_mps", "joule_w", "mech_w", "cot_joule", "cot_mech", "cot_elec",
               "track_err", "flight_ratio", "survival"]
    for label, rows in series:
        print(f"\n--- {label}")
        print("  ".join(f"{c:>11s}" for c in columns))
        for row in rows:
            print("  ".join(f"{row[c]:11.3f}" for c in columns))

    print(f"\n--- percent change against {baseline_label}")
    keys = ["joule_w", "mech_w", "cot_joule", "cot_mech", "cot_elec"]
    print("  ".join(f"{c:>13s}" for c in ["policy", "cmd_mps"] + keys))
    for label, rows in series[1:]:
        for row in rows:
            base = {r["cmd_mps"]: r for r in baseline_rows}.get(row["cmd_mps"])
            if base is None:
                continue
            cells = [f"{(row[k] - base[k]) / base[k] * 100:+12.1f}%" for k in keys]
            print("  ".join([f"{label:>13s}", f"{row['cmd_mps']:13.2f}"] + cells))


if __name__ == "__main__":
    main()
