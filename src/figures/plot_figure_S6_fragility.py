#!/usr/bin/env python3
"""Figure S6: wind fragility curves on one wind convention.

All curves are plotted against the modeled 10-min mean site wind V (the
ER11E04 peak wind driven by the STORM intensity):

* conductor: event failure probability of one reference length L0 = 0.25 km of
  line, q(V) = 2**((V - 24)/30) - 1 between 24 and 54 m/s (0 below, 1 above);
* mapped tower or pole: Emanuel-form curve, V_thresh = 24 m/s, V_half = 110 kn
  (56.5 m/s);
* solar panel: physical damage, lognormal in the 3-s gust (median 80 m/s,
  beta 0.32) with gust = 1.64 V, so its 50% point is at V = 80 / 1.64 = 48.8 m/s.
  With ``--solar-exposure`` the exposure-dependent gust factors of the solar
  runs are drawn instead (``solar_exposure_gust.GUST``): 1.66 inland, 1.52 for
  offshore flow and 1.38 for onshore flow within the coastal band.

The conductor and tower curves are imported from ``line_hazard_v2`` (the
production hazard); the solar curve is the formula of the solar runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from line_hazard_v2 import KNOT_MS, V_HALF_TOWER_KN, line_curve, tower_curve
from solar_exposure_gust import GUST

THRESHOLD = 24.0
L0_KM = 0.25
SOLAR_GUST_FACTOR, SOLAR_MEDIAN_GUST, SOLAR_BETA = 1.64, 80.0, 0.32
CONDUCTOR, TOWER, SOLAR = "#0072B2", "#009E73", "#E69F00"
SOLAR_CLASSES = (  # (gust factor, label, colour, line style)
    (GUST["inland"], "inland", "#E69F00", (0, (1.2, 1.4))),
    (GUST["offshore"], "coastal, offshore flow", "#D55E00", (0, (3, 1.5, 1, 1.5))),
    (GUST["onshore"], "coastal, onshore flow", "#CC79A7", (0, (6, 1.5))),
)
GUIDE, GRID = "#9aa3ab", "#e6e9ec"


def solar_damage(wind: np.ndarray, factor: float = SOLAR_GUST_FACTOR) -> np.ndarray:
    gust = factor * wind
    out = np.zeros_like(gust)
    positive = gust > 0
    out[positive] = norm.cdf(np.log(gust[positive] / SOLAR_MEDIAN_GUST) / SOLAR_BETA)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="path without suffix")
    parser.add_argument("--x-max", type=float, default=120.0, help="upper limit of the wind axis (m/s)")
    parser.add_argument("--solar-exposure", action="store_true",
                        help="draw the exposure-dependent solar curves instead of gust = 1.64 V")
    args = parser.parse_args()
    for suffix in (".png", ".pdf"):
        if args.output.with_suffix(suffix).exists():
            raise SystemExit(f"refusing to overwrite {args.output.with_suffix(suffix)}")

    v = np.linspace(0, args.x_max, int(args.x_max * 20) + 1)
    conductor = line_curve(v, THRESHOLD)
    tower = tower_curve(v, THRESHOLD)
    solar = solar_damage(v)
    v_half_tower = V_HALF_TOWER_KN * KNOT_MS
    v_half_solar = SOLAR_MEDIAN_GUST / SOLAR_GUST_FACTOR
    v_full_conductor = THRESHOLD + 30.0

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": "#444444",
                         "axes.linewidth": 0.9, "xtick.major.width": 0.9, "ytick.major.width": 0.9})
    fig, ax = plt.subplots(figsize=(5.4, 4.9 if args.solar_exposure else 3.9))
    ax.axhline(0.5, color=GRID, lw=1.0, zorder=0)
    ax.axvline(THRESHOLD, color=GUIDE, lw=1.0, ls=(0, (1, 2.5)), zorder=1)
    ax.text(THRESHOLD - 0.008 * args.x_max, 0.55, "threshold\n24 m s$^{-1}$", ha="right", va="center", fontsize=8,
            color="#6b747c", linespacing=1.1)
    ax.plot(v, conductor, color=CONDUCTOR, lw=2.0, label=f"Conductor, per {L0_KM:g} km ($L_0$)")
    ax.plot(v, tower, color=TOWER, lw=2.0, ls=(0, (5, 2)), label="Mapped tower or pole")
    if args.solar_exposure:
        for factor, name, colour, style in SOLAR_CLASSES:
            half = SOLAR_MEDIAN_GUST / factor
            ax.plot(v, solar_damage(v, factor), color=colour, lw=2.0, ls=style,
                    label=f"Solar, {name} ($g$ = {factor:.2f}; 50% at {half:.1f})")
            ax.plot([half], [0.5], "o", ms=5.0, mfc="white", mec=colour, mew=1.4, zorder=5)
    else:
        ax.plot(v, solar, color=SOLAR, lw=2.0, ls=(0, (1.2, 1.4)), label="Solar panel (gust = 1.64 $V$)")

    ax.plot([v_full_conductor], [1.0], "o", ms=5.5, mfc="white", mec=CONDUCTOR, mew=1.5, zorder=5)
    ax.annotate(f"1 at {v_full_conductor:.0f} m s$^{{-1}}$", (v_full_conductor, 1.0), xytext=(-10, -18),
                textcoords="offset points", ha="right", fontsize=8.5, color=CONDUCTOR,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6}, zorder=4)
    if not args.solar_exposure:
        ax.plot([v_half_solar], [0.5], "o", ms=5.5, mfc="white", mec=SOLAR, mew=1.5, zorder=5)
        ax.annotate(f"50% at {v_half_solar:.1f} m s$^{{-1}}$\n(80 m s$^{{-1}}$ gust)", (v_half_solar, 0.5),
                    xytext=(v_half_solar + 0.30 * args.x_max, 0.33), textcoords="data", ha="center", va="center", fontsize=8.5, color="#b5770d",
                    arrowprops={"arrowstyle": "-", "color": "#b5770d", "lw": 0.8, "shrinkA": 2, "shrinkB": 4})
    ax.plot([v_half_tower], [0.5], "o", ms=5.5, mfc="white", mec=TOWER, mew=1.5, zorder=5)
    ax.annotate(f"50% at {v_half_tower:.1f} m s$^{{-1}}$", (v_half_tower, 0.5),
                xytext=(v_half_tower + 0.25 * args.x_max, 0.3 if args.solar_exposure else 0.6),
                textcoords="data", ha="center", va="center", fontsize=8.5, color=TOWER,
                arrowprops={"arrowstyle": "-", "color": TOWER, "lw": 0.8, "shrinkA": 2, "shrinkB": 4})

    ax.set_xlim(0, args.x_max)
    ax.set_ylim(-0.02, 1.04)
    step = 40 if args.x_max > 150 else 20 if args.x_max > 80 else 10
    ax.set_xticks(np.arange(0, args.x_max + 1e-9, step))
    ax.set_xlabel("Modeled 10-min mean site wind $V$ (m s$^{-1}$)", fontsize=11)
    ax.set_ylabel("Component failure probability", fontsize=11)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if args.solar_exposure:
        ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.17), fontsize=8.5, frameon=False, handlelength=2.6)
    else:
        ax.legend(loc="lower right", fontsize=8.5, frameon=False, handlelength=2.6)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(args.output.with_suffix(suffix), dpi=300, bbox_inches="tight")
    print(f"solar 50% at {v_half_solar:.2f} m/s; tower 50% at {v_half_tower:.2f} m/s; "
          f"conductor reaches 1 at {v_full_conductor:.0f} m/s; conductor at 40 m/s = {line_curve(np.array([40.0]), THRESHOLD)[0]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
