"""
Why n_valence_per_unit = 8, drawn.

The point in one sentence: promoting one electron does not switch off a whole
SiO2 unit, it removes one oscillator out of the eight that unit carries.

    python tools_plot_why_eight.py [out.png]
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

BLUE, RED, GREEN, GREY = "#2c6fbb", "#c0392b", "#1e8449", "#7f8c8d"
O_COL, SI_COL, E_COL = "#e8a49c", "#9ecbf0", "#34495e"

SI = np.array([5.0, 5.6])
O_POS = [np.array(p) for p in ((2.3, 7.9), (7.7, 7.9), (2.3, 3.3), (7.7, 3.3))]


def draw_unit(ax, missing=None, dim_all=False):
    """One SiO2 unit. Each bond carries its two shared electrons as dots.

    `missing` is the index (0..7) of an electron that has been promoted.
    """
    k = 0
    for op in O_POS:
        v = op - SI
        L = np.linalg.norm(v)
        u = v / L
        perp = np.array([-u[1], u[0]])
        ax.plot([SI[0], op[0]], [SI[1], op[1]], color=GREY, lw=5.5,
                solid_capstyle="round", zorder=1, alpha=0.45 if dim_all else 1.0)
        # the two shared electrons, straddling the bond axis at its midpoint
        mid = SI + 0.52 * v
        for sgn in (-1, +1):
            pos = mid + sgn * 0.36 * perp
            gone = (missing is not None and k == missing)
            ax.plot(*pos, "o", ms=11,
                    color="white" if gone else E_COL,
                    markeredgecolor=RED if gone else E_COL,
                    markeredgewidth=2.0 if gone else 1.0, zorder=6)
            if gone:
                ax.plot(*pos, "x", ms=8, color=RED, mew=2.4, zorder=7)
            k += 1
        ax.plot(*op, "o", ms=30, color=O_COL, zorder=4,
                markeredgecolor="black", markeredgewidth=0.8)
        ax.annotate("O", op, ha="center", va="center", fontsize=12,
                    fontweight="bold", zorder=8)
    ax.plot(*SI, "o", ms=40, color=SI_COL, zorder=4,
            markeredgecolor="black", markeredgewidth=0.8)
    ax.annotate("Si", SI, ha="center", va="center", fontsize=13,
                fontweight="bold", zorder=8)


fig, ax = plt.subplots(1, 3, figsize=(16.6, 6.4))
fig.suptitle("Why n_valence_per_unit = 8",
             fontsize=16, fontweight="bold", y=0.975)

# ------------------------------------------------------------------ panel A
a = ax[0]
a.set_xlim(0, 10); a.set_ylim(0, 11); a.axis("off")
draw_unit(a)
a.annotate("A.  Count the electrons that polarize", (0.1, 10.6),
           fontsize=13, fontweight="bold", va="top")
a.annotate("4 Si-O bonds", (5.0, 1.95), ha="center", fontsize=12, color=GREY)
a.annotate("x 2 shared electrons each", (5.0, 1.25), ha="center", fontsize=12,
           color=GREY)
a.annotate("= 8 valence electrons per SiO$_2$", (5.0, 0.45), ha="center",
           fontsize=13.5, color=GREEN, fontweight="bold")
a.annotate("each dot is one oscillator:\nit is what makes n$_0$ = 1.45",
           (0.15, 9.4), fontsize=10.5, color=E_COL, va="top")

# ------------------------------------------------------------------ panel B
b = ax[1]
b.set_xlim(0, 10); b.set_ylim(0, 11); b.axis("off")
draw_unit(b, missing=3)
b.annotate("B.  Now ionize ONE electron", (0.1, 10.6),
           fontsize=13, fontweight="bold", va="top")
b.add_patch(FancyArrowPatch((6.45, 7.25), (8.9, 9.6), color=RED, lw=2.2,
                            arrowstyle="-|>", mutation_scale=18,
                            connectionstyle="arc3,rad=-0.25", zorder=9))
b.annotate("to the\nconduction\nband", (9.0, 9.9), fontsize=10.5, color=RED,
           ha="center", va="bottom", fontweight="bold")
b.annotate("7 of the 8 oscillators are still there.", (5.0, 1.95), ha="center",
           fontsize=12, color=GREY)
b.annotate("This unit lost 1/8 of its polarizability,", (5.0, 1.25),
           ha="center", fontsize=12.5, color=GREEN, fontweight="bold")
b.annotate("not all of it.", (5.0, 0.5), ha="center", fontsize=12.5,
           color=GREEN, fontweight="bold")

# ------------------------------------------------------------------ panel C
d = ax[2]
d.set_xlim(0, 10); d.set_ylim(0, 11); d.axis("off")
d.annotate("C.  So what does one promoted electron cost?", (0.1, 10.6),
           fontsize=13, fontweight="bold", va="top")

for row, (lab, frac, col, verdict) in enumerate((
        ("n_valence_per_unit = 8", 1 / 8, GREEN, "right"),
        ("n_valence_per_unit = 1", 1.0, RED, "wrong"))):
    y = 8.3 - row * 3.5
    d.annotate(f"{lab}   ({verdict})", (0.35, y + 1.28), fontsize=12,
               color=col, fontweight="bold")
    # the unit's polarizability, as eight slots
    for j in range(8):
        x = 0.35 + j * 1.09
        filled = (j > 0) if frac < 1 else False
        d.add_patch(Rectangle((x, y), 0.95, 1.0,
                              facecolor=col if filled else "white",
                              edgecolor=col, lw=1.8, alpha=0.75 if filled else 1.0))
        if not filled:
            d.plot([x + 0.16, x + 0.79], [y + 0.18, y + 0.82], color=col, lw=2.0)
            d.plot([x + 0.16, x + 0.79], [y + 0.82, y + 0.18], color=col, lw=2.0)
    d.annotate(f"removes {100*frac:.1f} % of the unit's polarizability",
               (0.35, y - 0.62), fontsize=11.5, color=col, va="top")

d.annotate(r"$x = \dfrac{N_{removed}}{N_0 \times 8}$   and   "
           r"$\Delta\varepsilon = -(n_0^2 - 1)\, x$",
           (5.0, 1.55), ha="center", fontsize=14, va="center",
           bbox=dict(boxstyle="round,pad=0.55", fc="#f4f6f7", ec=GREY, lw=1.2))
d.annotate("the wrong count says one promoted electron\n"
           "kills a whole SiO$_2$ unit. Eight times too much.",
           (5.0, 0.35), ha="center", fontsize=11, color=RED, va="center")

fig.tight_layout(rect=(0, 0.0, 1, 0.945))
out = Path(sys.argv[1] if len(sys.argv) > 1 else "why_eight.png")
fig.savefig(out, dpi=150, facecolor="white")
print(f"wrote {out}")
