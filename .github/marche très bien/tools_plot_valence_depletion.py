"""
Why n_valence_per_unit = 8, and what happens if you get it wrong.

N0 carries two different meanings in Martin et al. 1997 under one symbol: the
density of ionizable units, and the number of valence oscillators. The
depletion term needs the second. This draws the consequence, using the real
permittivity model of sim/permittivity.py, nothing sketched.

    python tools_plot_valence_depletion.py [out.png]
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent / "sim"))
from permittivity import MaterialResponse, XPM        # noqa: E402
from keldysh import n_sellmeier                       # noqa: E402

PROBE_NM = 490.0
LAM = PROBE_NM * 1e-9
N0_MOL = 2.2e22                 # ionizable units per cm3
N_VAL = 8.0                     # valence electrons per SiO2

n0 = n_sellmeier(LAM)
rho = np.logspace(17.0, 21.3, 400)          # free carriers, cm^-3

right = MaterialResponse(n2_m2W=2.74e-20, N0_cm3=N0_MOL, n_valence_per_unit=N_VAL)
wrong = MaterialResponse(n2_m2W=2.74e-20, N0_cm3=N0_MOL, n_valence_per_unit=1.0)


def parts(mat, rho_e, rho_s):
    r = mat.response(LAM, n0, rho_e_cm3=rho_e, rho_s_cm3=rho_s,
                     I_Wcm2=np.zeros_like(rho_e), xpm_factor=XPM,
                     include=("drude", "ste", "depletion"))
    return r


BLUE, RED, GREEN, PURPLE, GREY = "#2c6fbb", "#c0392b", "#1e8449", "#8e44ad", "#7f8c8d"
fig, ax = plt.subplots(1, 3, figsize=(16.4, 5.9))
fig.suptitle("n_valence_per_unit: N0 means two different things in Martin et al. 1997\n"
             f"fused silica, probe at {PROBE_NM:.0f} nm",
             fontsize=13.5, fontweight="bold", y=0.99)

# ============================================================ panel A: the count
a = ax[0]
a.set_xlim(0, 10); a.set_ylim(0, 10); a.axis("off")
a.set_title("A.  One SiO$_2$ unit, two different counts",
            loc="left", fontsize=12, fontweight="bold")

# a little Si-O bond diagram
si = (5.0, 7.7)
o_pos = [(2.7, 9.1), (7.3, 9.1), (2.7, 6.3), (7.3, 6.3)]
for op in o_pos:
    a.plot([si[0], op[0]], [si[1], op[1]], color=GREY, lw=6, solid_capstyle="round",
           zorder=1)
    a.plot([si[0], op[0]], [si[1], op[1]], color="white", lw=1.4, ls=":", zorder=2)
    a.plot(*op, "o", ms=26, color="#d98880", zorder=3)
    a.annotate("O", op, ha="center", va="center", fontsize=11, fontweight="bold",
               zorder=4)
a.plot(*si, "o", ms=34, color="#85c1e9", zorder=3)
a.annotate("Si", si, ha="center", va="center", fontsize=12, fontweight="bold", zorder=4)

a.annotate("4 Si-O bonds, 2 electrons each\n= 8 valence electrons per unit",
           (5.0, 5.2), ha="center", va="top", fontsize=11, color=GREEN,
           fontweight="bold")
a.annotate("(a) as an IONIZABLE UNIT it counts once\n"
           f"      N$_0$ = {N0_MOL:.1e} cm$^{{-3}}$\n"
           "      used in the source term  N$_0$ $\\sigma_K$ F$^K$",
           (0.15, 3.5), ha="left", va="top", fontsize=10.5, color=BLUE)
a.annotate("(b) as a POLARIZABLE OBJECT it counts eight times\n"
           f"      N$_0$ x 8 = {N0_MOL*N_VAL:.2e} cm$^{{-3}}$\n"
           "      used in the depletion factor  (N$_0$ - N$_{CB}$ - N$_{tr}$)",
           (0.15, 1.7), ha="left", va="top", fontsize=10.5, color=GREEN)

# ============================================ panel B: size of the depletion term
b = ax[1]
z = np.zeros_like(rho)
r_ok = parts(right, rho, z)
r_no = parts(wrong, rho, z)
b.loglog(rho, np.abs(r_ok["dn_drude"]), color=BLUE, lw=2.0,
         label="Drude term (free carriers)")
b.loglog(rho, np.abs(r_ok["dn_depletion"]), color=GREEN, lw=2.0,
         label="depletion, n_valence_per_unit = 8  (right)")
b.loglog(rho, np.abs(r_no["dn_depletion"]), color=RED, lw=2.0, ls="--",
         label="depletion, n_valence_per_unit = 1  (wrong)")
i20 = np.argmin(np.abs(rho - 1e20))
ratio_ok = abs(r_ok["dn_depletion"][i20] / r_ok["dn_drude"][i20])
ratio_no = abs(r_no["dn_depletion"][i20] / r_no["dn_drude"][i20])
b.axvline(1e20, color=GREY, ls=":", lw=1.2)
b.annotate(f"at 10$^{{20}}$ cm$^{{-3}}$\n"
           f"right: {100*ratio_ok:.1f} % of Drude\n"
           f"wrong: {100*ratio_no:.1f} % of Drude",
           (1.3e17, 2e-3), fontsize=10, ha="left",
           bbox=dict(boxstyle="round,pad=0.45", fc="white", ec=GREY, alpha=0.92))
b.set_xlabel("free-carrier density [cm$^{-3}$]")
b.set_ylabel(r"$|\Delta n|$ contribution")
b.set_title("B.  Getting it wrong inflates the term eightfold",
            loc="left", fontsize=12, fontweight="bold")
b.legend(loc="lower right", fontsize=9.2, framealpha=0.95)
b.grid(alpha=0.25, which="both")
b.set_ylim(1e-7, 1e0)

# ==================================== panel C: it flips the sign of the plateau
# At long delay the carriers have been trapped: rho_e -> 0 and the population
# sits in STEs. That is what the interferometer sees at late delay. Everything
# here is linear in rho_s, so one density tells the whole story.
d = ax[2]
RHO_S = 1e20
z1 = np.array([0.0])
p_ok = parts(right, z1, np.array([RHO_S]))
p_no = parts(wrong, z1, np.array([RHO_S]))
ste = float(p_ok["dn_ste"][0]) * 1e3
dep_ok, dep_no = float(p_ok["dn_depletion"][0]) * 1e3, float(p_no["dn_depletion"][0]) * 1e3
tot_ok, tot_no = ste + dep_ok, ste + dep_no

xs = np.array([0, 1, 2])
w = 0.34
d.bar(xs - w/2, [ste, dep_ok, tot_ok], w, color=[PURPLE, GREEN, GREEN],
      edgecolor="black", lw=0.7, label="n_valence_per_unit = 8  (right)")
d.bar(xs + w/2, [ste, dep_no, tot_no], w, color=[PURPLE, RED, RED],
      edgecolor="black", lw=0.7, hatch="///",
      label="n_valence_per_unit = 1  (wrong)")
for x, v in zip(xs - w/2, [ste, dep_ok, tot_ok]):
    d.annotate(f"{v:+.2f}", (x, v), ha="center",
               va="bottom" if v > 0 else "top", fontsize=9.5, fontweight="bold")
for x, v in zip(xs + w/2, [ste, dep_no, tot_no]):
    d.annotate(f"{v:+.2f}", (x, v), ha="center",
               va="bottom" if v > 0 else "top", fontsize=9.5, fontweight="bold")
d.axhline(0, color="black", lw=1.2)
d.set_xticks(xs)
d.set_xticklabels(["STE bands\n(same both ways)", "valence\ndepletion", "TOTAL"],
                  fontsize=10)
d.set_ylabel(r"$\Delta n \times 10^{3}$ at the long-delay plateau")
d.set_title(f"C.  And it flips the sign of the plateau\n"
            f"      trapped density $10^{{20}}$ cm$^{{-3}}$, no free carriers left",
            loc="left", fontsize=12, fontweight="bold")
d.annotate("Fig. 6 of the paper shows a\nPOSITIVE plateau at long delay",
           (2.0, 0.95 * tot_ok + 0.35), ha="center", fontsize=10, color=GREEN)
d.set_ylim(min(dep_no, tot_no) * 1.45, max(ste, tot_ok) * 1.45)
d.legend(loc="lower left", fontsize=9.2, framealpha=0.95)
d.grid(alpha=0.25, axis="y")

fig.tight_layout(rect=(0, 0.005, 1, 0.93))
out = Path(sys.argv[1] if len(sys.argv) > 1 else "valence_depletion.png")
fig.savefig(out, dpi=150, facecolor="white")
print(f"wrote {out}")
print(f"  probe {PROBE_NM:.0f} nm, n0 = {n0:.4f}")
print(f"  denominator used, right : {N0_MOL*N_VAL:.3e} cm^-3")
print(f"  denominator used, wrong : {N0_MOL:.3e} cm^-3")
print(f"  at rho = 1e20, depletion / Drude : right {100*ratio_ok:.1f} %, "
      f"wrong {100*ratio_no:.1f} %")
print(f"  at rho_s = 1e20 (x1e3): STE {ste:+.3f} both ways")
print(f"    depletion  right {dep_ok:+.3f}   wrong {dep_no:+.3f}")
print(f"    TOTAL      right {tot_ok:+.3f}   wrong {tot_no:+.3f}   <- opposite signs")
print("  everything is linear in rho_s, so the sign is the same at every density")
