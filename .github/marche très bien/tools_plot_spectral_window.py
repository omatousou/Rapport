"""
Why the solver clips its frequency axis to the Sellmeier window.

Draws the three facts behind Material.sellmeier_range_um, all computed from the
solver's own code and from the real grid of the 4 uJ run, nothing sketched.

    python tools_plot_spectral_window.py [out.png]
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.constants import c

sys.path.insert(0, str(Path(__file__).resolve().parent / "sim"))
import keldysh                                       # noqa: E402

# ---- the actual run ---------------------------------------------------------
LAM0, DELTA_T, NT, TMAX_FACTOR = 1030e-9, 263e-15, 4096, 10.0
LAM_LO, LAM_HI = keldysh.SELLMEIER_RANGE_UM
POLES_UM = np.sqrt(keldysh.SELLMEIER_L2)

f0 = c / LAM0
tp = DELTA_T / np.sqrt(2 * np.log(2))
dt = 2.0 * (TMAX_FACTOR * tp) / NT
ff = np.fft.fftfreq(NT, d=dt)
f_abs = f0 + ff                       # absolute frequency of every grid bin
u = f_abs / f0                        # = T-hat where T-hat is defined

# the solver's mask, copied from grids.py
u_lo, u_hi, w_edge = (c / (LAM_HI * 1e-6)) / f0, (c / (LAM_LO * 1e-6)) / f0, 0.05
mask = 0.25 * (1 + np.tanh((u - u_lo) / w_edge)) * (1 + np.tanh((u_hi - u) / w_edge))

BLUE, RED, GREY, GREEN = "#2c6fbb", "#c0392b", "#7f8c8d", "#1e8449"
fig, ax = plt.subplots(3, 1, figsize=(11.0, 12.4))
fig.suptitle("Why the frequency axis is clipped to the Sellmeier window\n"
             f"fused silica, {LAM0*1e9:.0f} nm pump, Nt = {NT}",
             fontsize=14, fontweight="bold", y=0.985)

# ================================================================== panel A
# Sellmeier has poles. Outside the window the fit is not inaccurate, it blows up.
a = ax[0]
lam = np.logspace(np.log10(0.04), np.log10(30.0), 40000)
n2m1 = sum(B * lam**2 / (lam**2 - L2)
           for B, L2 in zip(keldysh.SELLMEIER_B, keldysh.SELLMEIER_L2))
n2 = 1.0 + n2m1
nn = np.where(n2 > 0, np.sqrt(np.abs(n2)), np.nan)
nn[np.abs(n2) > 1e3] = np.nan
a.plot(lam, nn, color=BLUE, lw=1.6, label="n from the Sellmeier fit")
bad = n2 <= 0
a.fill_between(lam, 0, 4, where=bad, color=RED, alpha=0.16, lw=0)
for p in POLES_UM:
    a.axvline(p, color=RED, ls="--", lw=1.1)
    a.annotate(f"pole\n{p:.3f}", (p, 3.55), color=RED, fontsize=8.5,
               ha="center", va="top")
a.axvspan(LAM_LO, LAM_HI, color=GREEN, alpha=0.11, lw=0)
a.axvline(LAM_LO, color=GREEN, lw=1.6)
a.axvline(LAM_HI, color=GREEN, lw=1.6)
a.annotate(f"the window\n{LAM_LO} to {LAM_HI} um", (np.sqrt(LAM_LO * LAM_HI), 0.62),
           color=GREEN, fontsize=10, ha="center", fontweight="bold")
a.plot([LAM0 * 1e6], [keldysh.n_sellmeier(LAM0)], "o", color="black", ms=6, zorder=5)
a.annotate(f"pump, n = {keldysh.n_sellmeier(LAM0):.4f}", (LAM0 * 1e6, 1.72),
           fontsize=9, ha="center")
a.set_xscale("log"); a.set_xlim(0.04, 30); a.set_ylim(0, 4)
a.set_xlabel("wavelength [um]"); a.set_ylabel("refractive index n")
a.set_title("A.  The fit has poles. The window is chosen to sit between them.",
            loc="left", fontsize=11.5, fontweight="bold")
a.legend(handles=[
    plt.Line2D([], [], color=BLUE, lw=1.6, label="n from the Sellmeier fit"),
    Patch(facecolor=RED, alpha=0.16, label="n$^2$ < 0, the fit returns nonsense"),
    Patch(facecolor=GREEN, alpha=0.11, label="where the fit is meaningful")],
    loc="lower right", fontsize=9, framealpha=0.95)
a.grid(alpha=0.25)

# ================================================================== panel B
# T-hat = omega/omega0 changes sign below zero. And note where the mask edges
# actually fall relative to Nyquist.
b = ax[1]
o = np.argsort(u)
u_nyq = u.max()
neg = f_abs <= 0                      # bins at negative absolute frequency
b.plot(u[o], u[o], color=BLUE, lw=1.7)
b.plot(u[o], (u**2)[o], color="#8e44ad", lw=1.5, ls="-.")
b.axhline(0, color="black", lw=0.9)
b.axhline(1, color=GREY, lw=0.8, ls=":")
b.fill_between([u.min(), 0], -1.6, 7.6, color=RED, alpha=0.16, lw=0)
b.annotate(f"absolute frequency < 0\n{neg.sum()} of {NT} bins = {100*neg.mean():.1f} %\n"
           r"$\hat{T}$ is negative: an operator" "\n" r"standing for $\partial/\partial t$"
           " cannot be",
           (u.min() * 0.5, 4.6), color=RED, fontsize=9.5, ha="center", va="center")
b.axvline(u_lo, color=GREEN, lw=1.8)
b.annotate(f"mask edge\n{u_lo:.2f} $\\omega_0$ (5 um)", (u_lo + 0.05, -1.15),
           color=GREEN, fontsize=9, ha="left", fontweight="bold")
b.axvline(u_nyq, color="#d35400", lw=1.6, ls="--")
b.annotate(f"Nyquist\n{u_nyq:.2f} $\\omega_0$", (u_nyq - 0.05, -1.15),
           color="#d35400", fontsize=9, ha="right", fontweight="bold")
b.plot([u_nyq], [u_nyq**2], "o", color="#8e44ad", ms=7, zorder=5)
b.annotate(rf"$\hat{{T}}^2$ = {u_nyq**2:.1f} here." "\n"
           f"The mask's other edge is at {u_hi:.2f} $\\omega_0$,\n"
           f"{u_hi/u_nyq:.1f}x Nyquist, so it never acts:\n"
           "this amplification is NOT masked.",
           (0.62, 7.2), fontsize=9.5, ha="left", va="top", color="#5b2c6f")
b.set_xlim(u.min() * 1.06, u_nyq * 1.10); b.set_ylim(-1.6, 7.6)
b.set_xlabel(r"absolute frequency $\omega/\omega_0$ of the grid bin")
b.set_ylabel(r"$\hat{T}$  and  $\hat{T}^2$")
b.set_title(r"B.  Reason two: $\hat{T} = \omega/\omega_0$ changes sign where "
            r"$\omega < 0$.", loc="left", fontsize=11.5, fontweight="bold")
b.legend(handles=[
    plt.Line2D([], [], color=BLUE, lw=1.7, label=r"$\hat{T} = \omega/\omega_0$"),
    plt.Line2D([], [], color="#8e44ad", lw=1.5, ls="-.",
               label=r"$\hat{T}^2$, the factor on the Kerr term")],
    loc="upper left", fontsize=9, framealpha=0.95)
b.grid(alpha=0.25)

# ================================================================== panel C
# What the mask removes, and where the field actually is.
d = ax[2]
spec = np.exp(-(2 * np.pi * ff) ** 2 * tp**2 / 2.0)
spec = np.clip(spec / spec.max(), 1e-20, None)
cut = mask < 0.5
d.semilogy(u[o], spec[o], color=GREY, lw=1.5)
d.semilogy(u[o], np.clip(mask[o], 1e-20, None), color=GREEN, lw=2.2)
d.fill_between([u.min(), u_lo], 1e-20, 10, color=RED, alpha=0.13, lw=0)
d.axvline(u_lo, color=GREEN, lw=1.8)
d.axvline(u_nyq, color="#d35400", lw=1.6, ls="--")
d.annotate(f"removed: {cut.sum()} bins = {100*cut.mean():.1f} %\n"
           "ALL of it at the red end",
           (u.min() * 0.52, 1e-9), fontsize=10, ha="center", color=RED,
           fontweight="bold")
d.annotate("kept: everything from 0.21 $\\omega_0$ up.\nThe blue edge of the mask is off this\n"
           "grid entirely, so nothing is cut there.",
           (0.42, 3e-13), fontsize=9.5, ha="left", color=GREEN)
d.annotate("tanh edge, not a hard cut:\na sharp truncation would ring",
           (u_lo + 0.04, 4e-3), fontsize=9, ha="left", color=GREEN)
d.annotate("the input pulse is a spike next to\nthe grid. Self-phase modulation is\n"
           "what pushes light out toward the edges.",
           (0.52, 1e-6), fontsize=9.5, ha="left", color="#34495e")
d.set_xlim(u.min() * 1.06, u_nyq * 1.10); d.set_ylim(1e-18, 6)
d.set_xlabel(r"absolute frequency $\omega/\omega_0$ of the grid bin")
d.set_ylabel("normalised, log scale")
d.set_title("C.  What is actually removed.", loc="left",
            fontsize=11.5, fontweight="bold")
d.legend(handles=[
    plt.Line2D([], [], color=GREY, lw=1.5, label="input pulse spectrum (263 fs)"),
    plt.Line2D([], [], color=GREEN, lw=2.2, label="the tanh spectral mask"),
    plt.Line2D([], [], color="#d35400", lw=1.6, ls="--", label="Nyquist")],
    loc="upper right", fontsize=9, framealpha=0.95)
d.grid(alpha=0.25, which="major")

fig.tight_layout(rect=(0, 0.005, 1, 0.965))
out = Path(sys.argv[1] if len(sys.argv) > 1 else "spectral_window.png")
fig.savefig(out, dpi=150, facecolor="white")
print(f"wrote {out}")
print(f"  pump f0            = {f0/1e12:.1f} THz")
print(f"  Nyquist / f0       = {1/(2*dt)/f0:.2f}")
print(f"  absolute frequency = {f_abs.min()/1e12:.1f} to {f_abs.max()/1e12:.1f} THz")
print(f"  bins with omega<0  = {neg.sum()} / {NT} = {100*neg.mean():.1f} %")
print(f"  T^2 at blue edge   = {u_nyq**2:.2f}")
print(f"  mask edges         = {u_lo:.3f} and {u_hi:.3f} omega0")
print(f"  Nyquist            = {u_nyq:.3f} omega0 -> blue edge is {u_hi/u_nyq:.1f}x Nyquist")
print(f"  bins removed       = {cut.sum()} ({100*cut.mean():.1f} %), all at the red end")
print(f"  poles at           = {', '.join(f'{p:.4f}' for p in POLES_UM)} um")
