#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
self_steepening.py -- figures for the optical shock (self-steepening) term.

The shock term is the second half of the Kerr operator T_hat = 1 + (i/w0) d/dt.
Expanding it in the propagation equation gives

    dE/dz = i K0 n2 I E            (SPM, pure phase)
            - (n2/c) d(I E)/dt     (shock)

and writing E = sqrt(I) exp(i phi) turns the real part into the inviscid Burgers
equation

    dI/dz + 3 (n2/c) I dI/dt = 0                                          (1)

so every intensity level travels at its own speed 3 n2 I / c. High-intensity
points drift toward later times, the peak overtakes the trailing edge, and a
shock forms at

    z_shock = 0.39 c t_p / (n2 I_p)   for a Gaussian I_p exp(-t^2/t_p^2)   (2)

which is the classical Anderson-Lisak result. Eq. (2) is a falsifiable
prediction, and `fig_shock_scaling` checks it against a direct numerical
integration rather than asserting it.

Two families of figures:

  * standalone (no simulation needed, no GPU, ~1 s):
        fig_steepening_maps()     -- I(t,z) and spectrum(z) 2D maps, shock on/off
        fig_shock_scaling()       -- Rothenberg-style scan over the input pulse,
                                     measured z_shock vs Eq. (2)
  * from a real run (needs two result.npz, shock on and off):
        fig_steepening_from_run(res_on, res_off)

Fourier convention. The document uses E(t) = (1/2pi) int E(W) exp(-i W t) dW, so
the forward transform is int E(t) exp(+i W t) dt, which is numpy's *ifft*. Using
`np.fft.ifft` for the spectrum therefore puts the physical detuning W directly
on the `2 pi fftfreq` axis, with no sign flip. Using `fft` instead would mirror
the spectrum and make the blue asymmetry look red.

Usage:
    python self_steepening.py            # writes both standalone figures
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c as c_light

__all__ = [
    "shock_rhs", "propagate_shock", "shock_distance_theory",
    "fig_steepening_maps", "fig_shock_scaling", "fig_steepening_from_run",
]


# ================================================================================
#  Core: 1D propagation with SPM + shock only
# ================================================================================
def _dt_spectral(f, w):
    """d/dt via FFT. With numpy's fft/ifft pair the multiplier is +i*w."""
    return np.fft.ifft(1j * w * np.fft.fft(f))


def shock_rhs(E, w, K0, n2, n0, shock=True):
    """RHS of dE/dz = i K0 n2 T_hat[I E], with I = 0.5 eps0 n0 c |E|^2.

    Working in units where the field is already scaled so that I = |E|^2 keeps
    the algebra of Eq. (1) visible; the caller supplies E in sqrt(W/m^2).
    """
    I = np.abs(E) ** 2
    IE = I * E
    out = 1j * K0 * n2 * IE
    if shock:
        # T_hat second half: (i/w0) d/dt, and K0/w0 = 1/c  ->  -(n2/c) d(I E)/dt
        out = out - (n2 / c_light) * _dt_spectral(IE, w)
    return out


def propagate_shock(t_s, E0, z_max, nz, lam0=800e-9, n0=1.4533, n2=3.54e-20,
                    shock=True, nsave=None):
    """RK4 integration in z of the SPM+shock model.

    Returns (z, E_saved) with E_saved of shape (nsave, len(t_s)).
    """
    dt = t_s[1] - t_s[0]
    N = len(t_s)
    w = 2 * np.pi * np.fft.fftfreq(N, dt)
    K0 = 2 * np.pi / lam0                      # vacuum wavenumber w0/c
    dz = z_max / nz
    nsave = nsave or min(nz, 240)
    stride = max(1, nz // nsave)

    E = E0.astype(np.complex128).copy()
    zs, Es = [0.0], [E.copy()]
    for i in range(nz):
        k1 = shock_rhs(E, w, K0, n2, n0, shock)
        k2 = shock_rhs(E + 0.5 * dz * k1, w, K0, n2, n0, shock)
        k3 = shock_rhs(E + 0.5 * dz * k2, w, K0, n2, n0, shock)
        k4 = shock_rhs(E + dz * k3, w, K0, n2, n0, shock)
        E = E + (dz / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if (i + 1) % stride == 0:
            zs.append((i + 1) * dz)
            Es.append(E.copy())
    return np.array(zs), np.array(Es)


def shock_distance_theory(t_p_s, I_p, n2=3.54e-20):
    """Eq. (2): z_shock = 0.39 c t_p / (n2 I_p) for a Gaussian input.

    The coefficient is 1/(3 sqrt(2) e^{-1/2}) = 0.3888..., obtained from the
    steepest negative slope of a Gaussian, min dI/dt = -sqrt(2) e^{-1/2} I_p/t_p,
    inserted into the Burgers shock condition z = -1/(3 s min dI/dt).
    """
    coeff = 1.0 / (3.0 * np.sqrt(2.0) * np.exp(-0.5))
    return coeff * c_light * t_p_s / (n2 * I_p)


def _spectrum(E):
    """Physical-detuning spectrum. See the Fourier-convention note in the
    module docstring: the forward transform of exp(-i W t) is numpy's ifft."""
    return np.abs(np.fft.fftshift(np.fft.ifft(E, axis=-1), axes=-1)) ** 2


def _detuning_axis(t_s):
    dt = t_s[1] - t_s[0]
    return np.fft.fftshift(2 * np.pi * np.fft.fftfreq(len(t_s), dt))


# ================================================================================
#  Figure 1 -- 2D maps, shock on vs off
# ================================================================================
def fig_steepening_maps(t_p_fs=30.0, I_p_Wcm2=5e13, nt=8192, t_win_fs=400.0,
                        z_frac=0.85, nz=4000, lam0=800e-9, n0=1.4533,
                        n2=3.54e-20, save=None, show=True):
    """2x2 map: intensity I(t,z) and spectrum S(W,z), with and without the
    shock term, over a distance z_frac * z_shock.

    Shock OFF is pure SPM: the temporal profile is rigorously unchanged (SPM is
    a pure phase) and the spectrum is symmetric. Shock ON steepens the trailing
    edge and pushes the spectrum to the blue. Putting them side by side is what
    makes the asymmetry unambiguous -- it cannot be blamed on the input pulse.
    """
    t_p = t_p_fs * 1e-15
    I_p = I_p_Wcm2 * 1e4                                  # W/cm^2 -> W/m^2
    t_s = np.linspace(-t_win_fs, t_win_fs, nt) * 1e-15
    E0 = np.sqrt(I_p) * np.exp(-(t_s / t_p) ** 2 / 2.0)   # I = |E|^2 Gaussian
    z_sh = shock_distance_theory(t_p, I_p, n2)
    z_max = z_frac * z_sh

    out = {}
    for tag, sk in (("shock", True), ("noshock", False)):
        out[tag] = propagate_shock(t_s, E0, z_max, nz, lam0, n0, n2, shock=sk)

    W = _detuning_axis(t_s)
    t_fs = t_s * 1e15
    z_um = out["shock"][0] * 1e6

    # Spectral window measured from the shock-ON output, not guessed. Sizing it
    # on 1/t_p (or even on the symmetric SPM estimate K0 n2 z max|dI/dt|) cuts
    # off the blue tail: the whole point of the shock is that the blue edge runs
    # several times further than the red one, so a symmetric window built for
    # the shock-OFF case hides the effect it is meant to show.
    _S_on = _spectrum(out["shock"][1][-1])
    _S_on = _S_on / _S_on.max()
    _lev = 1e-4
    W_blue = W[(W > 0) & (_S_on > _lev)].max()
    W_red = W[(W < 0) & (_S_on > _lev)].min()
    Wlim = (1.1 * W_red, 1.1 * W_blue)

    fig, axes = plt.subplots(3, 2, figsize=(11.5, 11.0))
    finals = {}
    for col, (tag, title) in enumerate((("noshock", "SPM only (shock term OFF)"),
                                        ("shock", "SPM + shock term ON"))):
        _, Es = out[tag]
        I_map = np.abs(Es) ** 2 / I_p
        S_map = _spectrum(Es)
        S_map /= S_map.max()
        finals[tag] = (I_map, S_map)

        ax = axes[0, col]
        m = np.abs(t_fs) < 4 * t_p_fs
        im = ax.pcolormesh(t_fs[m], z_um, I_map[:, m], cmap="inferno",
                           shading="auto", vmin=0, vmax=1.05)
        ax.set_title(title)
        ax.set_xlabel("t (fs)")
        ax.set_ylabel("z (µm)")
        fig.colorbar(im, ax=ax, label="I / I$_p$")

        ax = axes[1, col]
        mw = (W > Wlim[0]) & (W < Wlim[1])
        im = ax.pcolormesh(W[mw] * 1e-15, z_um,
                           10 * np.log10(np.clip(S_map[:, mw], 1e-6, None)),
                           cmap="viridis", shading="auto", vmin=-45, vmax=0)
        ax.axvline(0.0, color="w", lw=0.9, ls=":")
        ax.set_xlabel(r"detuning $\Omega$ (rad/fs)   [$\Omega>0$ = blue]")
        ax.set_ylabel("z (µm)")
        fig.colorbar(im, ax=ax, label="spectrum (dB)")

    # ---- row 3: lineouts at the output plane, the two cases superposed
    ax = axes[2, 0]
    m = np.abs(t_fs) < 4 * t_p_fs
    ax.plot(t_fs[m], finals["noshock"][0][0][m], color="0.6", lw=1.2,
            label="input")
    ax.plot(t_fs[m], finals["noshock"][0][-1][m], "b-", lw=1.6,
            label="output, shock OFF")
    ax.plot(t_fs[m], finals["shock"][0][-1][m], "r-", lw=1.8,
            label="output, shock ON")
    ax.set_xlabel("t (fs)"); ax.set_ylabel("I / I$_p$")
    ax.set_title("Temporal profile at $z$ = %.0f µm" % z_um[-1])
    ax.legend(fontsize=8); ax.grid(alpha=0.2)

    ax = axes[2, 1]
    mw = (W > Wlim[0]) & (W < Wlim[1])
    # The right observable is the spectral EXTENT on each side, not the energy.
    # The blue tail is generated by the narrow steepened edge, so it is broad
    # but weak: an energy ratio is dominated by the central SPM peaks and comes
    # out slightly red-heavy even when the blue edge runs 4x further. Quoting
    # the energy ratio here would contradict the physics the figure shows.
    ext = {}
    for tag, cstyle, lab in (("noshock", "b-", "shock OFF"),
                             ("shock", "r-", "shock ON")):
        S = finals[tag][1][-1]
        Sn = S / S.max()
        ax.semilogy(W[mw] * 1e-15, np.clip(Sn[mw], 1e-6, None),
                    cstyle, lw=1.5, label=lab)
        b = W[(W > 0) & (Sn > 1e-3)].max()
        r = W[(W < 0) & (Sn > 1e-3)].min()
        ext[tag] = abs(b / r)
        ax.plot([r * 1e-15, b * 1e-15], [1e-3, 1e-3], cstyle[0] + "o",
                ms=5, mfc="none")
    ax.axhline(1e-3, color="0.6", lw=0.7, ls=":")
    ax.axvline(0.0, color="k", lw=0.8, ls=":")
    ax.set_ylim(1e-5, 2)
    ax.set_xlabel(r"detuning $\Omega$ (rad/fs)")
    ax.set_ylabel("normalized spectrum")
    ax.set_title("Output spectrum   (blue/red extent at $10^{-3}$: "
                 f"OFF {ext['noshock']:.2f}, ON {ext['shock']:.2f})",
                 fontsize=9.5)
    ax.legend(fontsize=8); ax.grid(alpha=0.2)

    fig.suptitle(f"Self-steepening: fused silica, {lam0*1e9:.0f} nm, "
                 f"$t_p$={t_p_fs:g} fs, $I_p$={I_p_Wcm2:.0e} W/cm², "
                 f"$z_{{shock}}$={z_sh*1e6:.0f} µm", fontsize=11)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=170)
    if show:
        plt.show()
    return fig, out


# ================================================================================
#  Figure 2 -- Rothenberg-style scan over the input pulse
# ================================================================================
#: Slope-growth threshold used to locate the shock numerically. Burgers gives
#: |dI/dt|(z) = |dI/dt|(0) / (1 - z/z_shock), so a threshold of G times the
#: initial slope is crossed at exactly z = (1 - 1/G) z_shock. G = 3 -> 2/3.
SLOPE_THRESHOLD = 3.0
THRESHOLD_FRACTION = 1.0 - 1.0 / SLOPE_THRESHOLD      # = 2/3 for G = 3


def max_slope_vs_z(Es, dt):
    """max_t |dI/dt| at each saved plane."""
    I = np.abs(Es) ** 2
    return np.max(np.abs(np.gradient(I, dt, axis=-1)), axis=-1)


def _measured_shock_distance(z, Es, dt, threshold=SLOPE_THRESHOLD):
    """z at which max|dI/dt| reaches `threshold` times its initial value.

    A finite threshold is used rather than the true divergence, because on a
    finite grid the slope saturates once the shock is unresolved. This is not a
    fudge: differentiating the Burgers equation along a characteristic gives
    dJ/dz = -3 s J^2 with J = dI/dt, hence

        |J|(z) = |J|(0) / (1 - z/z_shock)

    so the crossing of any fixed multiple G of the initial slope happens at the
    exactly known fraction (1 - 1/G) of z_shock. Comparing the measured
    crossing to (1 - 1/G) z_shock is therefore a parameter-free test of both
    the Burgers structure and the coefficient in Eq. (2).
    """
    slope = max_slope_vs_z(Es, dt)
    target = threshold * slope[0]
    idx = np.argmax(slope >= target)
    if slope[idx] < target:
        return np.nan
    if idx == 0:
        return z[0]
    # linear interpolation between the two bracketing saved planes
    s0, s1 = slope[idx - 1], slope[idx]
    return z[idx - 1] + (z[idx] - z[idx - 1]) * (target - s0) / (s1 - s0)


def fig_shock_scaling(t_p_list_fs=(15, 20, 30, 40, 60, 80), I_p_Wcm2=5e13,
                      nt=4096, nz=4000, lam0=800e-9, n0=1.4533, n2=3.54e-20,
                      save=None, show=True):
    """Scan the input pulse duration and compare the measured shock distance
    with Eq. (2). Rothenberg's study varies the input pulse; the point here is
    that Eq. (2) predicts a strict proportionality z_shock ∝ t_p at fixed peak
    intensity, which the integration either reproduces or does not.
    """
    I_p = I_p_Wcm2 * 1e4
    meas, theo, growth = [], [], []
    for t_p_fs in t_p_list_fs:
        t_p = t_p_fs * 1e-15
        t_win = 8.0 * t_p_fs
        t_s = np.linspace(-t_win, t_win, nt) * 1e-15
        E0 = np.sqrt(I_p) * np.exp(-(t_s / t_p) ** 2 / 2.0)
        z_th = shock_distance_theory(t_p, I_p, n2)
        z, Es = propagate_shock(t_s, E0, 0.9 * z_th, nz, lam0, n0, n2,
                                shock=True, nsave=400)
        dt = t_s[1] - t_s[0]
        meas.append(_measured_shock_distance(z, Es, dt))
        theo.append(z_th)
        sl = max_slope_vs_z(Es, dt)
        growth.append((z / z_th, sl / sl[0]))

    meas = np.array(meas) * 1e6
    theo = np.array(theo) * 1e6
    pred = THRESHOLD_FRACTION * theo          # where the threshold must be hit
    tp = np.array(t_p_list_fs, float)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))

    ax = axes[0]
    ax.plot(tp, theo, "k-", lw=1.8, label=r"Eq. (2): $z_{sh}=0.39\,c\,t_p/(n_2 I_p)$")
    ax.plot(tp, pred, "--", color="0.45", lw=1.6,
            label=rf"$({THRESHOLD_FRACTION:.3f})\,z_{{sh}}$ = predicted crossing")
    ax.plot(tp, meas, "o", ms=7, mfc="none", mec="crimson", mew=1.7,
            label=rf"measured (max$|\partial_t I|$ = {SLOPE_THRESHOLD:g}$\times$ initial)")
    ax.set_xlabel("input pulse duration $t_p$ (fs)")
    ax.set_ylabel("distance (µm)")
    ax.set_title("Shock distance vs input pulse")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(alpha=0.2)

    ax = axes[1]
    ratio = meas / theo
    ax.plot(tp, ratio, "s", color="tab:blue", ms=7, label="measured / Eq. (2)")
    ax.axhline(THRESHOLD_FRACTION, color="crimson", ls="--", lw=1.4,
               label=rf"Burgers prediction $1-1/{SLOPE_THRESHOLD:g}$ = {THRESHOLD_FRACTION:.4f}")
    dev = np.nanmax(np.abs(ratio - THRESHOLD_FRACTION))
    ax.set_ylim(THRESHOLD_FRACTION - 0.02, THRESHOLD_FRACTION + 0.02)
    ax.set_xlabel("input pulse duration $t_p$ (fs)")
    ax.set_ylabel("measured / Eq. (2)")
    ax.set_title(f"max deviation = {dev:.1e}")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.2)

    ax = axes[2]
    for t_p_fs, (zz, gg) in zip(t_p_list_fs, growth):
        ax.plot(zz, gg, lw=1.2, label=f"$t_p$={t_p_fs:g} fs")
    zz = np.linspace(0, 0.9, 200)
    ax.plot(zz, 1.0 / (1.0 - zz), "k--", lw=2.0, label=r"$1/(1-z/z_{sh})$")
    ax.set_xlabel(r"$z / z_{sh}$")
    ax.set_ylabel(r"max$|\partial_t I|$ / initial")
    ax.set_title("Slope growth collapses onto Burgers law")
    ax.set_ylim(0.5, 11)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.2)

    fig.suptitle(f"Burgers shock scaling, fused silica, $I_p$={I_p_Wcm2:.0e} W/cm²",
                 fontsize=11)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=170)
    if show:
        plt.show()
    return fig, (tp, meas, theo)


# ================================================================================
#  Figure 3 -- from real solver output (needs a GPU run, two scenarios)
# ================================================================================
def fig_steepening_from_run(res_on, res_off=None, r_index=0, save=None,
                            show=True, lam0=800e-9):
    """On-axis I(t,z) map and spectrum map from a real result.npz.

    Pass the `full` scenario as `res_on` and the `no_self_steepening` scenario
    of the ablation study as `res_off` to get the side-by-side comparison; with
    `res_off=None` only the first column is drawn.

    Requires the run to carry the (z,r,t) cube, i.e. `rho_t_stride>0`, so that
    `I_rzt` is present. `I_onaxis_t` is preferred when available since it is
    stored at full temporal resolution.
    """
    def _onaxis(res):
        I_full = res.get("I_onaxis_t")
        t_full = res.get("t_full_fs")
        if I_full is not None and t_full is not None:
            return np.asarray(I_full, float), np.asarray(t_full, float)
        I_rzt = res.get("I_rzt")
        if I_rzt is None:
            raise RuntimeError(
                "neither I_onaxis_t nor I_rzt in the npz: re-run with "
                "rho_t_stride>0 so the (z,r,t) cube is saved.")
        return (np.asarray(I_rzt, float)[:, r_index, :],
                np.asarray(res["t_sub_fs"], float))

    cols = [("full model (shock ON)", res_on)]
    if res_off is not None:
        cols.append(("shock term OFF", res_off))

    fig, axes = plt.subplots(2, len(cols), figsize=(5.6 * len(cols), 7.4),
                             squeeze=False)
    for col, (title, res) in enumerate(cols):
        I_zt, t_fs = _onaxis(res)
        z_um = np.asarray(res["z"], float) * 1e6
        t_s = t_fs * 1e-15
        W = _detuning_axis(t_s)

        # A real solver stores intensity, not the complex field, so the shock
        # signature that survives is the temporal asymmetry. The "spectrum"
        # below is that of sqrt(I) with flat phase: it shows the pulse-shape
        # asymmetry, NOT the true optical spectrum (which needs the phase).
        S = _spectrum(np.sqrt(np.clip(I_zt, 0, None)))
        S = S / S.max()

        ax = axes[0, col]
        im = ax.pcolormesh(t_fs, z_um, I_zt / I_zt.max(), cmap="inferno",
                           shading="auto")
        ax.set_title(title)
        ax.set_xlabel("t (fs)"); ax.set_ylabel("z (µm)")
        fig.colorbar(im, ax=ax, label="I / I$_{max}$")

        ax = axes[1, col]
        im = ax.pcolormesh(W * 1e-15, z_um,
                           10 * np.log10(np.clip(S, 1e-6, None)),
                           cmap="viridis", shading="auto", vmin=-50, vmax=0)
        ax.axvline(0.0, color="w", lw=0.8, ls=":")
        ax.set_xlabel(r"detuning $\Omega$ (rad/fs)")
        ax.set_ylabel("z (µm)")
        fig.colorbar(im, ax=ax, label="envelope spectrum (dB)")

    fig.suptitle("Self-steepening in the full solver, on-axis", fontsize=11)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=170)
    if show:
        plt.show()
    return fig


# ================================================================================
def main():
    import matplotlib
    matplotlib.use("Agg")
    fig_steepening_maps(save="self_steepening_maps.png", show=False)
    print("-> self_steepening_maps.png")
    fig_shock_scaling(save="self_steepening_scaling.png", show=False)
    print("-> self_steepening_scaling.png")


if __name__ == "__main__":
    main()
