#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figures_article.py — Post-traitement de result.npz (filament_sim.run) SANS relancer la simu.

Analogues des figures de Mao et al., Appl. Phys. A 79, 1695 (2004), avec les
paramètres réels de la simulation (pompe 1030 nm) et de l'expérience (sonde 490 nm) :

  Fig. 2  — ρ_e(t) au foyer, trois variantes du modèle d'ionisation, réintégrées
            en 0D à partir de I(t) sauvegardé (I_rzt) — coût ≈ ms :
              (i)   MPI seul :            dρ/dt = W_K(I)·(1 − ρ/ρmax)
              (ii)  + avalanche :         + (σ/Ui)·I·ρ·(1 − ρ/ρmax)
              (iii) + piégeage & STE :    − ρ/τ_r ; dρ_s/dt = ρ/τ_r − (W_K^s + (σ/Us)Iρ)·ρ_s/ρmax
            (iii) = modèle exact du noyau CUDA → doit coïncider avec rho_rzt sauvegardé
            (validation superposée en points).

  Fig. 13 — ρ_e on-axis vs z, mêmes trois variantes que Fig. 2 mais réintégrées
            à CHAQUE z (à partir de I_onaxis_t) au lieu d'un z fixé -- coût
            ≈ Nz x quelques ms. Validation superposée (rho_rz, on-axis).

  Fig. 10 — ΔΦ_sonde(τ) à 490 nm, à z fixé, géométrie latérale (Nomarski) :
              Δn(r,τ) = 2 n₂ I_pompe(r,τ)                                (XPM, facteur 2)
                        − ρ_e(r,τ)/(2 n₀' ρ_c')                          (Drude)
                        + [ω'²/(ω_tr² − ω'²)]·ρ_s(r,τ)/(2 n₀' ρ_c')      (Lorentz STE)
              δφ(y,τ) = (2π/λ') ∫ Δn(√(s²+y²), τ) ds                     (projection d'Abel)
            avec ρ_c'(490 nm) = ε₀ m ω'²/e² = 4,64e21 cm⁻³, n₀' = 1,4629.
            + prolongation analytique après la boîte temporelle :
              ρ_e → ρ_e·e^(−Δt/τ_r),  ρ_s → ρ_s + ρ_e·(1 − e^(−Δt/τ_r)),  I → 0
            + convolution par la durée de la sonde (FWHM ≈ FWHM_pompe/10 ≈ 26 fs).

Usage :
    python figures_article.py runs_ablation/full/result.npz
ou dans le notebook :
    from figures_article import load_res, fig2_populations, fig10_dephasage
    res = load_res(NPZ_PATH)      # ou directement le dict retourné par run()
    fig2_populations(res)                              # z* = argmax I
    fig10_dephasage(res, band_um=10.0)                 # moyenne |y|<10 µm comme l'expérience
"""
from __future__ import annotations
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, epsilon_0, m_e
from scipy.interpolate import PchipInterpolator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from keldysh import (n_sellmeier, KeldyshSiO2, keldysh_multiphoton,  # noqa: E402
                     keldysh_tunnel)          # shared with filament_sim.py

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

# ================================================================================
#  Paramètres — doivent refléter l'appel run() du notebook
# ================================================================================
@dataclass
class Params:
    wavelength: float = 1030e-9
    Ui_eV: float = 9.0
    Us_eV: float = 6.0            # potentiel de ré-ionisation STE (canal Keldysh Us)
    meff_rel: float = 0.64        # masse Keldysh
    tau_c: float = 1.7e-15
    tau_r: float = 330e-15
    rho_max: float = 2.1e22       # cm^-3
    n2: float = 3.54e-20          # m²/W — défaut de run() (le notebook ne le surcharge pas)
    lambda_probe: float = 490e-9
    # Résonance de l'oscillateur Lorentz du STE. Mao et al., Appl. Phys. A 79,
    # 1695 (2004) -- l'article même dont ce module reproduit les figures --
    # écrit Δn = N_STE e²/(2 n₀ m ε₀ (ω_tr² − ω²)) et précise que ω_tr est
    # « the resonance frequency of the STE's first excited level (~4.2 eV in
    # SiO2) ». Les 5.2 eV souvent cités sont le SOMMET DE LA BANDE
    # D'ABSORPTION du STE (Mao Fig. 12, « rise time of the 5.2 eV absorption
    # band »), mesuré en absorption transitoire -- ce n'est pas la résonance
    # à utiliser pour le changement d'INDICE. À 490 nm (2.53 eV) la différence
    # n'est pas cosmétique : f_STE = ω²/(ω_tr²−ω²) passe de 0.310 à 0.570,
    # soit un facteur 1.84 sur la contribution STE au déphasage.
    E_tr_eV: float = 4.2
    probe_fwhm_fs: float = 26.3   # ≈ 263/10

    def __post_init__(self):
        self.n0     = n_sellmeier(self.wavelength)
        self.omega0 = 2 * np.pi * c / self.wavelength
        self.komega = 2 * np.pi / self.wavelength * self.n0
        q_e = 1.602176634e-19
        self.Ui, self.Us = self.Ui_eV * q_e, self.Us_eV * q_e
        self.meff   = self.meff_rel * m_e
        # σ inverse-Bremsstrahlung (m_drude = m_e), même expression que build_grids() :
        self.sigmaomega = ((self.komega * q_e**2 * self.tau_c) /
                           (self.n0**2 * m_e * epsilon_0 * self.omega0 *
                            (1.0 + (self.omega0 * self.tau_c)**2))) * 1e4      # cm²
        self.beta_g = self.sigmaomega / self.Ui                                # cm²/J
        self.beta_s = self.sigmaomega / self.Us
        # sonde
        self.omega_p  = 2 * np.pi * c / self.lambda_probe
        self.n0_probe = n_sellmeier(self.lambda_probe)
        self.nc_probe = (epsilon_0 * m_e * self.omega_p**2 / q_e**2) * 1e-6    # cm^-3
        w_tr = self.E_tr_eV * q_e / 1.054571817e-34
        self.f_ste = self.omega_p**2 / (w_tr**2 - self.omega_p**2)             # facteur Lorentz
        # LUT Keldysh (PCHIP en log10 I, comme le solveur)
        Igrid = np.logspace(0.0, 17.0, 700)
        Wg = np.maximum(np.nan_to_num(KeldyshSiO2(self.wavelength, self.Ui_eV, self.meff, self.n0).rate(Igrid)), 0.0)
        Ws = np.maximum(np.nan_to_num(KeldyshSiO2(self.wavelength, self.Us_eV, self.meff, self.n0).rate(Igrid)), 0.0)
        self._lI = np.log10(Igrid)
        self._Wg = PchipInterpolator(self._lI, Wg)
        self._Ws = PchipInterpolator(self._lI, Ws)
        self._Imin, self._Imax = Igrid[0], Igrid[-1]

    def W_vb(self, I_Wcm2):
        return np.maximum(self._Wg(np.log10(np.clip(I_Wcm2, self._Imin, self._Imax))), 0.0)

    def W_ste(self, I_Wcm2):
        return np.maximum(self._Ws(np.log10(np.clip(I_Wcm2, self._Imin, self._Imax))), 0.0)

# ================================================================================
#  Accès npz robuste
# ================================================================================
def load_res(path):
    return {k: v for k, v in np.load(path, allow_pickle=True).items()}

def _get(res, key):
    if key not in res:
        return None
    a = res[key]
    if isinstance(a, np.ndarray) and a.dtype == object and a.ndim == 0:
        a = a.item()
    return a

def _axes(res):
    z = np.asarray(_get(res, "z"), float)
    r = _get(res, "rlist")
    if r is None:                          # clé 'r' = axe miroité (−r, +r)
        rfull = np.asarray(_get(res, "r"), float)
        r = rfull[rfull.size // 2:]
    r = np.asarray(r, float)
    t = np.asarray(_get(res, "t_sub_fs"), float)
    return z, r, t

def _r_cube(res):
    """Grille radiale du cube (z,r,t).

    Depuis l'ajout de `rho_r_stride` (le cube domine la taille du result.npz),
    ce cube n'est plus forcement echantillonne sur `rlist` : le solveur ecrit
    alors `r_sub`. Tout ce qui lit rho_rzt/I_rzt radialement DOIT passer par
    ici, sinon les longueurs ne concordent plus (np.interp leve
    "fp and xp are not of the same length")."""
    rs = _get(res, "r_sub")
    if rs is not None:
        return np.asarray(rs, float)
    return _axes(res)[1]

def _iz_of(res, z, z_um):
    if z_um is None:
        return int(np.argmax(np.asarray(_get(res, "Imax_z"), float)))
    return int(np.argmin(np.abs(z * 1e6 - z_um)))

# ================================================================================
#  Intégrateur 0D — miroir exact du noyau CUDA (pas exponentiel exact)
# ================================================================================
def _phi1(x):
    return np.expm1(x) / x if abs(x) > 1e-6 else 1.0 + x * (0.5 + x * (1.0/6.0 + x/24.0))

def _step(x, S, L, dt):
    xn = np.exp(L * dt) * x + S * dt * _phi1(L * dt)
    return xn if (np.isfinite(xn) and xn >= 0.0) else 0.0

def integre_0d(t_fs, I_Wcm2, prm: Params, avalanche=True, piegeage=True, ste=True):
    """dρ_e/dt, dρ_s/dt sur I(t) donné. Retourne (ρ_e, ρ_s) en cm^-3."""
    t = np.asarray(t_fs, float) * 1e-15
    I = np.clip(np.asarray(I_Wcm2, float), 0.0, None)
    W  = prm.W_vb(I)
    Ws = prm.W_ste(I) if ste else np.zeros_like(I)
    na, inv_tr = prm.rho_max, 1.0 / prm.tau_r
    ne = ns = 0.0
    NE, NS = np.zeros_like(I), np.zeros_like(I)
    for k in range(len(t) - 1):
        dt   = t[k+1] - t[k]
        Iavg = 0.5 * (I[k] + I[k+1])
        Wavg = 0.5 * (W[k] + W[k+1])
        Wsav = 0.5 * (Ws[k] + Ws[k+1])
        depl = min(max(1.0 - ne / na, 0.0), 1.0)
        Se = Wavg * depl + ((Wsav + prm.beta_s * Iavg * ne) * (ns / na) if ste else 0.0)
        Le = (prm.beta_g * Iavg * depl if avalanche else 0.0) - (inv_tr if piegeage else 0.0)
        Ss = inv_tr * ne if (ste and piegeage) else 0.0
        Ls = -(Wsav + prm.beta_s * Iavg * ne) / na if ste else 0.0
        ne_n = _step(ne, Se, Le, dt)
        ns_n = _step(ns, Ss, Ls, dt) if ste else 0.0
        tot = ne_n + ns_n
        if tot > na:
            ne_n *= na / tot; ns_n *= na / tot
        ne, ns = ne_n, ns_n
        NE[k+1], NS[k+1] = ne, ns
    return NE, NS

# ================================================================================
#  FIG. 2 — populations électroniques (3 variantes) à z*
# ================================================================================
def fig2_populations(res, prm: Params | None = None, z_um=None, dt_fs=0.25,
                     save=None, show=True):
    prm = prm or Params()
    z, r, t = _axes(res)
    iz = _iz_of(res, z, z_um)

    # Full-time-resolution on-axis trace (filament_sim.py >= this version)
    # is what the CUDA kernel actually integrated over -- prefer it. Falling
    # back to the rho_t_stride-subsampled I_rzt/rho_rzt (older npz) means a
    # narrow intensity spike between two saved samples can go missing, which
    # -- since the multiphoton rate scales roughly as I^K (K ~ 8-9 photons
    # here) -- can make this 0D reintegration look many orders of magnitude
    # below the real rho_e for no physical reason.
    I_onaxis_full = _get(res, "I_onaxis_t")
    t_full_fs     = _get(res, "t_full_fs")
    rho_onaxis_full = _get(res, "rho_onaxis_t")
    if I_onaxis_full is not None and t_full_fs is not None:
        t_native = np.asarray(t_full_fs, np.float64)
        I_t      = np.asarray(I_onaxis_full, np.float64)[iz, :]
        rho_sim, t_sim_axis = np.asarray(rho_onaxis_full, np.float64)[iz, :], t_native
    else:
        I_rzt = _get(res, "I_rzt")
        if I_rzt is None:
            raise RuntimeError("I_rzt/I_onaxis_t absent du npz : relancer avec rho_t_stride>0 pour la Fig. 2.")
        t_native = t
        I_t      = np.asarray(I_rzt, np.float64)[iz, 0, :]          # axe (r ≈ 0)
        rho_sim, t_sim_axis = np.asarray(_get(res, "rho_rzt"), np.float64)[iz, 0, :], t
        print("[fig2] I_onaxis_t/rho_onaxis_t absents (npz généré par une version antérieure) : "
              "utilisation de I_rzt/rho_rzt sous-échantillonnés (rho_t_stride) -- "
              "un pic d'intensité étroit peut être manqué, voir README.")

    tf  = np.arange(t_native[0], t_native[-1] + dt_fs, dt_fs)      # grille fine
    If  = np.clip(PchipInterpolator(t_native, I_t)(tf), 0.0, None)

    ne_mpi, _  = integre_0d(tf, If, prm, avalanche=False, piegeage=False, ste=False)
    ne_av,  _  = integre_0d(tf, If, prm, avalanche=True,  piegeage=False, ste=False)
    ne_fl, ns  = integre_0d(tf, If, prm, avalanche=True,  piegeage=True,  ste=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.semilogy(tf, ne_mpi, "k-",  lw=1.4, label="MPI only")
    ax.semilogy(tf, ne_av,  "k-",  lw=2.2, label="MPI + avalanche")
    ax.semilogy(tf, ne_fl,  "k--", lw=2.2, label="MPI + avalanche + trapping (STE)")
    ax.semilogy(tf, ns,     color="tab:green", lw=1.4, label=r"$\rho_s$ (STE)")
    m = rho_sim > 0
    every = max(1, np.count_nonzero(m) // 50)
    ax.semilogy(t_sim_axis[m], rho_sim[m], "-", color="crimson", lw=1.3, alpha=0.9,
                marker="o", ms=5.0, mfc="none", mec="crimson", mew=1.2, markevery=every,
                label=r"$\rho_e$ CUDA kernel (npz)")
    ax2 = ax.twinx()
    ax2.plot(tf, If / If.max(), "r:", lw=1.2)
    ax2.set_ylabel("I(t) / I$_{max}$", color="r"); ax2.set_ylim(0, 1.9)
    ax2.tick_params(axis="y", colors="r")
    top = max(ne_av.max(), ne_fl.max(), prm.rho_max)
    ax.set_ylim(1e14, 3 * top)
    ax.set_xlabel("t (fs)"); ax.set_ylabel(r"electron density (cm$^{-3}$)")
    ax.set_title(f"Populations at z = {z[iz]*1e6:+.0f} µm  "
                 f"(I$_{{max}}$ = {I_t.max():.2e} W/cm²)")
    ax.axhline(prm.rho_max, color="gray", lw=0.8, ls=":")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    if save: fig.savefig(save, dpi=180)
    if show: plt.show()
    return fig

# ================================================================================
#  FIG. 13 — densité électronique on-axis VS Z, trois variantes du modèle
# ================================================================================
def fig13_electron_density_vs_z(res, prm: Params | None = None, dt_fs=0.5,
                                save=None, show=True, z_shift_um=0.0,
                                xlim=(0, 150), ylim=(1e18, 1e21)):
    """
    Analogue de la Fig. 13 de Couairon 2005 : densité électronique on-axis en
    fonction de la distance de propagation z, pour trois variantes du modèle
    d'ionisation -- PI seul (dash-dot), PI + recombinaison sans avalanche
    (pointillé), PI + avalanche + recombinaison (continu) -- réintégrées en
    0D à CHAQUE z à partir de I_onaxis_t (voir fig2_populations pour le même
    principe à z fixé). Le rho_e réel du noyau CUDA (rho_rz, max sur le
    temps, on-axis) est superposé en points de validation.

    z_shift_um : décale l'axe z affiché (ex. -begin_um pour repasser au
    repère de l'article, entrée=0, si le solveur travaille avec foyer=0).
    xlim/ylim : bornes par défaut = celles lues directement sur la Fig. 13
    du papier (z in [0,150] µm, ρ in [1e18,1e21] cm⁻³) ; passer None pour
    revenir à l'auto-échelle.
    """
    prm = prm or Params()
    z, r, t = _axes(res)
    I_onaxis = _get(res, "I_onaxis_t")
    t_full   = _get(res, "t_full_fs")
    if I_onaxis is None or t_full is None:
        raise RuntimeError("I_onaxis_t/t_full_fs absent du npz : relance filament_sim.py "
                           "(version avec l'enregistrement on-axis pleine résolution).")

    Nz = len(z)
    ne_pi   = np.zeros(Nz)
    ne_rec  = np.zeros(Nz)
    ne_full = np.zeros(Nz)
    t_full = np.asarray(t_full, np.float64)
    for iz in range(Nz):
        I_t = np.asarray(I_onaxis[iz, :], np.float64)
        tf  = np.arange(t_full[0], t_full[-1] + dt_fs, dt_fs)
        If  = np.clip(PchipInterpolator(t_full, I_t)(tf), 0.0, None)
        ne, _   = integre_0d(tf, If, prm, avalanche=False, piegeage=False, ste=False)
        ne_r, _ = integre_0d(tf, If, prm, avalanche=False, piegeage=True,  ste=False)
        ne_f, _ = integre_0d(tf, If, prm, avalanche=True,  piegeage=True,  ste=False)
        ne_pi[iz], ne_rec[iz], ne_full[iz] = ne.max(), ne_r.max(), ne_f.max()

    rho_rz = _get(res, "rho_rz")
    r_mirror = _get(res, "r")
    rho_sim_onaxis = None
    if rho_rz is not None and r_mirror is not None:
        # rho_rz est mirroré (-R...+R, comme fluence_rz) -- l'index on-axis
        # doit être calculé sur le "r" mirroré, PAS sur le rlist non-mirroré
        # renvoyé par _axes() (qui vaut 0 en indice 0, valide pour rho_rzt/
        # I_rzt/rho_onaxis_t mais pas pour rho_rz/fluence_rz).
        ir = int(np.argmin(np.abs(np.asarray(r_mirror, np.float64))))
        rho_sim_onaxis = np.asarray(rho_rz, np.float64)[:, ir]

    z_um = z * 1e6 + z_shift_um
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.semilogy(z_um, np.clip(ne_pi,   1e14, None), "k-.", lw=1.6, label="PI only")
    ax.semilogy(z_um, np.clip(ne_rec,  1e14, None), "k:",  lw=1.8, label="PI + recombination")
    ax.semilogy(z_um, np.clip(ne_full, 1e14, None), "k-",  lw=2.0, label="PI + avalanche + recombination")
    if rho_sim_onaxis is not None:
        m = rho_sim_onaxis > 0
        every = max(1, np.count_nonzero(m) // 50)
        ax.semilogy(z_um[m], rho_sim_onaxis[m], "-", color="crimson", lw=1.3, alpha=0.9,
                    marker="o", ms=5.0, mfc="none", mec="crimson", mew=1.2, markevery=every,
                    label=r"$\rho_e$ CUDA kernel (validation)")
    if xlim is not None: ax.set_xlim(*xlim)
    if ylim is not None: ax.set_ylim(*ylim)
    ax.set_xlabel("z (µm)")
    ax.set_ylabel(r"on-axis electron density (cm$^{-3}$)")
    ax.set_title("Fig. 13 style — Electron density vs propagation distance")
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save: fig.savefig(save, dpi=180)
    if show: plt.show()
    return fig

# ================================================================================
#  FIG. 10 — ΔΦ_sonde(τ) à 490 nm, projection latérale
# ================================================================================
def _proj_chord(f_r, r, y=0.0, ns=600):
    """P(y) = 2∫ f(√(s²+y²)) ds — projection d'Abel via s = √(r²−y²)."""
    smax2 = r[-1]**2 - y * y
    if smax2 <= 0.0:
        return 0.0
    s  = np.linspace(0.0, np.sqrt(smax2), ns)
    rr = np.sqrt(s * s + y * y)
    return 2.0 * _trapz(np.interp(rr, r, f_r, right=0.0), s)

def fig10_dephasage(res, prm: Params | None = None, z_um=None, band_um=0.0,
                    extend_fs=900.0, save=None, show=True):
    prm = prm or Params()
    z, _, t = _axes(res)
    r = _r_cube(res)          # grille radiale DU CUBE (cf. rho_r_stride)
    iz = _iz_of(res, z, z_um)
    rho_e = np.asarray(_get(res, "rho_rzt"),   np.float64)[iz]      # (Nr_sub, Nt_sub)
    rho_s = np.asarray(_get(res, "rho_s_rzt"), np.float64)[iz]
    I_rzt = _get(res, "I_rzt")
    I = (np.asarray(I_rzt, np.float64)[iz] if I_rzt is not None
         else np.zeros_like(rho_e))
    if I_rzt is None:
        print("[fig10] I_rzt absent : pic Kerr (XPM) omis.")

    # --- prolongation analytique après la boîte : ρe décroit, ρs sature, I = 0
    dt = t[1] - t[0]
    if extend_fs > 0:
        n_ext = int(np.ceil(extend_fs / dt))
        te = t[-1] + dt * np.arange(1, n_ext + 1)
        dec = np.exp(-(te - t[-1]) * 1e-15 / prm.tau_r)[None, :]
        rho_e = np.hstack([rho_e, rho_e[:, -1:] * dec])
        rho_s = np.hstack([rho_s, rho_s[:, -1:] + rho_e[:, len(t)-1:len(t)] * (1.0 - dec)])
        I     = np.hstack([I, np.zeros((I.shape[0], n_ext))])
        t     = np.concatenate([t, te])

    # --- Δn(r, τ) par composante
    den  = 2.0 * prm.n0_probe * prm.nc_probe
    dn_K = 2.0 * prm.n2 * I * 1e4              # XPM : n2 [m²/W] × I [W/m²]
    dn_e = -rho_e / den
    dn_s = prm.f_ste * rho_s / den

    # --- projection latérale (corde y=0 ou moyenne |y|<band_um)
    ys = np.array([0.0]) if band_um <= 0 else np.linspace(0.0, band_um * 1e-6, 7)
    def proj(dn):
        out = np.empty(len(t))
        for k in range(len(t)):
            out[k] = np.mean([_proj_chord(dn[:, k], r, y) for y in ys])
        return (2 * np.pi / prm.lambda_probe) * out
    ph_K, ph_e, ph_s = proj(dn_K), proj(dn_e), proj(dn_s)

    # --- convolution par la durée de la sonde
    if prm.probe_fwhm_fs > 0:
        half = max(1, int(3 * prm.probe_fwhm_fs / dt))
        tk = dt * np.arange(-half, half + 1)
        g  = np.exp(-4 * np.log(2) * (tk / prm.probe_fwhm_fs) ** 2); g /= g.sum()
        ph_K, ph_e, ph_s = (np.convolve(p, g, mode="same") for p in (ph_K, ph_e, ph_s))
    tot = ph_K + ph_e + ph_s

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.axhline(0, color="gray", lw=0.8)
    ax.plot(t, ph_K, color="tab:orange", lw=1.2, label="Cross Kerr (2 n₂ I)")
    ax.plot(t, ph_e, color="tab:blue",   lw=1.2, label="Plasma (Drude)")
    ax.plot(t, ph_s, color="tab:green",  lw=1.2, label=f"STE (Lorentz, E$_{{tr}}$={prm.E_tr_eV:g} eV)")
    ax.plot(t, tot,  "k-", lw=2.2, label="Total")
    if extend_fs > 0:
        ax.axvspan(t[-1] - extend_fs, t[-1], color="0.93", zorder=0)
        ax.text(t[-1] - extend_fs, ax.get_ylim()[0], " analytic extension",
                fontsize=7, color="0.4", va="bottom")
    lab = "chord y=0" if band_um <= 0 else f"average |y|<{band_um:g} µm"
    ax.set_xlabel("delay τ (fs, co-moving frame at z*)")
    ax.set_ylabel(r"$\delta\varphi$ probe 490 nm (rad)")
    ax.set_title(f"ΔΦ(τ) at z = {z[iz]*1e6:+.0f} µm — {lab}, "
                 f"probe {prm.probe_fwhm_fs:.0f} fs FWHM")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    if save: fig.savefig(save, dpi=180)
    if show: plt.show()
    return fig

# ================================================================================
def main(argv=None):
    argv = argv or sys.argv[1:]
    npz = Path(argv[0]) if argv else Path("runs_ablation/full/result.npz")
    res = load_res(npz)
    prm = Params()
    f2  = fig2_populations(res, prm, save=npz.parent / "fig2_populations.png", show=False)
    f10 = fig10_dephasage(res, prm, band_um=10.0,
                          save=npz.parent / "fig10_dephasage.png", show=False)
    print(f"-> {npz.parent/'fig2_populations.png'}\n-> {npz.parent/'fig10_dephasage.png'}")
    return f2, f10

if __name__ == "__main__":
    main()

# ================================================================================
#  FIG. 2 — taux d'ionisation W_PI(I) : Keldysh général vs limites MPI / tunnel
# ================================================================================
def fig2_ionization_rate(prm: Params | None = None, I_min=1e12, I_max=6e14,
                         sigma6=9.6e-70, rho_at=2.1e22, I_marker=5e13,
                         save=None, show=True, ax=None,
                         xlim=None, ylim=(1e25, 3e37), title=None):
    """
    Fig. 2 of Couairon 2005 -- photoionization rate of fused silica (9 eV gap)
    versus laser intensity, with the two analytic limits of the same formula:

      solid        general Keldysh formula, Eqs. (7)-(8)  [what the solver uses]
      dotted       Keldysh multiphoton limit, gamma >> 1
      dash-dotted  Keldysh tunnel limit, gamma << 1
      dashed       W_MPI = sigma_6 I^6 rho_at
      vertical     maximum intensity reached numerically

    Requires no simulation: it is a direct evaluation of the ionization rate,
    so it is the most direct check that the solver's W_PI is the paper's. The
    two limits are asymptotics OF the general formula, so the solid curve
    merging into each one at the appropriate end is a parameter-free test that
    the formula is assembled correctly.
    """
    prm = prm or Params(wavelength=800e-9, Ui_eV=9.0)
    I = np.logspace(np.log10(I_min), np.log10(I_max), 1400)
    lam, Ui, meff, n0 = prm.wavelength, prm.Ui_eV, prm.meff, prm.n0

    W_gen = prm.W_vb(I)
    W_mpi = keldysh_multiphoton(I, lam, Ui, meff, n0)
    W_tun = keldysh_tunnel(I, lam, Ui, meff, n0)
    W_pow = sigma6 * I**6 * rho_at

    own = ax is None
    fig, ax = (plt.subplots(figsize=(7.4, 5.6)) if own else (ax.figure, ax))
    ax.loglog(I, np.clip(W_pow, 1e-30, None), "r--", lw=1.4,
              label=r"$W_{\mathrm{MPI}}=\sigma_6 I^6 \rho_{\mathrm{at}}$")
    ax.loglog(I, np.clip(W_mpi, 1e-30, None), "g:", lw=2.0,
              label=r"Keldysh, multiphoton limit ($\gamma \gg 1$)")
    ax.loglog(I, np.clip(W_tun, 1e-30, None), "b-.", lw=1.5,
              label=r"Keldysh, tunnel limit ($\gamma \ll 1$)")
    ax.loglog(I, np.clip(W_gen, 1e-30, None), "k-", lw=2.0,
              label="Keldysh, general formula (solver)")
    if I_marker:
        ax.axvline(I_marker, color="k", lw=1.2,
                   label=rf"$I_{{\max}}$ reached numerically $\approx$ {I_marker:.0e}")

    ax.set_xlim(*(xlim or (I_min, I_max)))
    if ylim: ax.set_ylim(*ylim)
    ax.set_xlabel(r"Laser intensity (W/cm$^2$)")
    ax.set_ylabel(r"$W_{\mathrm{PI}}$ (s$^{-1}$ cm$^{-3}$)")
    ax.set_title(title or f"Ionization rate, fused silica "
                          f"($U_i$ = {Ui:g} eV, $\\lambda$ = {lam*1e9:.0f} nm)")
    ax.text(0.06, 0.42, "Multiphoton", transform=ax.transAxes, fontsize=11)
    ax.text(0.72, 0.80, "Tunnel", transform=ax.transAxes, fontsize=11)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.grid(True, which="both", alpha=0.15)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    if own:
        fig.tight_layout()
        if save: fig.savefig(save, dpi=180)
        if show: plt.show()
    return fig


# ================================================================================
#  FIG. 10 — rho_e(t) avec / sans avalanche + intensité, à z fixé
# ================================================================================
def fig10_avalanche_vs_time(res, prm: Params | None = None, z_um=None, dt_fs=0.5,
                            save=None, show=True, z_shift_um=0.0,
                            xlim=(-300, 300), ylim=(1e13, 1e21), ylim_I=(1e8, 1e14)):
    """
    Fig. 10 de Couairon 2005 : densité électronique on-axis en fonction du
    TEMPS, à z fixé (57 µm depuis la face d'entrée), avec avalanche (continu)
    et sans avalanche (tirets), plus l'intensité du pulse (point-tirets,
    échelle de droite).

    ATTENTION : ce n'est PAS la même décomposition que
    plot_fig13_free_vs_trapped du notebook (libres vs piégés) -- la Fig. 10
    de l'article compare avalanche ON vs OFF, et l'article de 2005 n'a pas
    de canal STE (rho_s identiquement nul). Utiliser z_um dans le repère du
    solveur (foyer = 0) ; z_shift_um ne décale que le titre.
    """
    prm = prm or Params(wavelength=800e-9, Ui_eV=9.0)
    z = np.asarray(_get(res, "z"), float)
    I_onaxis = _get(res, "I_onaxis_t")
    t_full   = _get(res, "t_full_fs")
    if I_onaxis is None or t_full is None:
        raise RuntimeError("I_onaxis_t/t_full_fs absent du npz : relance filament_sim.py.")

    iz = int(np.argmax(np.asarray(_get(res, "Imax_z"), float))) if z_um is None \
        else int(np.argmin(np.abs(z * 1e6 - z_um)))

    t_full = np.asarray(t_full, np.float64)
    I_t = np.asarray(I_onaxis[iz, :], np.float64)
    tf  = np.arange(t_full[0], t_full[-1] + dt_fs, dt_fs)
    If  = np.clip(PchipInterpolator(t_full, I_t)(tf), 0.0, None)

    ne_av, _   = integre_0d(tf, If, prm, avalanche=True,  piegeage=True, ste=False)
    ne_noav, _ = integre_0d(tf, If, prm, avalanche=False, piegeage=True, ste=False)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.semilogy(tf, np.clip(ne_av,   1e-30, None), "k-",  lw=2.0, label=r"$\rho_e$ with avalanche")
    ax.semilogy(tf, np.clip(ne_noav, 1e-30, None), "k--", lw=1.6, label=r"$\rho_e$ without avalanche")
    if xlim is not None: ax.set_xlim(*xlim)
    if ylim is not None: ax.set_ylim(*ylim)
    ax.set_xlabel("time (fs)")
    ax.set_ylabel(r"$\rho$ (cm$^{-3}$)")
    ax.set_title(f"Fig. 10 — z = {z[iz]*1e6 + z_shift_um:.0f} µm")

    ax2 = ax.twinx()
    ax2.semilogy(tf, np.clip(If, 1e-30, None), "-.", color="crimson", lw=1.3,
                 label="intensity")
    if ylim_I is not None: ax2.set_ylim(*ylim_I)
    ax2.set_ylabel(r"I (W/cm$^2$)", color="crimson")
    ax2.tick_params(axis="y", colors="crimson")

    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, fontsize=8, loc="upper left")
    fig.tight_layout()
    if save: fig.savefig(save, dpi=180)
    if show: plt.show()
    return fig

# ================================================================================
#  Bulgakova, Stoian & Rosenfeld -- "Laser-induced modification of transparent
#  crystals and glasses", Figs. 11 et 12
# ================================================================================
#  Conditions (Sec. 5.2) : 800 nm, tau_L = 120 fs (FWHM) soit tau_las = 100 fs,
#  E_in = 1 uJ, w = 0.9 um, foyer geometrique a 90 um sous la surface.
#  Silice : n_lat = 6.6e22 cm-3, k" = 361 fs2/cm, n2 = 2.48e-16 cm2/W,
#  E_g0 = 9 eV, f_R = 0.18, m_r = 0.5 m_e, t_tr = 150 fs, omega0*tau_c = 3.
# ================================================================================
_BULG_LEVELS = {
    "fluence":  ([0.20, 0.80, 1.4, 1.9],   "fluence (J/cm$^2$)",              "viridis"),
    "absorbed": ([50.0, 600.0, 1200.0],    "absorbed energy (J/cm$^3$)",      "inferno"),
    "Ipeak":    ([2e12, 7e12, 3e13],       "peak intensity (W/cm$^2$)",       "magma"),
    "rho":      ([1e15, 1e17, 1e19, 3e20], "electron density (cm$^{-3}$)",    "cividis"),
}

def _bulg_panel(ax, z_um, r_um, field, levels, label, cmap="viridis", focus_um=None,
                xlim=(0, 150), rlim=(-5, 5), cbar=True, dyn_decades=4.0):
    """One Bulgakova-style panel.

    Filled contours at only 3-4 discrete levels (as the paper prints them)
    saturate: everything above the top level collapses into one flat blob and
    the structure inside it is invisible. Here the field is drawn as a
    CONTINUOUS log-scaled image spanning `dyn_decades` below its maximum, with
    the paper's levels overlaid as labelled contour lines -- so the gradient is
    readable AND the reference levels stay comparable to the publication.
    """
    from matplotlib.colors import LogNorm
    F = np.asarray(field, np.float64).T                       # (Nr, Nz)
    vmax = float(np.nanmax(F))
    if not np.isfinite(vmax) or vmax <= 0:
        ax.text(0.5, 0.5, f"{label}: no signal", ha="center", va="center",
                transform=ax.transAxes, fontsize=8)
        ax.set_xlim(*xlim); ax.set_ylim(*rlim); ax.set_ylabel(r"$r$ ($\mu$m)")
        return None
    vmin = vmax * 10.0**(-dyn_decades)
    im = ax.pcolormesh(z_um, r_um, np.clip(F, vmin, vmax),
                       norm=LogNorm(vmin=vmin, vmax=vmax), cmap=cmap,
                       shading="auto", rasterized=True)
    lv = [v for v in levels if vmin < v < vmax]
    if lv:
        cs = ax.contour(z_um, r_um, F, levels=lv, colors="white",
                        linewidths=0.8, alpha=0.9)
        ax.clabel(cs, inline=True, fontsize=6, fmt="%g")
    if focus_um is not None:
        ax.axvline(focus_um, color="white", lw=1.0, ls="--", alpha=0.8)
    if cbar:
        cb = ax.figure.colorbar(im, ax=ax, pad=0.012, fraction=0.046)
        cb.ax.tick_params(labelsize=7)
    ax.set_xlim(*xlim); ax.set_ylim(*rlim)
    ax.set_ylabel(r"$r$ ($\mu$m)")
    ax.text(0.985, 0.90, label, transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color="white",
            bbox=dict(fc="black", ec="none", alpha=0.45, pad=1.8))
    return im

def fig11_bulgakova(res, z_shift_um=90.0, focus_um=90.0, save=None, show=True,
                    xlim=(0, 150), rlim=(-5, 5)):
    """
    Fig. 11 : (a) fluence intégrée sur le pulse, (b) énergie absorbée par
    ionisation multiphotonique + bremsstrahlung inverse, (c) intensité crête
    locale, (d) densité d'électrons libres 50 fs après le maximum du pulse.
    La ligne blanche marque le foyer géométrique.

    Nécessite un run avec `rho_snapshot_t_fs=50.0` (panneau d).
    """
    z_um = np.asarray(_get(res, "z"), float) * 1e6 + z_shift_um
    r_um = np.asarray(_get(res, "r"), float) * 1e6
    panels = [
        ("fluence",  _get(res, "fluence_rz"),  "a"),
        ("absorbed", _get(res, "absorbed_rz"), "b"),
        ("Ipeak",    _get(res, "Ipeak_rz"),    "c"),
        ("rho",      _get(res, "rho_rz_at"),   "d"),
    ]
    fig, axes = plt.subplots(4, 1, figsize=(9.0, 8.4), sharex=True)
    for ax, (key, fld, tag) in zip(axes, panels):
        lv, name, cmap = _BULG_LEVELS[key]
        if fld is None:
            ax.text(0.5, 0.5, f"{name} unavailable\n(re-run with rho_snapshot_t_fs=50.0)",
                    ha="center", va="center", fontsize=8, transform=ax.transAxes)
            ax.set_ylabel(r"$r$ ($\mu$m)"); ax.set_xlim(*xlim); ax.set_ylim(*rlim)
            continue
        _bulg_panel(ax, z_um, r_um, fld, lv, f"({tag}) {name}", cmap=cmap,
                    focus_um=focus_um, xlim=xlim, rlim=rlim)
    axes[-1].set_xlabel(r"$z$ ($\mu$m)")
    fig.suptitle("Bulgakova, Stoian & Rosenfeld Fig. 11 -- fused silica, 800 nm, "
                 "120 fs, 1 $\\mu$J, $w$ = 0.9 $\\mu$m, focus at 90 $\\mu$m\n"
                 "(dashed white line: geometric focus; white contours: published levels)",
                 fontsize=10)
    fig.tight_layout()
    if save: fig.savefig(save, dpi=180)
    if show: plt.show()
    return fig

def fig12_bulgakova(res, z_shift_um=90.0, focus_um=None, save=None, show=True,
                    xlim=(0, 150), rlim=(-5, 5)):
    """
    Fig. 12 : absorption séquentielle -- énergie absorbée intégrée sur quatre
    tranches temporelles successives du pulse (maximum à t = 0), mêmes niveaux
    (50, 600, 1200 J/cm³) et mêmes conditions que la Fig. 11.

    Nécessite un run avec `absorb_time_bins_fs=(-100, -50, 0, 50, 100)`.
    """
    bins = _get(res, "absorbed_rz_bins")
    edges = _get(res, "absorb_bin_edges_fs")
    if bins is None or edges is None:
        raise RuntimeError("absorbed_rz_bins absent : relance filament_sim.py avec "
                           "absorb_time_bins_fs=(-100, -50, 0, 50, 100).")
    z_um = np.asarray(_get(res, "z"), float) * 1e6 + z_shift_um
    r_um = np.asarray(_get(res, "r"), float) * 1e6
    edges = np.asarray(edges, float)
    lv, _, cmap = _BULG_LEVELS["absorbed"]

    # Common colour scale across the four windows, otherwise each panel is
    # normalised to its own maximum and the sequence cannot be compared.
    vmax = float(np.nanmax(bins))
    nb = bins.shape[1]
    fig, axes = plt.subplots(nb, 1, figsize=(9.0, 1.95 * nb + 1.2), sharex=True)
    axes = np.atleast_1d(axes)
    for b, ax in enumerate(axes):
        fld = np.asarray(bins[:, b, :], np.float64)
        fld = np.where(fld > 0, fld, 0.0)
        fld[0, 0] = max(fld[0, 0], vmax)      # pin the shared scale
        _bulg_panel(ax, z_um, r_um, fld, lv,
                    f"{edges[b]:+.0f} fs < $t$ < {edges[b+1]:+.0f} fs", cmap=cmap,
                    focus_um=focus_um, xlim=xlim, rlim=rlim)
    axes[-1].set_xlabel(r"$z$ ($\mu$m)")
    fig.suptitle("Bulgakova, Stoian & Rosenfeld Fig. 12 -- sequential energy absorption "
                 "(J/cm$^3$), same conditions as Fig. 11\n"
                 "(pulse maximum at $t$ = 0; shared colour scale across the four windows)",
                 fontsize=10)
    fig.tight_layout()
    if save: fig.savefig(save, dpi=180)
    if show: plt.show()
    return fig


# ================================================================================
#  Export: every figure into a single multi-page PDF, English throughout
# ================================================================================
def save_all_figures(results, pdf_path="figures.pdf", prm=None,
                     paper_shift_um=75.0, bulg_shift_um=90.0,
                     z_fig10_um=-18.0, verbose=True):
    """
    Render every reproduction figure into one multi-page PDF.

    `results` is the notebook's dict {scenario_name: result}. Whatever is
    present gets plotted, whatever is missing is skipped with a note -- so this
    works after section 1 alone, or after the whole notebook.

    Recognised keys:
      couairon2005_1p1uJ  Couairon Figs. 6/7/9/12/13 + populations
      couairon2005_1uJ    Couairon Fig. 10 (avalanche on/off, 1 uJ)
      couairon2005_0.45uJ Couairon Fig. 7(a)
      bulgakova_1uJ       Bulgakova Figs. 11 and 12
    Any other key is treated as an ablation scenario and gets a summary page.

    Every page carries an English title and legend. One file, so it can be
    dropped straight into a report or shared as a single attachment.
    """
    from matplotlib.backends.backend_pdf import PdfPages
    prm = prm or Params(wavelength=800e-9, Ui_eV=9.0, rho_max=2.1e22,
                        tau_r=150e-15, n2=3.54e-20)
    pdf_path = str(pdf_path)
    made, skipped = [], []

    def _try(name, fn):
        try:
            fig = fn()
            if fig is not None:
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                made.append(name)
            else:
                skipped.append((name, "returned None"))
        except Exception as exc:
            skipped.append((name, f"{type(exc).__name__}: {exc}"))
            plt.close("all")

    with PdfPages(pdf_path) as pdf:
        # --- ionization rate: needs no run at all -------------------------
        _try("Keldysh ionization rate (Couairon Fig. 2)",
             lambda: fig2_ionization_rate(prm, show=False))

        r11 = results.get("couairon2005_1p1uJ")
        if r11 is not None:
            _try("Electron density vs z (Couairon Fig. 13)",
                 lambda: fig13_electron_density_vs_z(
                     r11, prm, z_shift_um=paper_shift_um, show=False))
            _try("Carrier populations vs time (Couairon Fig. 2 style)",
                 lambda: fig2_populations(r11, prm, show=False))

        r10 = results.get("couairon2005_1uJ")
        if r10 is not None:
            _try("Avalanche on/off vs time, 1 uJ (Couairon Fig. 10)",
                 lambda: fig10_avalanche_vs_time(
                     r10, prm, z_um=z_fig10_um, z_shift_um=paper_shift_um, show=False))

        rb = results.get("bulgakova_1uJ")
        if rb is not None:
            _try("Bulgakova Fig. 11 (fluence / absorbed / intensity / density)",
                 lambda: fig11_bulgakova(rb, z_shift_um=bulg_shift_um,
                                         focus_um=90.0, show=False))
            _try("Bulgakova Fig. 12 (sequential absorption)",
                 lambda: fig12_bulgakova(rb, z_shift_um=bulg_shift_um, show=False))

        # --- probe dephasing, only where the (r,t) cube was kept ----------
        for name, res in results.items():
            if _get(res, "rho_rzt") is None:
                continue
            _try(f"Probe dephasing -- {name}",
                 lambda res=res: fig10_dephasage(res, prm, band_um=10.0, show=False))

    if verbose:
        print(f"{len(made)} figure(s) -> {pdf_path}")
        for m in made:
            print(f"   ok      {m}")
        for n, why in skipped:
            print(f"   skipped {n}  ({why})")
    return pdf_path
