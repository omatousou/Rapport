"""
Numerical grids (Hankel radial modes, time/frequency axes, dispersion,
Keldysh look-up tables) and the initial field envelopes.

build_grids() returns a plain dict consumed by the operators and the
integrator; keeping it here means the grid layout can be inspected without
importing the propagation machinery.
"""

import sys
from pathlib import Path

import numpy as np
import cupy as cp

from scipy.special import jn_zeros
from cupyx.scipy.special import j0, j1
from cupyx.scipy import interpolate as cp_interpolate
from scipy.constants import c, epsilon_0, m_e
from scipy.constants import elementary_charge as q_e

sys.path.insert(0, str(Path(__file__).resolve().parent))
from keldysh import SELLMEIER_B, SELLMEIER_L2, n_sellmeier, KeldyshSiO2  # noqa: E402
from config import Config  # noqa: E402

# ================================================================================
#  4.  GRIDS
# ================================================================================
def build_grids(cfg: Config) -> dict:
    komega, omega0 = cfg.komega, cfg.omega0

    tp       = cfg.delta_t / np.sqrt(2 * np.log(2))
    tmax     = 5 * tp
    dt       = float(tmax * 2 / cfg.Nt)
    tlist    = cp.linspace(-tmax, tmax, cfg.Nt, endpoint=False)
    ff       = cp.fft.fftfreq(cfg.Nt, d=dt)

    N   = cfg.N
    j1l = cp.asarray(jn_zeros(0, N))
    R   = cfg.R_factor * cfg.w0
    idx = cp.meshgrid(cp.arange(N - 1, dtype=int), cp.arange(N - 1, dtype=int))
    Y   = (2 / j1l[N - 1]) / j1(j1l[idx[0]])**2 * j0(j1l[idx[0]] * j1l[idx[1]] / j1l[N - 1])
    rlist   = j1l[0:N - 1] * R / j1l[N - 1]
    rholist = j1l[0:N - 1] / R

    omega_safe = np.clip(omega0 + 2 * np.pi * cp.asnumpy(ff),
                         2 * np.pi * c / 5.0e-6, 2 * np.pi * c / 0.18e-6)
    lam_um = (2 * np.pi * c / omega_safe) * 1e6
    n2m1 = np.zeros_like(lam_um)
    for B, L2 in zip(SELLMEIER_B, SELLMEIER_L2):
        n2m1 += B * lam_um**2 / (lam_um**2 - L2)

    def _k_of(w):
        lu = 2 * np.pi * c / w * 1e6
        return (w / c) * np.sqrt(1.0 + sum(B * lu**2 / (lu**2 - L2)
                                           for B, L2 in zip(SELLMEIER_B, SELLMEIER_L2)))
    dw = omega0 / 1000.0
    k1 = (_k_of(omega0 + dw) - _k_of(omega0 - dw)) / (2 * dw)
    delta_k = cp.asarray((omega_safe / c) * np.sqrt(1.0 + n2m1)
                         - float(komega) - k1 * (2 * np.pi * cp.asnumpy(ff)))

    # Spectral band mask: keep only absolute frequencies inside the Sellmeier
    # validity window lambda in [0.18, 5] um (same clip as omega_safe above),
    # with smooth tanh edges to avoid Gibbs ringing. u = omega/omega0.
    u_norm = (cfg.frequency + ff) / cfg.frequency
    if cfg.enable_spectral_filter:
        u_lo = (c / 5.0e-6) / cfg.frequency          # ~0.16  (lambda = 5 um)
        u_hi = (c / 0.18e-6) / cfg.frequency         # ~4.44  (lambda = 0.18 um)
        w_edge = 0.05
        spec_mask = 0.25 * (1.0 + cp.tanh((u_norm - u_lo) / w_edge)) \
                         * (1.0 + cp.tanh((u_hi - u_norm) / w_edge))
    else:
        spec_mask = cp.ones_like(ff)

    # Self-steepening operator T-hat = 1 + (i/omega0) d/dt  <->  1 + Omega/omega0
    # = omega/omega0 in frequency domain. Fully disabled -> identity (no shock /
    # spectral tilt). Multiplied by spec_mask so the region where it would be
    # negative (unphysical, omega < 0) can never amplify anything.
    if cfg.enable_self_steepening:
        T_op = (1.0 + ff / cfg.frequency) * spec_mask
    else:
        T_op = cp.ones_like(ff)

    # Space-time focusing operator U-hat = 1 + (i k1/k0) d/dt <-> 1 + (k1/k0) Omega
    # in frequency domain (same Omega = 2*pi*ff convention as delta_k/T_op
    # above). Effective diffraction wavenumber becomes k0*U(Omega); disabled
    # -> identity, i.e. exactly the previous fixed-k0 diffraction behaviour.
    if cfg.enable_space_time_focusing:
        U_op = 1.0 + k1 * (2 * np.pi * ff) / komega
    else:
        U_op = cp.ones_like(ff)
    inv_U_op = 1.0 / U_op

    R_f  = cp.fft.fft(cp.fft.ifftshift(cp.where(
                tlist > 0,
                cfg.Omega_r2 * cfg.tau_s * cp.exp(-tlist / cfg.tau_d) * cp.sin(tlist / cfg.tau_s),
                cp.zeros_like(tlist)))) * dt

    tt, rr     = cp.meshgrid(tlist, rlist)
    _,  rhorho = cp.meshgrid(tlist, rholist)

    sigmaomega = ((float(komega) * q_e**2 * cfg.tau_c) /
                  (cfg.n0**2 * cfg.meff_drude * epsilon_0 * omega0 *
                   (1.0 + (omega0 * cfg.tau_c)**2))) * 1e4
    avalanche_coef = (sigmaomega / cfg.Ui) if cfg.enable_avalanche else 0.0
    avalanche_coef_s = (sigmaomega / cfg.Us if (cfg.enable_ste and cfg.enable_avalanche) else 0.0)
    inv_taur_eff   = (1.0 / cfg.tau_r) if cfg.enable_recombination else 0.0
    inv_tau_ste    = (1.0 / cfg.tau_ste) if (cfg.tau_ste and cfg.enable_ste) else 0.0
    invE2 = 0.5 * cfg.n0 * c * epsilon_0 * 1e-4

    # STE bound-oscillator prefactor: dn_STE = ste_lorentz * rho_s, then the
    # field picks up i*k0*dn_STE, so ste_pref = k0 * ste_lorentz has units of
    # m^-1 per cm^-3 (rho_s is stored in cm^-3, like rho).
    rho_c_pump = (epsilon_0 * m_e * omega0**2 / q_e**2) * 1e-6          # cm^-3
    w_tr = cfg.E_tr_eV * q_e / 1.054571817e-34
    f_ste = omega0**2 / (w_tr**2 - omega0**2)
    ste_pref = ((omega0 / c) * f_ste / (2.0 * cfg.n0 * rho_c_pump)
                if (cfg.enable_ste and cfg.enable_ste_index) else 0.0)
    mask_r = cp.exp(-(rr / (0.9 * R))**20)

    I_cpu = np.ravel(10**(np.arange(17) + 0.0)[:, None] * (0.01 * np.arange(900) + 1)[None, :])

    W_cpu_g = np.maximum(np.nan_to_num(
        KeldyshSiO2(cfg.wavelength, cfg.Ui_eV, cfg.meff, cfg.n0).rate(I_cpu)), 0.0)
    f_spline_g = cp_interpolate.PchipInterpolator(
        cp.asarray(I_cpu, dtype=cp.float64), cp.asarray(W_cpu_g, dtype=cp.float64))

    W_cpu_s = np.maximum(np.nan_to_num(
        KeldyshSiO2(cfg.wavelength, cfg.Us_eV, cfg.meff, cfg.n0).rate(I_cpu)), 0.0)
    f_spline_s = cp_interpolate.PchipInterpolator(
        cp.asarray(I_cpu, dtype=cp.float64), cp.asarray(W_cpu_s, dtype=cp.float64))

    logI = cp.linspace(float(np.log10(I_cpu[0])), float(np.log10(I_cpu[-1])), 4096, dtype=cp.float64)
    keldysh = dict(
        f_spline=f_spline_g,
        W_LUT=cp.nan_to_num(cp.abs(f_spline_g(10.0**logI)).astype(cp.float64), nan=0.0, posinf=0.0, neginf=0.0),
        W_LUT_s=cp.nan_to_num(cp.abs(f_spline_s(10.0**logI)).astype(cp.float64), nan=0.0, posinf=0.0, neginf=0.0),
        NLUT=4096,
        logImin=float(np.log10(I_cpu[0])),
        inv_dlog=float(4095 / (float(np.log10(I_cpu[-1])) - float(np.log10(I_cpu[0])))),
        Imin=float(I_cpu[0]), Imax=float(I_cpu[-1]),
    )

    return dict(
        komega=float(komega), tmax=tmax, dt=dt, tlist=tlist, ff=ff,
        Y=Y, R=R, j1last=j1l[N - 1], rlist=rlist, rholist=rholist,
        delta_k=delta_k, k1=k1, T_op=T_op, inv_U_op=inv_U_op, R_f=R_f,
        spec_mask=spec_mask,
        rr=rr, tt=tt, rhorho=rhorho,
        sigmaomega=sigmaomega, avalanche_coef=avalanche_coef,
        avalanche_coef_s=avalanche_coef_s,
        inv_taur_eff=inv_taur_eff, inv_tau_ste=inv_tau_ste,
        ste_pref=ste_pref, f_ste=f_ste,
        invE2=invE2, mask_r=mask_r,
        b=cfg.b, E0=cfg.E0, keldysh=keldysh,
    )

# ================================================================================
#  5.  INITIAL ENVELOPES
# ================================================================================
def envelope_gaussian_focused(rr, tt, cfg, g):
    tp   = cfg.delta_t / np.sqrt(2 * np.log(2))
    curv = 1 + 1j * 2 * cfg.begin / g["b"]
    return (g["E0"] / curv) * cp.exp(-rr**2 / cfg.w0**2 / curv) * cp.exp(-tt**2 / tp**2)

ENVELOPES = {"gaussian_focused": envelope_gaussian_focused}

