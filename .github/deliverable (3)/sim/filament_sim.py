"""
================================================================================
 Femtosecond filamentation in bulk fused silica (SiO2) -- unified solver
 Faithful to Couairon, Sudrie, Franco, Prade, Mysyrowicz,
   PRB 71, 125435 (2005).  SI units throughout.

 This is `NewSim3juillet.py` with the nonlinear term (eq. 3 of the notes)
 split into six independently switchable physics channels so that ablation
 studies ("what does this term actually do to the filament?") can be run
 from a notebook without touching the solver internals:

     enable_kerr_instantaneous     (1-f_R) |E|^2  E                term
     enable_kerr_raman             f_R (R * |E|^2) E               term
     enable_self_steepening        T-hat operator on Kerr + PI-loss
     enable_photoionization_loss   i T-hat Ui WPI/(c eps0 n0 |E|^2) E term
     enable_plasma_defocusing      -sigma*omega0*tau_c/2 * rho  E  term
     enable_plasma_absorption      +i sigma/2 * rho  E             term

 These six flags act ONLY on the field-propagation equation (eq. 3). The
 carrier rate equations (eq. 6-7: avalanche / recombination / STE channel)
 are controlled by the pre-existing `enable_avalanche`, `enable_recombination`
 and `enable_ste` flags -- disabling a field-loss channel here does not stop
 that channel from feeding the free-carrier density, only from acting back
 on the optical field. This is deliberate: it lets you isolate "what does
 the Kerr term do to the beam" from "how many electrons get created".
================================================================================
"""

import os
import sys
import json
import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Callable, Union

import numpy as np
import cupy as cp
from tqdm.auto import tqdm

from scipy.special import jn_zeros
from cupyx.scipy.special import j0, j1
from cupyx.scipy import interpolate as cp_interpolate
from scipy.constants import c, epsilon_0, m_e
from scipy.constants import elementary_charge as q_e

# keldysh.py must live in this same directory (sim/); inserting it explicitly
# means the import works regardless of the caller's sys.path/cwd setup.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from keldysh import SELLMEIER_B, SELLMEIER_L2, n_sellmeier, KeldyshSiO2

# ================================================================================
#  1.  CONFIG -- all parameters live here
# ================================================================================
@dataclass
class Config:
    # ---- numerical grid ----
    nz: int = 40000
    Nt: int = 4096
    N:  int = 1025
    begin: float = -600e-6
    end:   float = 7400e-6
    R_factor: float = 40.0
    save_stride: int = 1
    ckpt_every: int = 500
    verbose: bool = True
    out_dir: Optional[str] = None

    # ---- laser ----
    # w0 and z_focus_air_um are meant to be taken directly from an air-side
    # beam characterization (beam profiler / knife-edge before the sample) :
    # paraxial refraction of a converging Gaussian beam at a flat interface
    # preserves the waist size (w0 is the SAME in air and in the medium, to
    # this order) and only rescales the focus distance by the medium index
    # (z_focus_glass = n0 * z_focus_air). See envelope_gaussian_focused()
    # below and __post_init__ for the derivation/usage -- this is the same
    # rule already used by unified_filament_slider_v3.py
    # (Z_FOCUS_GLASS_DIST_UM = N_GLASS * Z_FOCUS_AIR_DIST_UM).
    wavelength: float = 1030e-9
    energy_uJ:  float = 15.0
    peak_power_W: Optional[float] = None
    w0:      float = 10e-6
    delta_t: float = 263e-15
    # Distance (in air, µm) from the sample entrance face to where the beam
    # would focus if it kept propagating in air (i.e. what you measure/fit
    # from an air-side beam-profiler scan). If set, this OVERRIDES `begin`:
    # begin is computed as -n0 * z_focus_air_um (so the box starts exactly
    # z_focus_air_um "of air" worth of linear propagation before the focus,
    # correctly rescaled into the medium). Leave None to keep specifying
    # `begin` directly in medium-native metres, as before.
    z_focus_air_um: Optional[float] = None

    # ---- material (SiO2, SI) ----
    n2:    float = 2.4e-20
    Ui_eV: float = 9.0
    meff_rel: float = 0.64
    meff_drude_rel: float = 1.0
    tau_c: float = 1.7e-15
    tau_r: float = 330e-15
    f_R:   float = 0.18
    tau_d: float = 32e-15
    tau_s: float = 12e-15
    rho_max: float = 2.1e22

    # ---- STE channel (Chimier PRB 2011; Mao et al. Appl. Phys. A 79, 1695) ----
    # Us_eV: gap seen by an already-trapped exciton being re-ionized by the
    # laser -- replaces Mao's fixed m_x-photon cross-section sigma_x by a
    # Keldysh rate evaluated at Us, so the STE re-ionization channel follows
    # the same physics as the valence-band one.
    Us_eV: float = 6.0
    enable_ste: bool = True
    # Non-radiative STE decay time to the ground state. Sakurai et al.
    # (Quantum Electronics) tabulate a relaxation time of 1 ps for fused
    # silica alongside the 150 fs trapping time. None -> no decay channel
    # (STEs only leave by laser re-ionization), which was the previous
    # behaviour; set e.g. 1e-12 to enable it.
    tau_ste: Optional[float] = None

    # ---- physics switches: field-propagation equation (eq. 3) ----
    # Kerr = instantaneous electronic response + delayed Raman/molecular response.
    enable_kerr_instantaneous: bool = True
    enable_kerr_raman: bool = True
    # T-hat operator (1 + i/omega0 d/dt), applied to the Kerr+photoionization
    # bracket. Turning this off removes shock-front / spectral-asymmetry effects.
    enable_self_steepening: bool = True
    # Energy the field loses to multiphoton/tunnelling ionization (does NOT
    # gate the carrier density itself -- see enable_avalanche/enable_ste below).
    enable_photoionization_loss: bool = True
    # Plasma-induced phase term (index depression -> defocusing), real sigma*omega0*tau_c.
    enable_plasma_defocusing: bool = True
    # Inverse-Bremsstrahlung amplitude loss (real sigma), imaginary in eq. 3.
    enable_plasma_absorption: bool = True
    # Space-time focusing operator U-hat = 1 + (i k1/k0) d/dt, multiplying the
    # LHS d/dz and the diffraction+dispersion bracket in eq. 2 (Couairon 2005).
    # In frequency domain (envelope ~ exp(-i Omega t)) this makes the
    # *effective diffraction wavenumber* frequency-dependent:
    #   k0_eff(Omega) = k0 * U(Omega),   U(Omega) = 1 + (k1/k0) Omega.
    # Tightly focused, short/broadband pulses (large NA, i.e. large Rayleigh-
    # range-to-pulse-duration mismatch) couple space and time through this
    # term even in the purely LINEAR regime -- it is what reshapes/broadens
    # the on-axis temporal profile beyond a plain Gaussian in Couairon 2005's
    # own Fig. 10. Previously not implemented (half_diffraction used a fixed
    # k0 for every frequency); see LinearOperator.half_linear.
    enable_space_time_focusing: bool = True
    # Restrict the field's spectrum to the band where the Sellmeier fit (and
    # therefore delta_k) is actually defined, lambda in [0.18, 5] um. Outside
    # it the envelope model is meaningless: T-hat = 1 + Omega/omega0 = omega/omega0
    # turns NEGATIVE for absolute frequencies below zero (25% of the grid at
    # Nt=2048, 38% at Nt=4096 -- refining the time grid makes it worse, not
    # better), and T-hat^2 then amplifies whatever aliases into that region by
    # up to 9x every step, feeding a high-frequency ripple back into the field.
    enable_spectral_filter: bool = True
    # Bound-oscillator (Lorentz) index change carried by the self-trapped
    # excitons themselves, at the PUMP wavelength. A STE is a localized
    # in-gap state (a dangling Si-O bond pair), NOT a conduction-band
    # carrier: it must be kept out of the Drude terms -- which it is, split()
    # only ever receives rho_e -- but it still polarizes, through a bound
    # resonance at the STE first excited level (~4.2 eV in SiO2, Mao et al.
    # Appl. Phys. A 79, 1695 (2004)):
    #     dn_STE = + [w^2/(w_tr^2 - w^2)] * rho_s / (2 n0 rho_c)
    # Below resonance (800 nm = 1.55 eV << 4.2 eV) this is POSITIVE, i.e. it
    # adds to the Kerr focusing -- and it is the permanent index change that
    # Couairon 2005 identifies as type I damage. Magnitude at the pump:
    # +3.1e-3 at rho_s = 1e20 cm^-3, +1.6e-2 at 5e20, against a Kerr
    # dn = 1.8e-2 at the clamping intensity, so 18-88% of the Kerr term --
    # not negligible. Previously this term existed ONLY in the post-processing
    # for the 490 nm probe; the pump never saw it while propagating.
    # No effect when enable_ste=False (rho_s stays identically 0).
    enable_ste_index: bool = True
    E_tr_eV: float = 4.2

    # ---- physics switches: carrier rate equations (eq. 6-7) ----
    enable_avalanche: bool = True
    enable_recombination: bool = True

    # ---- probe wavelength ----
    lambda_probe: float = 490e-9

    # ---- time-resolved rho snapshot ----
    rho_t_stride: int = 15
    # Bulgakova, Stoian & Rosenfeld, "Laser-induced modification of transparent
    # crystals and glasses", Figs. 11-12. Edges (fs, pulse max at t=0) of the
    # time windows over which the absorbed energy density is integrated
    # separately; their Fig. 12 uses (-100, -50, 0, 50, 100). None -> only the
    # pulse-integrated total (their Fig. 11b) is recorded.
    absorb_time_bins_fs: Optional[tuple] = None
    # Instant (fs, pulse max at t=0) at which to snapshot the free-electron
    # density map; their Fig. 11d uses +50 fs. None -> not recorded (rho_rz
    # already stores the max over time, which is a different quantity).
    rho_snapshot_t_fs: Optional[float] = None

    # ---- adaptive step (reserved, not used by the fixed-step integrator below) ----
    adaptive: bool = False
    dphi_max: float = 0.05
    drho_rel_max: float = 0.10
    dz_min: float = 10e-9
    dz_max: float = 200e-9
    dz_init: Optional[float] = None

    def __post_init__(self):
        self.omega0    = 2 * np.pi * c / self.wavelength
        self.omega_probe = 2 * np.pi * c / self.lambda_probe
        self.n0_probe    = n_sellmeier(self.lambda_probe)
        self.nc_probe    = (epsilon_0 * m_e * self.omega_probe**2 / q_e**2) * 1e-6
        self.frequency = c / self.wavelength
        self.n0        = n_sellmeier(self.wavelength)

        if self.z_focus_air_um is not None:
            # Paraxial refraction of a converging Gaussian beam at a flat
            # interface: focus distance scales by n0, waist size (w0) is
            # unchanged -- so w0 can be taken straight from an air-side beam
            # profiler measurement with no conversion at all.
            self.begin = -self.n0 * self.z_focus_air_um * 1e-6

        self.Ui        = self.Ui_eV * q_e
        self.Us        = self.Us_eV * q_e
        self.meff      = self.meff_rel * m_e
        self.meff_drude = self.meff_drude_rel * m_e
        self.chi3      = 4 / 3 * epsilon_0 * self.n0**2 * c * self.n2
        self.Omega_r2  = 1.0 / self.tau_s**2 + 1.0 / self.tau_d**2
        self.komega    = 2 * np.pi / self.wavelength * self.n0
        self.dz        = (self.end - self.begin) / self.nz
        self.b         = self.komega * self.w0**2
        if self.dz_init is None:
            self.dz_init = min(self.dz, self.dz_max)

        if self.peak_power_W is not None:
            I0_Wm2 = 2.0 * self.peak_power_W / (np.pi * self.w0**2)
        else:
            # Couairon 2005 Sec. III: P_in = E_in / (t_p*sqrt(pi/2)) with
            # t_p = FWHM/sqrt(2 ln2) the 1/e half-width of the FIELD envelope
            # exp(-t^2/t_p^2), then E0^2 = 2 P_in/(pi w0^2). Using the FWHM
            # directly in place of t_p*sqrt(pi/2) overstates the pulse energy
            # by t_p*sqrt(pi/2)/FWHM = 1.0644, i.e. +6.4%.
            tp_env = self.delta_t / np.sqrt(2 * np.log(2))
            P_in_W = self.energy_uJ * 1e-6 / (tp_env * np.sqrt(np.pi / 2))
            I0_Wm2 = 2.0 * P_in_W / (np.pi * self.w0**2)
        self.I0_Wcm2 = I0_Wm2 * 1e-4
        self.E0 = float(np.sqrt(2.0 * self.I0_Wcm2 * 1e4 / (self.n0 * c * epsilon_0)))

        if self.out_dir is None:
            self.out_dir = (f"sio2_{datetime.datetime.now():%Y%m%d_%H%M%S}"
                            f"_E{self.energy_uJ * 1e3:.0f}nJ_{self.delta_t * 1e15:.0f}fs")

        r_max_um   = self.R_factor * self.w0 * 1e6
        Nr_points  = self.N - 1
        dr_um      = r_max_um / Nr_points
        w0_um      = self.w0 * 1e6
        pts_in_w0  = w0_um / dr_um
        if pts_in_w0 < 3:
            print(f"[WARN] RADIAL UNDER-SAMPLING: {pts_in_w0:.1f} pts/w0. Raise N.")
        elif pts_in_w0 < 6:
            print(f"[WARN] Radial sampling marginal: {pts_in_w0:.1f} pts/w0.")


# ================================================================================
#  2.  KELDYSH PHOTOIONISATION RATE -- see keldysh.py (shared with
#      keldysh_reference_fig2.py, which validates it against the paper)
# ================================================================================

# ================================================================================
#  3.  CUDA KERNEL -- plasma rate equation (Chimier 2011 corrected)
# ================================================================================
_RATE_KERNEL_SRC = r'''
__device__ __forceinline__ double interpolate_pi_rate(
    double I, const double* table, int size, double log_min, double inv_log_step, double Imin, double Imax)
{
    if (!isfinite(I) || I < 0.0) I = 0.0;
    if (I < Imin) I = Imin;
    if (I > Imax) I = Imax;
    double fidx = (log10(I) - log_min) * inv_log_step;
    int idx = (int)fidx;
    if (idx < 0) idx = 0;
    if (idx > size - 2) idx = size - 2;
    double r = table[idx] + (fidx - (double)idx) * (table[idx+1] - table[idx]);
    return (!isfinite(r) || r < 0.0) ? 0.0 : r;
}

__device__ __forceinline__ double exact_exp_step(double x, double S, double L, double dt)
{
    double Ldt = L * dt;
    double phi1 = (fabs(Ldt) > 1e-6) ? (exp(Ldt) - 1.0) / Ldt : 1.0 + Ldt * (0.5 + Ldt * (1.0/6.0 + Ldt / 24.0));
    double x_new = exp(Ldt) * x + S * dt * phi1;
    return (!isfinite(x_new) || x_new < 0.0) ? 0.0 : x_new;
}

extern "C" __global__
void solve_rate_equation_kernel(
    const double2* __restrict__ E_field,
    double* __restrict__ ne, double* __restrict__ ns,
    const int Nr, const int Nt, const double dt,
    const double field_to_I,
    const double beta_g, const double beta_s,
    const double na, const double inv_tau_r, const double inv_tau_ste,
    const int enable_ste,
    const double* __restrict__ W_PI_val, const double* __restrict__ W_PI_ste,
    const int table_size, const double log_min_I, const double inv_log_step, const double Imin, const double Imax)
{
    int ri = (int)(blockDim.x * blockIdx.x + threadIdx.x);
    if (ri >= Nr) return;
    int base = ri * Nt;

    double ne_val = ne[base];
    double ns_val = ns[base];
    if (!isfinite(ne_val) || ne_val < 0.0) ne_val = 0.0;
    if (!isfinite(ns_val) || ns_val < 0.0) ns_val = 0.0;

    double2 f0 = E_field[base];
    double I_c = (f0.x*f0.x + f0.y*f0.y) * field_to_I;
    double W_c = interpolate_pi_rate(I_c, W_PI_val, table_size, log_min_I, inv_log_step, Imin, Imax);
    double Ws_c = enable_ste ? interpolate_pi_rate(I_c, W_PI_ste, table_size, log_min_I, inv_log_step, Imin, Imax) : 0.0;

    for (int step = 0; step < Nt - 1; ++step)
    {
        double2 f1 = E_field[base + step + 1];
        double I_n = (f1.x*f1.x + f1.y*f1.y) * field_to_I;
        double W_n = interpolate_pi_rate(I_n, W_PI_val, table_size, log_min_I, inv_log_step, Imin, Imax);
        double Ws_n = enable_ste ? interpolate_pi_rate(I_n, W_PI_ste, table_size, log_min_I, inv_log_step, Imin, Imax) : 0.0;

        double I_avg = 0.5 * (I_c + I_n);
        double W_avg = 0.5 * (W_c + W_n);
        double Ws_avg = 0.5 * (Ws_c + Ws_n);

        double depl = 1.0 - ne_val / na;
        if (depl < 0.0) depl = 0.0;
        if (depl > 1.0) depl = 1.0;

        double S_e = W_avg * depl + (enable_ste ? (Ws_avg + beta_s * I_avg * ne_val) * (ns_val / na) : 0.0);
        double L_e = beta_g * I_avg * depl - inv_tau_r;

        double S_s = enable_ste ? inv_tau_r * ne_val : 0.0;
        // STE loss: laser re-ionization back to the conduction band (Mao et al.,
        // Appl. Phys. A 79, 1695 (2004): -sigma_x N_STE I^m_x, here generalized
        // to a Keldysh rate at the STE gap Us) plus non-radiative decay to the
        // ground state with time tau_ste (Sakurai et al. tabulate 1 ps for
        // fused silica; inv_tau_ste = 0 disables it, the previous behaviour).
        double L_s = enable_ste ? -(Ws_avg + beta_s * I_avg * ne_val) / na - inv_tau_ste : 0.0;

        double ne_new = exact_exp_step(ne_val, S_e, L_e, dt);
        double ns_new = enable_ste ? exact_exp_step(ns_val, S_s, L_s, dt) : 0.0;

        if (ne_new + ns_new > na) {
            double scale = na / (ne_new + ns_new);
            ne_new *= scale;
            ns_new *= scale;
        }

        ne[base + step + 1] = ne_new;
        ns[base + step + 1] = ns_new;

        ne_val = ne_new;
        ns_val = ns_new;
        I_c = I_n;
        W_c = W_n;
        Ws_c = Ws_n;
    }
}
'''
rate_eq_kernel = cp.RawKernel(_RATE_KERNEL_SRC, 'solve_rate_equation_kernel')


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

# ================================================================================
#  6.  LINEAR OPERATOR
# ================================================================================
class LinearOperator:
    """
    Diffraction + dispersion, combined into a single step: both are diagonal
    once expressed in (Hankel radial mode, temporal frequency Omega) space,
    so applying them together (rather than as two separately-ordered half
    steps) is exact -- they commute -- and costs one FFT/IFFT pair per call
    instead of the two pairs a naive frequency-resolved diffraction step
    would need on top of the existing dispersion step.

    The diffraction term uses inv_U_op = 1/U(Omega) (space-time focusing,
    U=1 when disabled) so its effective wavenumber is k0*U(Omega) instead of
    a fixed k0 -- see Config.enable_space_time_focusing.
    """
    def __init__(self, cfg: Config, g: dict):
        self.komega   = g["komega"]
        self.Y        = g["Y"]
        self.j1last   = g["j1last"]
        self.R        = g["R"]
        self.Rk       = self.R**2 / self.j1last
        self.iRk      = self.j1last / self.R**2
        self.rhorho   = g["rhorho"]
        self.delta_k  = g["delta_k"]
        self.inv_U_op = g["inv_U_op"]
        self.spec_mask = g["spec_mask"]

    def half_linear(self, u, dz):
        psik = self.Rk * cp.dot(self.Y, u)
        psik_f = cp.fft.fft(psik, axis=1)
        phase = (self.delta_k - self.rhorho**2 / (2 * self.komega) * self.inv_U_op) * dz / 2
        psik_f = psik_f * cp.exp(1j * phase) * self.spec_mask
        psik = cp.fft.ifft(psik_f, axis=1)
        return self.iRk * cp.dot(self.Y, psik)

# ================================================================================
#  7.  NONLINEAR OPERATOR  (eq. 3 -- six switchable channels)
# ================================================================================
class NonlinearOperator:
    def __init__(self, cfg: Config, g: dict):
        self.n0, self.Ui, self.f_R = cfg.n0, cfg.Ui, cfg.f_R
        self.T_op, self.R_f, self.invE2 = g["T_op"], g["R_f"], g["invE2"]
        # kerr_pref is the shared Kerr-phase prefactor; the two flags below
        # gate the instantaneous / Raman *contributions* to kerr_I, not this
        # prefactor, so partial disabling keeps the correct (1-f_R)/f_R weights.
        self.kerr_pref = 3 * cfg.chi3 * cfg.omega0**2 / (8 * g["komega"] * c**2)
        self.plasma_pref  = (g["sigmaomega"] / 2.0) * 100.0
        self.plasma_phase = 1.0 + 1j * cfg.omega0 * cfg.tau_c

        self.en_kerr_inst     = bool(cfg.enable_kerr_instantaneous)
        self.en_kerr_raman    = bool(cfg.enable_kerr_raman)
        self.en_pi_loss       = bool(cfg.enable_photoionization_loss)
        self.en_plasma_defoc  = bool(cfg.enable_plasma_defocusing)
        self.en_plasma_absorb = bool(cfg.enable_plasma_absorption)
        self.ste_pref = g["ste_pref"]

        kel = g["keldysh"]
        self.f_spline = kel["f_spline"]
        self.Imin, self.Imax = kel["Imin"], kel["Imax"]
        self._kel       = kel
        self.sigmaomega = g["sigmaomega"]
        self.avalanche  = g["avalanche_coef"]
        self.avalanche_s = g["avalanche_coef_s"]
        self.inv_taur   = g["inv_taur_eff"]
        self.inv_tau_ste = g["inv_tau_ste"]
        self.rho_max    = cfg.rho_max
        self.enable_ste = int(cfg.enable_ste)

    def split(self, u, rho, rho_s=None):
        u = cp.ascontiguousarray(u.astype(cp.complex128, copy=False))
        absu2 = cp.abs(u)**2
        W_PI  = cp.nan_to_num(cp.abs(self.f_spline(
            cp.clip(absu2 * self.invE2, self.Imin, self.Imax))),
            nan=0.0, posinf=0.0, neginf=0.0)

        depl_field = cp.clip(1.0 - rho / self.rho_max, 0.0, 1.0)
        photo = (W_PI * 1e6) * self.Ui / (self.n0 * c * epsilon_0 * (absu2 + 1e-30)) * depl_field
        if not self.en_pi_loss:
            # Zeroing here removes the photoionization loss/phase term from the
            # FIELD equation only; carrier generation (rate_eq_kernel, eq. 6-7)
            # is untouched and keeps running off the same Keldysh rate W_PI.
            photo = photo * 0.0

        alpha = photo
        if self.en_plasma_absorb:
            alpha = alpha + self.plasma_pref * rho
        alpha = cp.maximum(cp.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0), 0.0)

        kerr_I = cp.zeros_like(absu2)
        if self.en_kerr_inst:
            kerr_I = kerr_I + (1.0 - self.f_R) * absu2
        if self.en_kerr_raman:
            kerr_I = kerr_I + self.f_R * cp.fft.ifft(cp.fft.fft(absu2, axis=1) * self.R_f, axis=1).real

        # Couairon 2005 Eq. (4): T-hat^2 multiplies the Kerr bracket but only
        # T-hat^1 multiplies the photoionization-loss term, so the two cannot
        # share a single transform. The +photo*u added back after the IFFT
        # cancels the zeroth order of -T*photo*u (that part is already applied
        # through exp(-alpha*dz/2) in Integrator.step), leaving only the
        # self-steepening correction -(T-1)*photo*u here -- no double counting.
        NL_freq = (cp.fft.fft(1j * self.kerr_pref * kerr_I * u, axis=1) * self.T_op**2
                   - cp.fft.fft(photo * u, axis=1) * self.T_op)
        rhs = cp.fft.ifft(NL_freq, axis=1) + photo * u
        if self.en_plasma_defoc:
            rhs = rhs - self.plasma_pref * (self.plasma_phase - 1.0) * rho * u
        if self.ste_pref and rho_s is not None:
            # Bound (non-Drude) STE polarizability -- pure phase, no loss:
            # the pump at 1.55 eV is far below the 4.2 eV STE resonance, so
            # there is no single-photon STE absorption to account for here.
            rhs = rhs + 1j * self.ste_pref * rho_s * u
        rhs = cp.nan_to_num(cp.where(absu2 < 1e-30, 0.0 + 0.0j, rhs), nan=0.0, posinf=0.0, neginf=0.0)
        return rhs, alpha

    def loss_rates(self, u, rho):
        """r,t-resolved absorption-rate fields (1/m), photoionization and
        plasma-absorption channels kept separate -- mirrors split()'s photo/
        alpha computation exactly, so this matches what the field actually
        loses. Used only for Fig. 12-style energy-loss bookkeeping
        (Integrator._record), never on the hot RK4 path."""
        absu2 = cp.abs(u) ** 2
        W_PI = cp.nan_to_num(cp.abs(self.f_spline(
            cp.clip(absu2 * self.invE2, self.Imin, self.Imax))),
            nan=0.0, posinf=0.0, neginf=0.0)
        depl_field = cp.clip(1.0 - rho / self.rho_max, 0.0, 1.0)
        photo = (W_PI * 1e6) * self.Ui / (self.n0 * c * epsilon_0 * (absu2 + 1e-30)) * depl_field
        if not self.en_pi_loss:
            photo = photo * 0.0
        plasma = self.plasma_pref * rho if self.en_plasma_absorb else cp.zeros_like(photo)
        return photo, plasma

    def update_plasma(self, u, rho, rho_s, dt, blocks, threads):
        rho  [:, 0] = 0.0
        rho_s[:, 0] = 0.0
        Nr, Nt = u.shape
        rate_eq_kernel(
            (blocks,), (threads,),
            (cp.ascontiguousarray(u), rho, rho_s,
             Nr, Nt, dt,
             float(self.invE2),
             float(self.avalanche),
             float(self.avalanche_s),
             float(self.rho_max),
             float(self.inv_taur),
             float(self.inv_tau_ste),
             int(self.enable_ste),
             self._kel["W_LUT"], self._kel["W_LUT_s"],
             int(self._kel["NLUT"]),
             float(self._kel["logImin"]), float(self._kel["inv_dlog"]),
             float(self._kel["Imin"]), float(self._kel["Imax"])))

# ================================================================================
#  8.  INTEGRATOR
# ================================================================================
class Integrator:
    def __init__(self, cfg: Config, envelope: Union[str, Callable] = "gaussian_focused"):
        self.cfg = cfg
        os.makedirs(cfg.out_dir, exist_ok=True)

        g = build_grids(cfg)
        self.g = g
        self.dz, self.dt = cfg.dz, g["dt"]
        self.lin = LinearOperator(cfg, g)
        self.nl  = NonlinearOperator(cfg, g)

        fn = envelope if callable(envelope) else ENVELOPES[envelope]
        self.u = cp.ascontiguousarray(fn(g["rr"], g["tt"], cfg, g).astype(cp.complex128, copy=False))
        self.Nr, self.Nt = self.u.shape
        self.rho   = cp.zeros((self.Nr, self.Nt), dtype=cp.float64)
        self.rho_s = cp.zeros((self.Nr, self.Nt), dtype=cp.float64)
        self.mask_r = g["mask_r"]
        self.threads = 256
        self.blocks  = (self.Nr + self.threads - 1) // self.threads

        fl0 = cp.sum(cp.abs(self.u)**2, axis=1) * self.dt * g["invE2"] * 1e4
        self.U0_uJ = float(cp.sum(fl0 * 2.0 * cp.pi * g["rlist"]
                                  * cp.diff(g["rlist"], prepend=0.0))) * 1e6
        if cfg.verbose:
            print(f"[init] U_beam(0) = {self.U0_uJ:.3f} uJ ", flush=True)

        n_saves = cfg.nz // cfg.save_stride + 1
        self.fluence_rz = cp.zeros((n_saves, self.Nr), dtype=cp.float32)
        self.rho_rz     = cp.zeros((n_saves, self.Nr), dtype=cp.float32)
        self.rho_s_rz   = cp.zeros((n_saves, self.Nr), dtype=cp.float32)
        self.Imax_z     = np.zeros(n_saves, dtype=np.float64)
        self.z_saved    = np.zeros(n_saves, dtype=np.float64)
        # dE/dz (uJ/m) at each saved z, photoionization and plasma-absorption
        # channels kept separate (see NonlinearOperator.loss_rates) -- turned
        # into the cumulative fractional losses of Fig. 12 (Couairon 2005) in
        # _results(). No separate STE loss channel is modeled (the paper's
        # Fig. 12 has none either), so E_STE_z stays 0.
        self._dEdz_photo_uJm  = np.zeros(n_saves, dtype=np.float64)
        self._dEdz_plasma_uJm = np.zeros(n_saves, dtype=np.float64)
        self.E_STE_z    = np.zeros(n_saves, dtype=np.float64)
        self.k_save = 0

        tlist_fs = cp.asnumpy(g['tlist']) * 1e15
        self.t_full_fs = tlist_fs

        self._t_stride = max(1, cfg.rho_t_stride) if cfg.rho_t_stride > 0 else 0
        if self._t_stride > 0:
            Nt_sub = (self.Nt - 1) // self._t_stride + 1
            self.rho_rzt   = np.zeros((n_saves, self.Nr, Nt_sub), dtype=np.float32)
            self.rho_s_rzt = np.zeros((n_saves, self.Nr, Nt_sub), dtype=np.float32)
            self.I_rzt     = np.zeros((n_saves, self.Nr, Nt_sub), dtype=np.float32)
            self.t_sub_fs  = tlist_fs[::self._t_stride]
        else:
            self.rho_rzt, self.rho_s_rzt, self.I_rzt, self.t_sub_fs = None, None, None, None

        # On-axis (r index 0, closest to the axis), FULL time resolution --
        # independent of rho_t_stride and cheap (no radial dimension, unlike
        # rho_rzt/I_rzt above: ~Nt floats per saved z-plane). This is what
        # figures_article.py's 0D reintegration should read: a rho_t_stride
        # subsampled I_rzt can miss a narrow intensity spike between two
        # saved samples, and since multiphoton rate scales roughly as I^K
        # (K ~ 8-9 photons for a 9 eV gap at 1030 nm), missing that spike can
        # make the reintegrated density look many orders of magnitude below
        # what the CUDA kernel (which ran on the full grid) actually computed.
        # Absorbed energy density (J/cm^3) deposited locally at each (r, z):
        # integral over t of 2*alpha*I, alpha from NonlinearOperator.loss_rates.
        self.absorbed_rz = cp.zeros((n_saves, self.Nr), dtype=cp.float32)
        tl_fs = tlist_fs
        if cfg.absorb_time_bins_fs:
            edges = np.asarray(cfg.absorb_time_bins_fs, dtype=np.float64)
            self._absorb_masks = [cp.asarray((tl_fs >= a) & (tl_fs < b))
                                  for a, b in zip(edges[:-1], edges[1:])]
            self.absorb_bin_edges_fs = edges
            self.absorbed_rz_bins = cp.zeros((n_saves, len(self._absorb_masks), self.Nr),
                                             dtype=cp.float32)
        else:
            self._absorb_masks, self.absorb_bin_edges_fs, self.absorbed_rz_bins = None, None, None
        # Peak intensity at each (r, z) -- max over time, per radius (Imax_z is
        # the max over r AND t, a different quantity).
        self.Ipeak_rz = cp.zeros((n_saves, self.Nr), dtype=cp.float32)
        # Free-electron density at one instant rather than its max over time.
        if cfg.rho_snapshot_t_fs is not None:
            self._it_snap = int(np.argmin(np.abs(tl_fs - cfg.rho_snapshot_t_fs)))
            self.rho_rz_at = cp.zeros((n_saves, self.Nr), dtype=cp.float32)
        else:
            self._it_snap, self.rho_rz_at = None, None

        self.rho_onaxis_t   = np.zeros((n_saves, self.Nt), dtype=np.float32)
        self.rho_s_onaxis_t = np.zeros((n_saves, self.Nt), dtype=np.float32)
        self.I_onaxis_t     = np.zeros((n_saves, self.Nt), dtype=np.float32)

    def step(self, dz):
        u, rho, rho_s = self.u, self.rho, self.rho_s
        u = self.lin.half_linear(u, dz)
        _, a = self.nl.split(u, rho, rho_s); u = u * cp.exp(-0.5 * dz * a)
        k1, _ = self.nl.split(u,               rho, rho_s)
        k2, _ = self.nl.split(u + 0.5 * dz * k1, rho, rho_s)
        k3, _ = self.nl.split(u + 0.5 * dz * k2, rho, rho_s)
        k4, _ = self.nl.split(u +       dz * k3, rho, rho_s)
        u = u + dz / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        _, a = self.nl.split(u, rho, rho_s); u = u * cp.exp(-0.5 * dz * a)
        u = self.lin.half_linear(u, dz)
        self.u = u * self.mask_r

    def propagate(self):
        cfg = self.cfg
        pbar = tqdm(range(cfg.nz + 1), desc="Filamentation", unit="step",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

        for i in pbar:
            self.nl.update_plasma(self.u, self.rho, self.rho_s, self.dt, self.blocks, self.threads)
            self.step(self.dz)
            z_now = cfg.begin + i * self.dz

            if i % cfg.save_stride == 0:
                self._record(z_now)

            if i > 0 and (i % cfg.ckpt_every == 0):
                k = self.k_save - 1
                I_peak = float(self.Imax_z[k])

                flu_cpu = cp.asnumpy(self.fluence_rz[k])
                r_cpu = cp.asnumpy(self.g["rlist"])
                dr_cpu = np.diff(r_cpu, prepend=0.0)
                U_now_uJ = float(np.sum(flu_cpu * 2.0 * np.pi * r_cpu * dr_cpu)) * 100.0
                pct_u = U_now_uJ / self.U0_uJ * 100.0

                pbar.set_postfix(z=f"{z_now*1e6:+.0f}µm", U=f"{pct_u:.1f}%", I_peak=f"{I_peak:.2e}")

        pbar.close()
        return self._results()

    def _record(self, z_now):
        k = self.k_save
        g = self.g
        absu2 = cp.abs(self.u)**2
        I_full = absu2 * g["invE2"]
        self.fluence_rz[k] = (cp.sum(absu2, axis=1) * self.dt * g["invE2"]).astype(cp.float32, copy=False)
        self.rho_rz[k]     = cp.max(self.rho,   axis=1).astype(cp.float32, copy=False)
        self.rho_s_rz[k]   = cp.max(self.rho_s, axis=1).astype(cp.float32, copy=False)
        self.Imax_z[k]     = float(I_full.max())
        self.z_saved[k]    = float(z_now)

        # Fig. 12 (Couairon 2005): dE/dz per channel, integrated over the
        # transverse plane the same way U0_uJ is (fluence x 2*pi*r*dr, cm^2
        # -> m^2, J -> uJ), with an extra 2*alpha_channel(r,t) weight since
        # dI/dz = -2*alpha*I for the field-amplitude decay applied in step().
        photo_al, plasma_al = self.nl.loss_rates(self.u, self.rho)

        # Local absorbed energy density (J/cm^3): dW/dV = int 2 alpha I dt.
        # I_full is W/cm^2 and alpha is 1/m, so 2*alpha*I*dt is J/(cm^2 m);
        # x100 converts the 1/m into 1/cm -> J/cm^3.
        dep = 2.0 * (photo_al + plasma_al) * I_full * self.dt * 100.0
        self.absorbed_rz[k] = cp.sum(dep, axis=1).astype(cp.float32, copy=False)
        if self._absorb_masks is not None:
            for b, m in enumerate(self._absorb_masks):
                self.absorbed_rz_bins[k, b] = cp.sum(dep * m, axis=1).astype(cp.float32, copy=False)
        self.Ipeak_rz[k] = cp.max(I_full, axis=1).astype(cp.float32, copy=False)
        if self._it_snap is not None:
            self.rho_rz_at[k] = self.rho[:, self._it_snap].astype(cp.float32, copy=False)

        r_cpu, dr_cpu = g["rlist"], cp.diff(g["rlist"], prepend=0.0)
        floss_photo  = cp.sum(2.0 * photo_al  * I_full, axis=1) * self.dt
        floss_plasma = cp.sum(2.0 * plasma_al * I_full, axis=1) * self.dt
        self._dEdz_photo_uJm[k]  = float(cp.sum(floss_photo  * 2.0 * cp.pi * r_cpu * dr_cpu)) * 1e4 * 1e6
        self._dEdz_plasma_uJm[k] = float(cp.sum(floss_plasma * 2.0 * cp.pi * r_cpu * dr_cpu)) * 1e4 * 1e6

        if self._t_stride > 0 and self.rho_rzt is not None:
            self.rho_rzt  [k] = cp.asnumpy(self.rho  [:, ::self._t_stride].astype(cp.float32, copy=False))
            self.rho_s_rzt[k] = cp.asnumpy(self.rho_s[:, ::self._t_stride].astype(cp.float32, copy=False))
            self.I_rzt    [k] = cp.asnumpy(I_full[:, ::self._t_stride].astype(cp.float32, copy=False))

        self.rho_onaxis_t  [k] = cp.asnumpy(self.rho  [0].astype(cp.float32, copy=False))
        self.rho_s_onaxis_t[k] = cp.asnumpy(self.rho_s[0].astype(cp.float32, copy=False))
        self.I_onaxis_t    [k] = cp.asnumpy(I_full[0].astype(cp.float32, copy=False))

        self.k_save += 1

    def _cumulative_energy_fraction(self, dEdz_uJm):
        """Trapezoidal cumulative integral of dE/dz (uJ/m) over z_saved,
        normalized by U0_uJ -- the fractional cumulative energy loss curve
        of Fig. 12 (Couairon 2005)."""
        z = self.z_saved[:self.k_save]
        dEdz = dEdz_uJm[:self.k_save]
        if len(z) < 2:
            return np.zeros_like(dEdz)
        seg_uJ = 0.5 * (dEdz[:-1] + dEdz[1:]) * np.diff(z)
        cum_uJ = np.concatenate([[0.0], np.cumsum(seg_uJ)])
        return cum_uJ / self.U0_uJ

    def _results(self):
        cfg, g = self.cfg, self.g
        def _mirror(a):
            return np.hstack([a[:, ::-1], a])
        r_cpu     = cp.asnumpy(g["rlist"])
        flu_cpu   = cp.asnumpy(self.fluence_rz[:self.k_save])
        rho_cpu   = cp.asnumpy(self.rho_rz[:self.k_save])
        rho_s_cpu = cp.asnumpy(self.rho_s_rz[:self.k_save])

        E_MPI_z    = self._cumulative_energy_fraction(self._dEdz_photo_uJm)
        E_plasma_z = self._cumulative_energy_fraction(self._dEdz_plasma_uJm)
        E_total    = E_MPI_z + E_plasma_z

        out = dict(
            r=np.concatenate([-r_cpu[::-1], r_cpu]),
            rlist=r_cpu,
            z=self.z_saved[:self.k_save],
            fluence_rz=np.hstack([flu_cpu[:, ::-1], flu_cpu]),
            rho_rz=np.hstack([rho_cpu[:, ::-1], rho_cpu]),
            rho_s_rz=np.hstack([rho_s_cpu[:, ::-1], rho_s_cpu]),
            Imax_z=self.Imax_z[:self.k_save],
            E_plasma_z=E_plasma_z,
            E_MPI_z=E_MPI_z,
            E_STE_z=self.E_STE_z[:self.k_save],
            E_total_z=E_total,
            rho_rzt=(self.rho_rzt[:self.k_save] if self.rho_rzt is not None else None),
            rho_s_rzt=(self.rho_s_rzt[:self.k_save] if self.rho_s_rzt is not None else None),
            I_rzt=(self.I_rzt[:self.k_save] if self.I_rzt is not None else None),
            t_sub_fs=(self.t_sub_fs if self.t_sub_fs is not None else None),
            absorbed_rz=_mirror(cp.asnumpy(self.absorbed_rz[:self.k_save])),
            Ipeak_rz=_mirror(cp.asnumpy(self.Ipeak_rz[:self.k_save])),
            rho_rz_at=(_mirror(cp.asnumpy(self.rho_rz_at[:self.k_save]))
                       if self.rho_rz_at is not None else None),
            absorbed_rz_bins=(np.stack([_mirror(cp.asnumpy(self.absorbed_rz_bins[:self.k_save, b]))
                                        for b in range(self.absorbed_rz_bins.shape[1])], axis=1)
                              if self.absorbed_rz_bins is not None else None),
            absorb_bin_edges_fs=self.absorb_bin_edges_fs,
            rho_onaxis_t=self.rho_onaxis_t[:self.k_save],
            rho_s_onaxis_t=self.rho_s_onaxis_t[:self.k_save],
            I_onaxis_t=self.I_onaxis_t[:self.k_save],
            t_full_fs=self.t_full_fs,
        )
        np.savez_compressed(os.path.join(cfg.out_dir, "result.npz"), **out)
        self._dump_params()
        return out

    def _dump_params(self):
        """Companion params.json (probe optics + which physics channels were
        active), consumed by web/abel_phase_explorer.py to compute Delta n and
        to label ablation scenarios without re-deriving them from the npz."""
        cfg = self.cfg
        params = dict(
            n0=cfg.n0, n0_probe=cfg.n0_probe,
            nc_probe_cm3=cfg.nc_probe,
            lambda_probe_nm=cfg.lambda_probe * 1e9,
            wavelength_nm=cfg.wavelength * 1e9,
            n2=cfg.n2, U_g_eV=cfg.Ui_eV, Us_eV=cfg.Us_eV,
            energy_uJ=cfg.energy_uJ, w0_um=cfg.w0 * 1e6,
            delta_t_fs=cfg.delta_t * 1e15,
            # z_sim = 0 is always the gaussian geometric focus (see
            # envelope_gaussian_focused: curvature = 1 at begin = 0), so the
            # focus-to-interface distance in lab space is simply -begin_um
            # whenever `begin` was set to minus that distance (as in the
            # original notebook). Exposed here so abel_phase_explorer.py does
            # not need a hardcoded experimental-geometry constant.
            begin_um=cfg.begin * 1e6, end_um=cfg.end * 1e6,
            z_focus_air_um=cfg.z_focus_air_um,
            toggles=dict(
                enable_kerr_instantaneous=cfg.enable_kerr_instantaneous,
                enable_kerr_raman=cfg.enable_kerr_raman,
                enable_self_steepening=cfg.enable_self_steepening,
                enable_photoionization_loss=cfg.enable_photoionization_loss,
                enable_plasma_defocusing=cfg.enable_plasma_defocusing,
                enable_plasma_absorption=cfg.enable_plasma_absorption,
                enable_space_time_focusing=cfg.enable_space_time_focusing,
                enable_spectral_filter=cfg.enable_spectral_filter,
                tau_ste_fs=(cfg.tau_ste * 1e15 if cfg.tau_ste else None),
                enable_ste_index=cfg.enable_ste_index,
                E_tr_eV=cfg.E_tr_eV, f_ste_pump=float(self.g["f_ste"]),
                enable_avalanche=cfg.enable_avalanche,
                enable_recombination=cfg.enable_recombination,
                enable_ste=cfg.enable_ste,
            ),
        )
        with open(os.path.join(cfg.out_dir, "params.json"), "w") as f:
            json.dump(params, f, indent=2)

# ================================================================================
#  9.  ENTRY POINT
# ================================================================================
def run(*, Nz, Nt, Nr, wavelength,
        energy_uJ=15.0, peak_power_W=None,
        envelope="gaussian_focused",
        w0=10e-6, delta_t=263e-15,
        begin=-600e-6, end=1800e-6,
        n2=3.54e-20, Ui_eV=9.0, R_factor=100.0,
        save_stride=1, ckpt_every=500, verbose=True, out_dir=None,
        **material):
    """
    material may include ANY Config field, in particular the six field-level
    toggles (enable_kerr_instantaneous, enable_kerr_raman,
    enable_self_steepening, enable_photoionization_loss,
    enable_plasma_defocusing, enable_plasma_absorption) plus the pre-existing
    carrier-level toggles (enable_avalanche, enable_recombination, enable_ste).
    """
    cfg = Config(
        nz=Nz, Nt=Nt, N=Nr, wavelength=wavelength,
        energy_uJ=energy_uJ, peak_power_W=peak_power_W,
        w0=w0, delta_t=delta_t, begin=begin, end=end,
        n2=n2, Ui_eV=Ui_eV, R_factor=R_factor,
        save_stride=save_stride, ckpt_every=ckpt_every, verbose=verbose, out_dir=out_dir,
        **material,
    )
    return Integrator(cfg, envelope=envelope).propagate()

# Names of the six field-propagation toggles, in the order they appear in
# equation (3). Used by the notebook to build the ablation-scenario loop and
# the checkbox UI without hardcoding the list twice.
FIELD_TOGGLES = [
    "enable_kerr_instantaneous",
    "enable_kerr_raman",
    "enable_self_steepening",
    "enable_photoionization_loss",
    "enable_plasma_defocusing",
    "enable_plasma_absorption",
]

if __name__ == "__main__":
    print("Module loaded. Use run(...) to start simulation.")
