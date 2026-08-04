"""
================================================================================
 Femtosecond filamentation in bulk fused silica (SiO2) -- unified solver
 Faithful to Couairon, Sudrie, Franco, Prade, Mysyrowicz,
   PRB 71, 125435 (2005).  SI units throughout.
================================================================================
"""

import os
import time
import json
import datetime
from dataclasses import dataclass
from typing import Optional, Callable, Union

import numpy as np
import cupy as cp
from tqdm.auto import tqdm

from scipy.special import jn_zeros, ellipk, ellipe, dawsn
from cupyx.scipy.special import j0, j1
from cupyx.scipy import interpolate as cp_interpolate
from scipy.constants import c, epsilon_0, m_e
from scipy.constants import elementary_charge as q_e

# ================================================================================
#  Sellmeier dispersion of fused silica
# ================================================================================
SELLMEIER_B  = np.array([0.6961663, 0.4079426, 0.8974794])
SELLMEIER_L2 = np.array([0.0684043, 0.1162414, 9.896161]) ** 2

def n_sellmeier(lam_m):
    lam_um = lam_m * 1e6
    n2m1 = sum(B * lam_um**2 / (lam_um**2 - L2)
               for B, L2 in zip(SELLMEIER_B, SELLMEIER_L2))
    return float(np.sqrt(1.0 + n2m1))

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
    wavelength: float = 1030e-9
    energy_uJ:  float = 15.0
    peak_power_W: Optional[float] = None
    w0:      float = 10e-6
    delta_t: float = 263e-15

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

    # ---- STE channel (Chimier PRB 2011) ----
    Us_eV: float = 6.0
    enable_ste: bool = True

    # ---- physics switches ----
    enable_kerr: bool = True
    enable_avalanche: bool = True
    enable_recombination: bool = True

    # ---- probe wavelength ----
    lambda_probe: float = 490e-9

    # ---- time-resolved rho snapshot ----
    rho_t_stride: int = 15

    # ---- adaptive step ----
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
            I0_Wm2 = 2.0 * self.energy_uJ * 1e-6 / (np.pi * self.w0**2 * self.delta_t)
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
#  2.  KELDYSH PHOTOIONISATION RATE
# ================================================================================
class KeldyshSiO2:
    def __init__(self, wavelength, Ui_eV, meff, n0, N_sum=50):
        self.U     = Ui_eV * q_e
        self.meff  = meff
        self.n0    = n0
        self.omega = 2 * np.pi * c / wavelength
        self.hbar  = 1.054571817e-34
        self.N_sum = N_sum

    def rate(self, intensity_Wcm2):
        I_Wm2 = np.asarray(intensity_Wcm2) * 1e4
        E     = np.sqrt(2.0 * I_Wm2 / (self.n0 * c * epsilon_0))
        gm    = np.maximum(self.omega * np.sqrt(self.meff * self.U) / (q_e * E), 1e-12)
        gm1   = gm / np.sqrt(1 + gm**2)
        gm2   = gm1 / gm
        K1, E1 = ellipk(gm1**2), ellipe(gm1**2)
        K2, E2 = ellipk(gm2**2), ellipe(gm2**2)
        x     = (2 * self.U * E2) / (np.pi * gm1 * self.hbar * self.omega)
        n     = np.arange(self.N_sum)[:, np.newaxis]
        arg   = np.pi * np.sqrt((2 * np.floor(x + 1) - 2 * x + n) / (2 * K2 * E2))
        sigma_val = np.sum(np.exp(-np.pi * n * (K1 - E1) / E2) * dawsn(arg), axis=0)
        Q    = np.sqrt(np.pi / (2 * K2)) * sigma_val
        pref = (2 * self.omega / (9 * np.pi)) * (self.omega * self.meff / (self.hbar * gm1))**(3 / 2)
        return pref * Q * np.exp(-np.pi * np.floor(x + 1) * (K1 - E1) / E2) * 1e-6


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
    const double na, const double inv_tau_r,
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
        double L_s = enable_ste ? -(Ws_avg + beta_s * I_avg * ne_val) / na : 0.0;

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

    T_op = 1.0 + ff / cfg.frequency
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
    invE2 = 0.5 * cfg.n0 * c * epsilon_0 * 1e-4
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
        delta_k=delta_k, k1=k1, T_op=T_op, R_f=R_f,
        rr=rr, tt=tt, rhorho=rhorho,
        sigmaomega=sigmaomega, avalanche_coef=avalanche_coef,
        avalanche_coef_s=avalanche_coef_s,
        inv_taur_eff=inv_taur_eff, invE2=invE2, mask_r=mask_r,
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
    def __init__(self, cfg: Config, g: dict):
        self.komega  = g["komega"]
        self.Y       = g["Y"]
        self.j1last  = g["j1last"]
        self.R       = g["R"]
        self.Rk      = self.R**2 / self.j1last
        self.iRk     = self.j1last / self.R**2
        self.rhorho  = g["rhorho"]
        self.delta_k = g["delta_k"]

    def half_diffraction(self, u, dz):
        psik = self.Rk * cp.dot(self.Y, u)
        psik *= cp.exp(-1j * self.rhorho**2 / (2 * self.komega) * dz / 2)
        return self.iRk * cp.dot(self.Y, psik)

    def half_dispersion(self, u, dz):
        psik = cp.fft.fft(u, axis=1)
        psik *= cp.exp(1j * self.delta_k * dz / 2)
        return cp.fft.ifft(psik, axis=1)

# ================================================================================
#  7.  NONLINEAR OPERATOR
# ================================================================================
class NonlinearOperator:
    def __init__(self, cfg: Config, g: dict):
        self.n0, self.Ui, self.f_R = cfg.n0, cfg.Ui, cfg.f_R
        self.T_op, self.R_f, self.invE2 = g["T_op"], g["R_f"], g["invE2"]
        self.kerr_pref = (3 * cfg.chi3 * cfg.omega0**2 / (8 * g["komega"] * c**2)
                          if cfg.enable_kerr else 0.0)
        self.plasma_pref  = (g["sigmaomega"] / 2.0) * 100.0
        self.plasma_phase = 1.0 + 1j * cfg.omega0 * cfg.tau_c

        kel = g["keldysh"]
        self.f_spline = kel["f_spline"]
        self.Imin, self.Imax = kel["Imin"], kel["Imax"]
        self._kel       = kel
        self.sigmaomega = g["sigmaomega"]
        self.avalanche  = g["avalanche_coef"]
        self.avalanche_s = g["avalanche_coef_s"]
        self.inv_taur   = g["inv_taur_eff"]
        self.rho_max    = cfg.rho_max
        self.enable_ste = int(cfg.enable_ste)

    def split(self, u, rho):
        u = cp.ascontiguousarray(u.astype(cp.complex128, copy=False))
        absu2 = cp.abs(u)**2
        W_PI  = cp.nan_to_num(cp.abs(self.f_spline(
            cp.clip(absu2 * self.invE2, self.Imin, self.Imax))),
            nan=0.0, posinf=0.0, neginf=0.0)
        
        depl_field = cp.clip(1.0 - rho / self.rho_max, 0.0, 1.0)
        photo = (W_PI * 1e6) * self.Ui / (self.n0 * c * epsilon_0 * (absu2 + 1e-30)) * depl_field
        
        alpha = self.plasma_pref * rho + photo
        alpha = cp.maximum(cp.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0), 0.0)

        kerr_I = (1.0 - self.f_R) * absu2 \
            + self.f_R * cp.fft.ifft(cp.fft.fft(absu2, axis=1) * self.R_f, axis=1).real
        NL_freq = cp.fft.fft(1j * self.kerr_pref * kerr_I * u - photo * u, axis=1)
        rhs = cp.fft.ifft(NL_freq * self.T_op, axis=1) \
            + photo * u \
            - self.plasma_pref * (self.plasma_phase - 1.0) * rho * u
        rhs = cp.nan_to_num(cp.where(absu2 < 1e-30, 0.0 + 0.0j, rhs), nan=0.0, posinf=0.0, neginf=0.0)
        return rhs, alpha

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
        self.E_plasma_z = np.zeros(n_saves, dtype=np.float64)
        self.E_MPI_z    = np.zeros(n_saves, dtype=np.float64)
        self.E_STE_z    = np.zeros(n_saves, dtype=np.float64)
        self.k_save = 0

        self._t_stride = max(1, cfg.rho_t_stride) if cfg.rho_t_stride > 0 else 0
        if self._t_stride > 0:
            Nt_sub = (self.Nt - 1) // self._t_stride + 1
            self.rho_rzt   = np.zeros((n_saves, self.Nr, Nt_sub), dtype=np.float32)
            self.rho_s_rzt = np.zeros((n_saves, self.Nr, Nt_sub), dtype=np.float32)
            self.I_rzt     = np.zeros((n_saves, self.Nr, Nt_sub), dtype=np.float32)
            tlist_fs = cp.asnumpy(g['tlist']) * 1e15
            self.t_sub_fs  = tlist_fs[::self._t_stride]
        else:
            self.rho_rzt, self.rho_s_rzt, self.I_rzt, self.t_sub_fs = None, None, None, None

    def step(self, dz):
        u, rho = self.u, self.rho
        u = self.lin.half_diffraction(u, dz)
        u = self.lin.half_dispersion(u, dz)
        _, a = self.nl.split(u, rho); u = u * cp.exp(-0.5 * dz * a)
        k1, _ = self.nl.split(u,               rho)
        k2, _ = self.nl.split(u + 0.5 * dz * k1, rho)
        k3, _ = self.nl.split(u + 0.5 * dz * k2, rho)
        k4, _ = self.nl.split(u +       dz * k3, rho)
        u = u + dz / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        _, a = self.nl.split(u, rho); u = u * cp.exp(-0.5 * dz * a)
        u = self.lin.half_dispersion(u, dz)
        u = self.lin.half_diffraction(u, dz)
        self.u = u * self.mask_r

    def propagate(self):
        cfg = self.cfg
        # Initialisation de la barre de progression Jupyter
        pbar = tqdm(range(cfg.nz + 1), desc="Filamentation", unit="step", 
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
        
        for i in pbar:
            self.nl.update_plasma(self.u, self.rho, self.rho_s, self.dt, self.blocks, self.threads)
            self.step(self.dz)
            z_now = cfg.begin + i * self.dz
            
            if i % cfg.save_stride == 0:
                self._record(z_now)
                
            # Mise à jour de la barre tous les ckpt_every steps
            if i > 0 and (i % cfg.ckpt_every == 0):
                k = self.k_save - 1
                I_peak = float(self.Imax_z[k])
                
                # Calcul propre de l'énergie pour l'affichage
                flu_cpu = cp.asnumpy(self.fluence_rz[k])
                r_cpu = cp.asnumpy(self.g["rlist"])
                dr_cpu = np.diff(r_cpu, prepend=0.0)
                U_now_uJ = float(np.sum(flu_cpu * 2.0 * np.pi * r_cpu * dr_cpu)) * 100.0 
                pct_u = U_now_uJ / self.U0_uJ * 100.0
                
                # On attache les infos à la barre tqdm au lieu de faire un print()
                pbar.set_postfix(z=f"{z_now*1e6:+.0f}µm", U=f"{pct_u:.1f}%", I_peak=f"{I_peak:.2e}")
                
        pbar.close()
        return self._results()

    def _live_diag(self, z_now):
        # Cette fonction n'est plus appelée par propagate() mais on la garde pour compatibilité
        k = self.k_save - 1
        I_peak = float(self.Imax_z[k])
        flu_cpu = cp.asnumpy(self.fluence_rz[k])
        r_cpu = cp.asnumpy(self.g["rlist"])
        dr_cpu = np.diff(r_cpu, prepend=0.0)
        U_now_uJ = float(np.sum(flu_cpu * 2.0 * np.pi * r_cpu * dr_cpu)) * 100.0 
        print(f"[z={z_now*1e6:+8.1f}um] U_beam={U_now_uJ/self.U0_uJ*100:.1f}% I_peak={I_peak:.2e}", flush=True)

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
        
        if self._t_stride > 0 and self.rho_rzt is not None:
            self.rho_rzt  [k] = cp.asnumpy(self.rho  [:, ::self._t_stride].astype(cp.float32, copy=False))
            self.rho_s_rzt[k] = cp.asnumpy(self.rho_s[:, ::self._t_stride].astype(cp.float32, copy=False))
            self.I_rzt    [k] = cp.asnumpy(I_full[:, ::self._t_stride].astype(cp.float32, copy=False))
            
        self.k_save += 1

    def _results(self):
        cfg, g = self.cfg, self.g
        r_cpu     = cp.asnumpy(g["rlist"])
        flu_cpu   = cp.asnumpy(self.fluence_rz[:self.k_save])
        rho_cpu   = cp.asnumpy(self.rho_rz[:self.k_save])
        rho_s_cpu = cp.asnumpy(self.rho_s_rz[:self.k_save])
        
        E_total = self.E_plasma_z[:self.k_save] + self.E_MPI_z[:self.k_save]
        
        out = dict(
            r=np.concatenate([-r_cpu[::-1], r_cpu]),
            z=self.z_saved[:self.k_save],
            fluence_rz=np.hstack([flu_cpu[:, ::-1], flu_cpu]),
            rho_rz=np.hstack([rho_cpu[:, ::-1], rho_cpu]),
            rho_s_rz=np.hstack([rho_s_cpu[:, ::-1], rho_s_cpu]),
            Imax_z=self.Imax_z[:self.k_save],
            E_plasma_z=self.E_plasma_z[:self.k_save],
            E_MPI_z=self.E_MPI_z[:self.k_save],
            E_STE_z=self.E_STE_z[:self.k_save],
            E_total_z=E_total,
            rho_rzt=(self.rho_rzt[:self.k_save] if self.rho_rzt is not None else None),
            rho_s_rzt=(self.rho_s_rzt[:self.k_save] if self.rho_s_rzt is not None else None),
            I_rzt=(self.I_rzt[:self.k_save] if self.I_rzt is not None else None),
            t_sub_fs=(self.t_sub_fs if self.t_sub_fs is not None else None),
        )
        np.savez_compressed(os.path.join(cfg.out_dir, "result.npz"), **out)
        return out

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
    cfg = Config(
        nz=Nz, Nt=Nt, N=Nr, wavelength=wavelength,
        energy_uJ=energy_uJ, peak_power_W=peak_power_W,
        w0=w0, delta_t=delta_t, begin=begin, end=end,
        n2=n2, Ui_eV=Ui_eV, R_factor=R_factor,
        save_stride=save_stride, ckpt_every=ckpt_every, verbose=verbose, out_dir=out_dir,
        **material,
    )
    return Integrator(cfg, envelope=envelope).propagate()

if __name__ == "__main__":
    print("Module loaded. Use run(...) to start simulation.")