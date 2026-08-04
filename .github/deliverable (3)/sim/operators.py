"""
Split-step operators: the linear half step (diffraction + dispersion, diagonal
in the Hankel/frequency basis) and the nonlinear term (eq. 3, six switchable
channels) together with the carrier update that drives the CUDA kernel.
"""

import sys
from pathlib import Path

import numpy as np
import cupy as cp

from scipy.constants import c, epsilon_0

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import Config      # noqa: E402
from kernels import rate_eq_kernel  # noqa: E402

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
        self.inv_U_nl = g["inv_U_nl"]      # U^-1 for the nonlinear step; see split()
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
        # share a single transform.
        #
        # Eq. (2) of the same paper carries U-hat in front of d/dz, so solving
        # for du/dz divides the ENTIRE right-hand side by U -- Kerr, ionization,
        # plasma and the STE polarizability alike, not just the Laplacian that
        # half_linear already handles. Everything is therefore assembled in
        # frequency space and multiplied by inv_U_nl in one go. Omitting that
        # factor while keeping T^2 leaves the self-steepening roughly twice too
        # strong, since U^-1 T^2 = 1 + 0.99*Omega/w0 whereas T^2 = 1 + 2*Omega/w0.
        #
        # The exp(-alpha*dz/2) channel in Integrator.step already applies the
        # zeroth order of the two dissipative terms, so alpha*u is added back
        # here and the net contribution to du/dz is exactly ifft(NL_freq*inv_U).
        # With space-time focusing off inv_U_nl is 1 and this reduces
        # algebraically to the previous expression.
        plasma_coeff = ((1.0 if self.en_plasma_absorb else 0.0)
                        + ((self.plasma_phase - 1.0) if self.en_plasma_defoc else 0.0))
        # Bound (non-Drude) STE polarizability -- pure phase, no loss: the pump
        # at 1.55 eV is far below the 4.2 eV STE resonance, so there is no
        # single-photon STE absorption to account for here.
        extra = -self.plasma_pref * plasma_coeff * rho * u
        if self.ste_pref and rho_s is not None:
            extra = extra + 1j * self.ste_pref * rho_s * u

        NL_freq = (cp.fft.fft(1j * self.kerr_pref * kerr_I * u, axis=1) * self.T_op**2
                   - cp.fft.fft(photo * u, axis=1) * self.T_op
                   + cp.fft.fft(extra, axis=1))
        rhs = cp.fft.ifft(NL_freq * self.inv_U_nl, axis=1) + alpha * u
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

