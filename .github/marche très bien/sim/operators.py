"""
Split-step operators: the linear half step (diffraction + dispersion, diagonal
in the Hankel/frequency basis) and the nonlinear term (eq. 3) together with the
carrier update that drives the CUDA kernel.

The nonlinear term is written as a registry: each line of the field equation is
one FieldTerm below, and split() is a loop over the enabled ones. See the block
comment above FIELD_TERMS for the contract, and MODIFYING_THE_EQUATIONS.md for
how to add a term.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

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
    a fixed k0 -- see Config.enable_space_time_focusing. Note that inv_U_op
    multiplies the diffraction term ONLY, not delta_k: diffraction goes as
    1/k(omega) and dispersion does not. Written with U-hat moved to the left
    of d/dz, that is why the dispersion term reads i D^ U^ u.
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
#  7.  NONLINEAR OPERATOR  (eq. 3, as a registry of terms)
# ================================================================================
#  One line of the field equation, one FieldTerm. A term declares WHAT it is,
#  not how to apply it, and split() is a loop. The important field is `kind`.
#
#    kind = "phase"  fn returns the term's contribution to du/dz directly.
#
#    kind = "loss"   fn returns a non-negative absorption RATE, in 1/m. The
#                    loop then does BOTH things a loss needs: it adds the rate
#                    to alpha, which Integrator.step applies as the two
#                    exp(-alpha dz/2) factors around the RK4 block, and it puts
#                    -rate*u into the right-hand side so the RK4 does not apply
#                    the same loss a second time.
#
#  That second point is the reason this file is a registry. The double
#  bookkeeping used to be spelled out by hand for every dissipative channel,
#  and forgetting either half made the absorption silently wrong by a factor
#  of about two. A term now cannot be half registered: declaring a rate is the
#  only way to declare a loss, and the loop derives both halves from it.
#
#  T_power is the power of T-hat in front of the term. Couairon 2005 Eq. (4)
#  puts T^2 on the Kerr bracket, T^1 on the photoionization loss and T^0 on the
#  rest, and those powers are not interchangeable. Terms are grouped by T_power
#  before transforming, so the number of FFTs per call is at most three however
#  many terms are enabled.
#
#  Everything here is the right-hand side BEFORE dividing by U-hat. split()
#  applies inv_U_nl to the assembled sum, because Eq. (2) carries U-hat in
#  front of d/dz and therefore divides the whole right-hand side.
# ================================================================================

@dataclass(frozen=True)
class FieldTerm:
    """One line of the field equation."""
    name: str          # short label, also the key used by loss_rates()
    flag: str          # Config field that switches it off
    kind: str          # "phase" or "loss"
    T_power: int       # 0, 1 or 2
    fn: Callable       # (op, ctx) -> array, or None to contribute nothing
    equation: str      # the line as it appears in the written equation


class _Ctx:
    """Quantities shared by several terms, each computed at most once.

    Only `photo` is expensive (a spline lookup over the whole r,t plane), and
    it is skipped entirely when the photoionization loss is disabled.
    """
    __slots__ = ("op", "u", "absu2", "rho", "rho_s", "_photo")

    def __init__(self, op, u, absu2, rho, rho_s):
        self.op, self.u, self.absu2 = op, u, absu2
        self.rho, self.rho_s = rho, rho_s
        self._photo = None

    @property
    def photo(self):
        """Photoionization absorption rate, 1/m.

        Ui * W_PI / (n0 c eps0 |u|^2), times the valence depletion factor.
        The 1e6 converts the tabulated rate from cm^-3 s^-1 to m^-3 s^-1.
        """
        if self._photo is None:
            op = self.op
            W_PI = cp.nan_to_num(cp.abs(op.f_spline(
                cp.clip(self.absu2 * op.invE2, op.Imin, op.Imax))),
                nan=0.0, posinf=0.0, neginf=0.0)
            depl = cp.clip(1.0 - self.rho / op.rho_max, 0.0, 1.0)
            self._photo = ((W_PI * 1e6) * op.Ui
                           / (op.n0 * c * epsilon_0 * (self.absu2 + 1e-30)) * depl)
        return self._photo


def _kerr_instantaneous(op, ctx):
    return 1j * op.kerr_pref * (1.0 - op.f_R) * ctx.absu2 * ctx.u


def _kerr_raman(op, ctx):
    raman = cp.fft.ifft(cp.fft.fft(ctx.absu2, axis=1) * op.R_f, axis=1).real
    return 1j * op.kerr_pref * op.f_R * raman * ctx.u


def _photoionization_loss(op, ctx):
    return ctx.photo


def _plasma_absorption(op, ctx):
    return op.plasma_pref * ctx.rho


def _plasma_defocusing(op, ctx):
    # plasma_phase - 1 = i w0 tau_c, so this is -i (sigma_w w0 tau_c / 2) N u.
    return -op.plasma_pref * (op.plasma_phase - 1.0) * ctx.rho * ctx.u


def _ste_index(op, ctx):
    # ste_pref is already zero when enable_ste is off, and rho_s is absent on
    # the loss_rates path, so both cases contribute nothing.
    if not op.ste_pref or ctx.rho_s is None:
        return None
    return 1j * op.ste_pref * ctx.rho_s * ctx.u


#  Order matters twice over: it is the order the equation is written in, and it
#  is the order the floating point sums are accumulated in.
FIELD_TERMS = (
    FieldTerm("kerr_instantaneous", "enable_kerr_instantaneous", "phase", 2,
              _kerr_instantaneous,
              "+ i T^2 kerr_pref (1-f_R) |u|^2 u"),
    FieldTerm("kerr_raman", "enable_kerr_raman", "phase", 2,
              _kerr_raman,
              "+ i T^2 kerr_pref f_R (R*|u|^2) u"),
    FieldTerm("photoionization_loss", "enable_photoionization_loss", "loss", 1,
              _photoionization_loss,
              "- T^ (Ui W_PI / n0 c eps0 |u|^2) (1 - N/N_at) u"),
    FieldTerm("plasma_absorption", "enable_plasma_absorption", "loss", 0,
              _plasma_absorption,
              "- (sigma_w / 2) N u"),
    FieldTerm("plasma_defocusing", "enable_plasma_defocusing", "phase", 0,
              _plasma_defocusing,
              "- i (sigma_w w0 tau_c / 2) N u"),
    FieldTerm("ste_index", "enable_ste_index", "phase", 0,
              _ste_index,
              "+ i (w0 / 2 n0 c rho_c) f_STE N_STE u"),
)


class NonlinearOperator:
    def __init__(self, cfg: Config, g: dict):
        self.n0, self.Ui, self.f_R = cfg.n0, cfg.Ui, cfg.f_R
        self.T_op, self.R_f, self.invE2 = g["T_op"], g["R_f"], g["invE2"]
        self.inv_U_nl = g["inv_U_nl"]      # U^-1 for the nonlinear step; see split()
        # kerr_pref is the shared Kerr-phase prefactor. It is algebraically
        # equal to (w0/c) n2 I / |u|^2, i.e. the Kerr term is i k_vac dn u.
        self.kerr_pref = 3 * cfg.chi3 * cfg.omega0**2 / (8 * g["komega"] * c**2)
        self.plasma_pref  = (g["sigmaomega"] / 2.0) * 100.0
        self.plasma_phase = 1.0 + 1j * cfg.omega0 * cfg.tau_c
        self.ste_pref = g["ste_pref"]

        # The registry, filtered once. A disabled term is simply absent, which
        # is why split() has no branches on the flags.
        self.terms = tuple(t for t in FIELD_TERMS if bool(getattr(cfg, t.flag)))
        # T^ and T^2, precomputed. T^0 is handled by skipping the multiply.
        self._T_pow = {1: self.T_op, 2: self.T_op**2}

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

    def active_equation(self) -> str:
        """The field equation as actually assembled, one enabled term per line."""
        lines = ["U^ du/dz =   (i / 2k0) grad_perp^2 u",
                 "           + i D^ U^ u"]
        lines += [f"           {t.equation}" for t in self.terms]
        return "\n".join(lines)

    def split(self, u, rho, rho_s=None):
        """Right-hand side of the field equation, minus the part Integrator.step
        already applies exponentially.

        Returns (rhs, alpha). Over one z step the two exp(-alpha dz/2) factors
        contribute -alpha*u and the trailing + alpha * u here cancels them, so
        the net nonlinear contribution to du/dz is exactly
        ifft(NL_freq * inv_U_nl) and nothing else. Splitting alpha out is a
        numerical device for applying the stiff dissipative channels
        exponentially rather than through the RK4, not physics.
        """
        u = cp.ascontiguousarray(u.astype(cp.complex128, copy=False))
        absu2 = cp.abs(u)**2
        ctx = _Ctx(self, u, absu2, rho, rho_s)

        # One pass over the enabled terms. Phase terms go straight into their
        # T_power group; loss terms go into alpha AND into their group as
        # -rate*u, which is the whole point of the registry.
        groups = {}
        alpha = None
        for term in self.terms:
            v = term.fn(self, ctx)
            if v is None:
                continue
            if term.kind == "loss":
                alpha = v if alpha is None else alpha + v
                v = -v * u
            g = groups.get(term.T_power)
            groups[term.T_power] = v if g is None else g + v

        if alpha is None:
            alpha = cp.zeros_like(absu2)
        alpha = cp.maximum(cp.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0), 0.0)

        # At most three FFTs, one per T-hat power, highest first so the sum is
        # accumulated in the order the equation is written.
        NL_freq = None
        for p in (2, 1, 0):
            v = groups.get(p)
            if v is None:
                continue
            f = cp.fft.fft(v, axis=1)
            if p:
                f = f * self._T_pow[p]
            NL_freq = f if NL_freq is None else NL_freq + f

        if NL_freq is None:
            rhs = alpha * u
        else:
            rhs = cp.fft.ifft(NL_freq * self.inv_U_nl, axis=1) + alpha * u
        rhs = cp.nan_to_num(cp.where(absu2 < 1e-30, 0.0 + 0.0j, rhs),
                            nan=0.0, posinf=0.0, neginf=0.0)
        return rhs, alpha

    def loss_rates(self, u, rho):
        """r,t-resolved absorption rates (1/m), one per dissipative channel.

        Returns (photoionization, plasma) for the energy-loss bookkeeping of
        Integrator._record. It calls the same term functions split() calls, so
        the two cannot drift apart -- they used to be two copies of the same
        arithmetic. Never on the hot RK4 path.
        """
        absu2 = cp.abs(u) ** 2
        ctx = _Ctx(self, u, absu2, rho, None)
        rates = {}
        for term in self.terms:
            if term.kind != "loss":
                continue
            v = term.fn(self, ctx)
            if v is not None:
                rates[term.name] = v
        zero = cp.zeros_like(absu2)
        return (rates.get("photoionization_loss", zero),
                rates.get("plasma_absorption", zero))

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
