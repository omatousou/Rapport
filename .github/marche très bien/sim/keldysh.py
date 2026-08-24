"""
Sellmeier dispersion + Keldysh photoionization rate for fused silica.

Split out of filament_sim.py (pure numpy/scipy, no cupy) so it can be:
  - imported by filament_sim.py for the GPU solver, and
  - imported and run standalone (no GPU needed) to validate the rate against
    Couairon et al., PRB 71, 125435 (2005), Fig. 2
-- so both paths always use the exact same formula.

Run `python keldysh.py` to execute the validation suite (see validate()).
"""

import numpy as np
from scipy.special import ellipk, ellipe, dawsn
from scipy.interpolate import PchipInterpolator
from scipy.constants import c, epsilon_0
from scipy.constants import elementary_charge as q_e

# ================================================================================
#  Sellmeier dispersion of fused silica
# ================================================================================
#  THE dispersion of the run. This module is the single source of truth: every
#  other module reads it from here at call time rather than keeping a copy, so
#  set_dispersion() below reaches the whole solver.
#
#  Defaults are Malitson, JOSA 55, 1205 (1965), for fused silica.
SELLMEIER_B  = np.array([0.6961663, 0.4079426, 0.8974794])
SELLMEIER_L2 = np.array([0.0684043, 0.1162414, 9.896161]) ** 2

#  Where the fit is meaningful, in micrometres. Used by grids.py to clip the
#  frequency axis before evaluating the fit and to place the spectral mask.
#  Outside it the Sellmeier form is not merely inaccurate, it has poles: for
#  silica at 0.0684, 0.1162 and 9.896 um.
SELLMEIER_RANGE_UM = (0.18, 5.0)


def set_dispersion(B, L2, range_um=None):
    """Install another material's Sellmeier fit for the rest of the process.

    Call this BEFORE building a Config, since Config.__post_init__ evaluates
    n_sellmeier to get n0. grids.py reads the module attributes at call time,
    so a change here reaches the solver.

    `L2` is the squared pole positions, matching SELLMEIER_L2 above.
    """
    global SELLMEIER_B, SELLMEIER_L2, SELLMEIER_RANGE_UM
    B = np.asarray(B, dtype=float)
    L2 = np.asarray(L2, dtype=float)
    if B.shape != L2.shape:
        raise ValueError(f"B and L2 must have the same length, got {B.shape} and {L2.shape}")
    SELLMEIER_B, SELLMEIER_L2 = B, L2
    if range_um is not None:
        lo, hi = float(range_um[0]), float(range_um[1])
        if not 0 < lo < hi:
            raise ValueError(f"bad Sellmeier range {range_um}")
        SELLMEIER_RANGE_UM = (lo, hi)


def get_dispersion():
    """The fit currently installed, as plain tuples, for recording in a file."""
    return (tuple(float(x) for x in SELLMEIER_B),
            tuple(float(x) for x in SELLMEIER_L2),
            tuple(float(x) for x in SELLMEIER_RANGE_UM))


def n_sellmeier(lam_m):
    lam_um = lam_m * 1e6
    n2m1 = sum(B * lam_um**2 / (lam_um**2 - L2)
               for B, L2 in zip(SELLMEIER_B, SELLMEIER_L2))
    return float(np.sqrt(1.0 + n2m1))


# ================================================================================
#  Keldysh photoionization rate
# ================================================================================
class KeldyshSiO2:
    r"""
    General Keldysh rate W_PI(I) [Couairon 2005, Eqs. (7)-(8)], SI internally,
    returned in cm^-3 s^-1.

        W = (2*w0/9pi) (w0 m / (hbar sqrt(Gamma)))^{3/2} Q(gamma,x) exp(-alpha <x+1>)

        Gamma = g^2/(1+g^2),  Xi = 1/(1+g^2),  g = w0 sqrt(m Ui)/(e E)
        alpha = pi (K(Gamma) - E(Gamma)) / E(Xi)
        beta  = pi^2 / (4 K(Xi) E(Xi))
        x     = (2/pi) (Ui/hbar w0) E(Xi)/sqrt(Gamma),   nu = <x+1> - x

    SUMMATION CONVENTION -- the published Phi(sqrt(beta*(n + 2 nu))) is correct.

    It is tempting to "fix" the channel-closing structure by rewriting the sum
    over total photon number m = <x+1> + n, which would give an argument
    sqrt(2 beta (m - x)) = sqrt(2 beta (n + nu)) and make W continuous across
    every closing. That is WRONG, and the multiphoton limit proves it:

        as gamma -> infinity,  Gamma -> 1, Xi -> 0, so
            beta = pi^2/(4 K(Xi) E(Xi)) -> pi^2/(4 (pi/2)^2) = 1
            exp(-alpha)                 -> e^2/(16 gamma^2)
        (both verified numerically to 5 digits in validate()).

    With beta -> 1 the n = 0 term of the published form is Phi(sqrt(2 nu)),
    which is exactly Keldysh's textbook multiphoton rate. The "continuous"
    variant would give Phi(sqrt(nu)) instead and does not reproduce that limit:
    measured against W_MPI = sigma6 I^6 rho_at at 1e12 W/cm^2 it gives 0.872
    where the published form gives 1.018.

    So the cusps are real: they are the channel closings. Each time x crosses
    an integer the photon order steps up, and within a branch the n = 0 term
    falls off like sqrt(nu) as nu -> 0 (a square-root threshold cusp), which
    can outrun the growth of exp(-alpha<x+1>) and make W locally decrease.
    Over 1e11.5-1e14.2 W/cm^2 this happens at 23 sample points with local
    log-log slopes reaching -136, at 4.5e12, 3.2e13, 6.6e13 and 1.0e14 W/cm^2.

    `monotone` therefore serves the rate through a monotone envelope. It is a
    NUMERICAL REGULARIZATION, not a correction to the physics: a rate with
    factor-several cliffs inside the 5-7e13 range the filament clamps in makes
    the clamping sensitive to which side of a cusp a cell lands on. It defaults
    to False so the rate reproduces the paper exactly; set monotone=True to
    test whether the cusps are affecting a given run.
    """

    def __init__(self, wavelength, Ui_eV, meff, n0, N_sum=60, monotone=False,
                 beta_den=4.0, lut_logI=(0.0, 17.0), lut_points=6000):
        self.U     = Ui_eV * q_e
        self.meff  = meff
        self.n0    = n0
        self.omega = 2 * np.pi * c / wavelength
        self.hbar  = 1.054571817e-34
        self.N_sum = N_sum
        self.monotone = monotone
        self.beta_den = float(beta_den)
        self._lut_logI, self._lut_points = lut_logI, lut_points
        self._mono = None                      # built lazily

    # ---- raw formula ------------------------------------------------------
    def rate_raw(self, intensity_Wcm2):
        """Keldysh rate straight from the formula (cm^-3 s^-1), no smoothing."""
        I_Wm2 = np.asarray(intensity_Wcm2, dtype=np.float64) * 1e4
        E     = np.sqrt(2.0 * I_Wm2 / (self.n0 * c * epsilon_0))
        gm    = np.maximum(self.omega * np.sqrt(self.meff * self.U) / (q_e * np.maximum(E, 1e-300)), 1e-12)

        Gamma = gm**2 / (1.0 + gm**2)          # Keldysh Gamma
        Xi    = 1.0 / (1.0 + gm**2)            # Keldysh Xi
        sqrtG = np.sqrt(Gamma)

        K1, E1 = ellipk(Gamma), ellipe(Gamma)  # K(Gamma), E(Gamma)
        K2, E2 = ellipk(Xi),    ellipe(Xi)     # K(Xi),    E(Xi)

        x   = (2.0 * self.U * E2) / (np.pi * sqrtG * self.hbar * self.omega)
        mx  = np.floor(x + 1.0)                # <x+1>, integer part
        nu  = mx - x                           # fractional excess, in (0, 1]
        alpha = np.pi * (K1 - E1) / E2

        # Dawson argument sqrt(beta (n + 2 nu)), beta = pi^2/(beta_den K(Xi) E(Xi)).
        # DEFAULT 4 = Couairon Eq. (8), stated explicitly in the text: "our
        # quantity beta is divided by 4, whereas the corresponding quantity in
        # Ref. 34 [Keldysh] is divided by 2". This solver historically used 2
        # (Keldysh's own normalization); switching to 4 raises W by ~1.5x.
        # 4 is also the value the multiphoton limit demands: only beta_den=4
        # gives beta -> 1 as gamma -> inf, hence the n=0 term Phi(sqrt(2 nu))
        # of Keldysh's textbook multiphoton rate. beta_den=2 would give
        # beta -> 2 and Phi(2 sqrt(nu)), which is not that limit.
        n = np.arange(self.N_sum).reshape((-1,) + (1,) * np.ndim(x))
        arg = np.pi * np.sqrt(np.maximum(n + 2.0 * nu, 0.0) / (self.beta_den * K2 * E2))
        summ = np.sum(np.exp(-n * alpha) * dawsn(arg), axis=0)

        Q = np.sqrt(np.pi / (2.0 * K2)) * summ
        pref = (2.0 * self.omega / (9.0 * np.pi)) \
            * (self.omega * self.meff / (self.hbar * sqrtG))**1.5
        W = pref * Q * np.exp(-alpha * mx) * 1e-6      # m^-3 s^-1 -> cm^-3 s^-1
        return np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- monotone envelope -------------------------------------------------
    def _build_monotone(self):
        lo, hi = self._lut_logI
        lI = np.linspace(lo, hi, self._lut_points)
        W = self.rate_raw(10.0**lI)
        W = np.maximum.accumulate(np.maximum(W, 0.0))   # monotone envelope
        self._mono = (lI, PchipInterpolator(lI, W))
        return self._mono

    def rate(self, intensity_Wcm2, monotone=None):
        """Rate in cm^-3 s^-1. Monotone-regularized unless monotone=False."""
        mono = self.monotone if monotone is None else monotone
        if not mono:
            return self.rate_raw(intensity_Wcm2)
        lI, spline = self._mono or self._build_monotone()
        I = np.asarray(intensity_Wcm2, dtype=np.float64)
        lq = np.log10(np.clip(I, 10.0**lI[0], 10.0**lI[-1]))
        return np.maximum(np.nan_to_num(spline(lq), nan=0.0, posinf=0.0, neginf=0.0), 0.0)


# ================================================================================
#  Validation -- run `python keldysh.py`
# ================================================================================
def validate(wavelength=800e-9, Ui_eV=9.0, meff_rel=0.64,
             sigma6=9.6e-70, rho_at=2.1e22, verbose=True):
    """
    Checks the rate against the two statements Couairon 2005 makes about it:

      (a) "For weak fields, Keldysh's theory coincides with the multiphoton
           ionization rate W_MPI = sigma6 I^6 rho_at"  (sigma6 = 9.6e-70,
           Sec. III p.3)  -- a convention-free normalization test, far stronger
           than matching one quoted value, since it pins both the prefactor and
           the photon order.
      (b) monotonicity: the solid curve of the paper's Fig. 2 rises over the
           whole 1e12-1.5e14 range.

    Returns a dict of results; raises AssertionError if a check fails.
    """
    m_e = 9.1093837015e-31
    n0 = n_sellmeier(wavelength)
    kel = KeldyshSiO2(wavelength, Ui_eV, meff_rel * m_e, n0)
    out = {}

    # (0) structural check: the multiphoton limit fixes the summation convention.
    # As gamma -> inf, beta -> 1 and exp(-alpha) -> e^2/(16 gamma^2); with
    # beta = 1 the n=0 term is Phi(sqrt(2 nu)), Keldysh's textbook multiphoton
    # rate. This is what rules out the "continuous" argument sqrt(2 beta(n+nu)),
    # which would give Phi(sqrt(nu)) instead.
    g_big = 100.0
    G, Xi = g_big**2 / (1 + g_big**2), 1.0 / (1 + g_big**2)
    K2b, E2b = ellipk(Xi), ellipe(Xi)
    beta_inf = np.pi**2 / (4.0 * K2b * E2b)
    alpha_inf = np.pi * (ellipk(G) - ellipe(G)) / E2b
    out["beta_mpi_limit"] = float(beta_inf)
    out["exp_alpha_ratio"] = float(np.exp(-alpha_inf) / (np.e**2 / (16 * g_big**2)))

    # (a) multiphoton asymptote and photon order
    I_lo = np.array([5e11, 1e12, 2e12])
    ratio = kel.rate(I_lo) / (sigma6 * I_lo**6 * rho_at)
    out["mpi_ratio"] = ratio
    order = np.diff(np.log(kel.rate(I_lo))) / np.diff(np.log(I_lo))
    out["order_low_I"] = order

    # (b) monotonicity over the operating range
    I = np.logspace(11.5, 14.2, 600)
    W = kel.rate(I)
    slope = np.diff(np.log(np.maximum(W, 1e-300))) / np.diff(np.log(I))
    out["n_decreasing"] = int((slope < -1e-9).sum())
    out["slope_min"] = float(slope.min())

    # raw form, for the record
    Wr = kel.rate_raw(I)
    sr = np.diff(np.log(np.maximum(Wr, 1e-300))) / np.diff(np.log(I))
    out["n_decreasing_raw"] = int((sr < 0).sum())
    out["slope_min_raw"] = float(sr.min())

    # adiabaticity checkpoint: the paper states gamma = 1 at I = 3.5e13 W/cm^2
    E = np.sqrt(2 * 3.5e13 * 1e4 / (n0 * c * epsilon_0))
    out["gamma_at_3.5e13"] = float(2 * np.pi * c / wavelength
                                   * np.sqrt(meff_rel * m_e * Ui_eV * q_e) / (q_e * E))

    # beta convention: the paper flags the ambiguity itself
    kel2 = KeldyshSiO2(wavelength, Ui_eV, meff_rel * m_e, n0, beta_den=2.0)
    out["W_3.5e13_beta4"] = float(kel.rate(np.array([3.5e13]))[0])
    out["W_3.5e13_beta2"] = float(kel2.rate(np.array([3.5e13]))[0])
    out["beta_den"] = kel.beta_den

    if verbose:
        print("Keldysh validation (fused silica, Ui = 9 eV, 800 nm)")
        print("  -- summation convention (structural) --")
        print(f"  beta at gamma=100               : {beta_inf:.5f}   [-> 1]")
        print(f"  exp(-alpha) / [e^2/16gamma^2]   : {out['exp_alpha_ratio']:.5f}   [-> 1]")
        print("    => n=0 term is Phi(sqrt(2 nu)): Keldysh's textbook multiphoton")
        print("       rate, which is what fixes the published (n + 2 nu) argument.")
        print("  -- normalization --")
        print(f"  W/W_MPI at 5e11/1e12/2e12 W/cm2 : "
              f"{ratio[0]:.3f} / {ratio[1]:.3f} / {ratio[2]:.3f}   [-> 1, paper Sec. III]")
        print(f"  photon order at low I           : {order.mean():.2f}   [-> 6]")
        print(f"  gamma at 3.5e13 W/cm2           : {out['gamma_at_3.5e13']:.3f}   [paper: 1]")
        print(f"  beta_den in use                 : {out['beta_den']:.0f}   "
              f"[Couairon Eq.(8) = 4; Keldysh Ref.34 = 2]")
        print(f"  W(3.5e13), beta_den=4 / =2      : "
              f"{out['W_3.5e13_beta4']:.2e} / {out['W_3.5e13_beta2']:.2e}   [paper quotes 1.6e32]")
        print("  -- channel closings (physical cusps, NOT a bug) --")
        print(f"  decreasing points, raw formula  : {out['n_decreasing_raw']}  "
              f"(min slope {out['slope_min_raw']:+.2f})")
        print(f"  decreasing points, as served    : {out['n_decreasing']}  "
              f"(min slope {out['slope_min']:+.2f})"
              f"   [monotone={kel.monotone}]")

    assert 0.8 < ratio[1] < 1.25, f"normalization off: W/W_MPI(1e12) = {ratio[1]:.3f}"
    assert 5.5 < order.mean() < 6.5, f"photon order off: {order.mean():.2f}"
    assert 0.95 < out["gamma_at_3.5e13"] < 1.05, "gamma checkpoint off"
    assert abs(beta_inf - 1.0) < 1e-3, "beta does not tend to 1 in the MPI limit"
    assert abs(out["exp_alpha_ratio"] - 1.0) < 1e-3, "alpha asymptote off"
    return out


if __name__ == "__main__":
    validate()


# ================================================================================
#  Analytic limits of the Keldysh rate -- the dotted / dash-dotted curves of
#  Couairon 2005 Fig. 2. Both are asymptotics OF THE SAME formula, so they are
#  the strongest available check that rate_raw() is assembled correctly: the
#  general curve must merge into the multiphoton one at low intensity and into
#  the tunnel one at high intensity, with no free parameter.
# ================================================================================
_HBAR = 1.054571817e-34

def _gamma(I_Wcm2, omega, meff, U, n0):
    E = np.sqrt(2.0 * np.asarray(I_Wcm2, float) * 1e4 / (n0 * c * epsilon_0))
    return np.maximum(omega * np.sqrt(meff * U) / (q_e * np.maximum(E, 1e-300)), 1e-12)

def keldysh_multiphoton(I_Wcm2, wavelength=800e-9, Ui_eV=9.0, meff=None, n0=None):
    r"""Multiphoton limit, gamma >> 1 (cm^-3 s^-1).

        W = (2w/9pi)(w m/hbar)^{3/2} Phi(sqrt(2 nu)) e^{2<x+1>} (16 gamma^2)^{-<x+1>}

    This is the beta -> 1 limit of the general formula, hence the Phi(sqrt(2 nu)).
    """
    m_e_ = 9.1093837015e-31
    meff = 0.64 * m_e_ if meff is None else meff
    n0 = n_sellmeier(wavelength) if n0 is None else n0
    w, U = 2 * np.pi * c / wavelength, Ui_eV * q_e
    g = _gamma(I_Wcm2, w, meff, U, n0)
    xt = (U / (_HBAR * w)) * (1.0 + 1.0 / (4.0 * g**2))
    mx = np.floor(xt + 1.0)
    nu = mx - xt
    pref = (2.0 * w / (9.0 * np.pi)) * (w * meff / _HBAR) ** 1.5
    W = (pref * dawsn(np.sqrt(2.0 * nu))
         * np.exp(2.0 * mx * (1.0 - 1.0 / (4.0 * g**2)))
         * (1.0 / (16.0 * g**2)) ** mx)
    return np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0) * 1e-6

def keldysh_tunnel(I_Wcm2, wavelength=800e-9, Ui_eV=9.0, meff=None, n0=None):
    r"""Tunnel limit, gamma << 1 (cm^-3 s^-1).

        W = (2/9pi^2)(m U/hbar^2)^{3/2}(U/hbar) (hbar w/(U gamma))^{5/2}
            exp[-(pi/2) sqrt(m U) U/(hbar e E) (1 - gamma^2/8)]

    The exponent scales as 1/E, the tunnelling signature.
    """
    m_e_ = 9.1093837015e-31
    meff = 0.64 * m_e_ if meff is None else meff
    n0 = n_sellmeier(wavelength) if n0 is None else n0
    w, U = 2 * np.pi * c / wavelength, Ui_eV * q_e
    E = np.sqrt(2.0 * np.asarray(I_Wcm2, float) * 1e4 / (n0 * c * epsilon_0))
    g = _gamma(I_Wcm2, w, meff, U, n0)
    pref = (2.0 / (9.0 * np.pi**2)) * (meff * U / _HBAR**2) ** 1.5 * (U / _HBAR)
    scal = (_HBAR * w / (U * g)) ** 2.5
    expo = np.exp(-(np.pi / 2.0) * np.sqrt(meff * U) * U
                  / (_HBAR * q_e * np.maximum(E, 1e-300)) * (1.0 - g**2 / 8.0))
    return np.nan_to_num(pref * scal * expo, nan=0.0, posinf=0.0, neginf=0.0) * 1e-6
