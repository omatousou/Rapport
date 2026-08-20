"""
Equivalence test for the term-registry rewrite of NonlinearOperator.split().

split() used to be one block of code with the six field terms written out by
hand. It is now a loop over the FIELD_TERMS registry. This file proves the
rewrite changed no physics, by comparing the new split() against a frozen,
verbatim copy of the old one on random inputs, for every one of the 64
combinations of the six field flags.

It runs on the CPU. split() only uses functions that numpy has too, so cupy is
replaced by a numpy shim at import time and no GPU is needed. Run it with

    python sim/test_operators_equivalence.py

The comparison target is the whole return value of split(), which is enough:
over one z step the two exp(-alpha dz/2) factors of Integrator.step contribute
-alpha*u and the trailing + alpha*u inside split() cancels them, so the entire
nonlinear contribution to du/dz is ifft(NL_freq * inv_U_nl). Two split()
implementations that return the same (rhs, alpha) therefore integrate the same
equation, and nothing else about the solver needs to be run.
"""

import itertools
import sys
import types
from pathlib import Path

import numpy as np

# ---- numpy stands in for cupy ------------------------------------------------
# kernels.py builds a cp.RawKernel at import time, so the shim needs that name
# even though the kernel is never launched here.
if "cupy" not in sys.modules:
    _shim = types.ModuleType("cupy")
    _shim.__dict__.update(np.__dict__)
    _shim.RawKernel = lambda *a, **k: None
    _shim.asnumpy = lambda x: np.asarray(x)
    sys.modules["cupy"] = _shim

import cupy as cp  # noqa: E402  (the shim)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scipy.constants import c, epsilon_0  # noqa: E402
from config import Config                 # noqa: E402
import operators                          # noqa: E402


# ================================================================================
#  The old implementation, copied verbatim from before the rewrite.
# ================================================================================
#  Do not tidy this up and do not make it call the registry. Its only job is to
#  be the version that produced the published figures. It reads the same
#  attributes the old NonlinearOperator.__init__ set, which are rebuilt by
#  _legacy_flags() below.
def reference_split(op, en, u, rho, rho_s=None):
    u = cp.ascontiguousarray(u.astype(cp.complex128, copy=False))
    absu2 = cp.abs(u)**2
    W_PI = cp.nan_to_num(cp.abs(op.f_spline(
        cp.clip(absu2 * op.invE2, op.Imin, op.Imax))),
        nan=0.0, posinf=0.0, neginf=0.0)

    depl_field = cp.clip(1.0 - rho / op.rho_max, 0.0, 1.0)
    photo = (W_PI * 1e6) * op.Ui / (op.n0 * c * epsilon_0 * (absu2 + 1e-30)) * depl_field
    if not en["enable_photoionization_loss"]:
        photo = photo * 0.0

    alpha = photo
    if en["enable_plasma_absorption"]:
        alpha = alpha + op.plasma_pref * rho
    alpha = cp.maximum(cp.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0), 0.0)

    kerr_I = cp.zeros_like(absu2)
    if en["enable_kerr_instantaneous"]:
        kerr_I = kerr_I + (1.0 - op.f_R) * absu2
    if en["enable_kerr_raman"]:
        kerr_I = kerr_I + op.f_R * cp.fft.ifft(cp.fft.fft(absu2, axis=1) * op.R_f, axis=1).real

    plasma_coeff = ((1.0 if en["enable_plasma_absorption"] else 0.0)
                    + ((op.plasma_phase - 1.0) if en["enable_plasma_defocusing"] else 0.0))
    extra = -op.plasma_pref * plasma_coeff * rho * u
    if op.ste_pref and rho_s is not None and en["enable_ste_index"]:
        extra = extra + 1j * op.ste_pref * rho_s * u

    NL_freq = (cp.fft.fft(1j * op.kerr_pref * kerr_I * u, axis=1) * op.T_op**2
               - cp.fft.fft(photo * u, axis=1) * op.T_op
               + cp.fft.fft(extra, axis=1))
    rhs = cp.fft.ifft(NL_freq * op.inv_U_nl, axis=1) + alpha * u
    rhs = cp.nan_to_num(cp.where(absu2 < 1e-30, 0.0 + 0.0j, rhs),
                        nan=0.0, posinf=0.0, neginf=0.0)
    return rhs, alpha


# ================================================================================
#  A NonlinearOperator built on synthetic grids
# ================================================================================
FLAGS = ("enable_kerr_instantaneous", "enable_kerr_raman",
         "enable_photoionization_loss", "enable_plasma_absorption",
         "enable_plasma_defocusing", "enable_ste_index")


def make_operator(flags, Nr=6, Nt=32, seed=0, ste_on=True):
    """A real NonlinearOperator on a small synthetic grid.

    build_grids needs cupyx.scipy.special, which the numpy shim does not
    provide, so the grid dict is synthesized instead. The arrays only have to
    be the right shape and finite: both implementations see the same ones.
    """
    rng = np.random.default_rng(seed)
    cfg = Config(**{f: bool(v) for f, v in zip(FLAGS, flags)},
                 enable_ste=ste_on, verbose=False)

    # A smooth, positive, strongly nonlinear stand-in for the Keldysh spline.
    def f_spline(I):
        return 1e20 * (np.asarray(I) / 1e13) ** 5

    T_op = 1.0 + 0.4 * rng.standard_normal(Nt)
    g = dict(
        komega=cfg.komega,
        T_op=T_op,
        R_f=rng.standard_normal(Nt) + 1j * rng.standard_normal(Nt),
        invE2=0.5 * cfg.n0 * c * epsilon_0 * 1e-4,
        inv_U_nl=1.0 + 0.3 * rng.standard_normal(Nt),
        sigmaomega=1.1669e-17,
        ste_pref=(1.7913e-16 if ste_on else 0.0),
        keldysh=dict(f_spline=f_spline, Imin=1.0, Imax=1e17),
        avalanche_coef=0.0, avalanche_coef_s=0.0,
        inv_taur_eff=0.0, inv_tau_ste=0.0,
    )
    return operators.NonlinearOperator(cfg, g)


def make_inputs(op, Nr=6, Nt=32, seed=1):
    """Fields and densities in the range the solver actually visits."""
    rng = np.random.default_rng(seed)
    # |u|^2 * invE2 spans roughly 1e11 to 1e14 W/cm^2.
    amp = np.sqrt(10.0 ** rng.uniform(11.0, 14.0, size=(Nr, Nt)) / op.invE2)
    u = amp * np.exp(1j * rng.uniform(0, 2 * np.pi, size=(Nr, Nt)))
    rho = 10.0 ** rng.uniform(14.0, 21.5, size=(Nr, Nt))
    rho_s = 10.0 ** rng.uniform(14.0, 20.5, size=(Nr, Nt))
    return u, rho, rho_s


def relerr(a, b):
    """Max relative difference, scaled by the magnitude of the reference."""
    a, b = np.asarray(a), np.asarray(b)
    scale = np.max(np.abs(b))
    if scale == 0.0:
        return float(np.max(np.abs(a)))
    return float(np.max(np.abs(a - b)) / scale)


# ================================================================================
#  The test
# ================================================================================
def main(tol=1e-14):
    worst_rhs, worst_alpha, worst_case = 0.0, 0.0, None
    n_exact = 0
    cases = 0

    for ste_on in (True, False):
        for flags in itertools.product((True, False), repeat=len(FLAGS)):
            op = make_operator(flags, ste_on=ste_on)
            u, rho, rho_s = make_inputs(op)
            en = dict(zip(FLAGS, flags))

            rhs_new, alpha_new = op.split(u, rho, rho_s)
            rhs_ref, alpha_ref = reference_split(op, en, u, rho, rho_s)

            e_rhs = relerr(rhs_new, rhs_ref)
            e_alpha = relerr(alpha_new, alpha_ref)
            cases += 1
            if e_rhs == 0.0 and e_alpha == 0.0:
                n_exact += 1
            if e_rhs > worst_rhs:
                worst_rhs, worst_case = e_rhs, (ste_on, en)
            worst_alpha = max(worst_alpha, e_alpha)

    # rho_s absent: the STE term must contribute nothing, on both paths.
    op = make_operator((True,) * len(FLAGS))
    u, rho, _ = make_inputs(op)
    en = dict.fromkeys(FLAGS, True)
    rhs_new, _ = op.split(u, rho, None)
    rhs_ref, _ = reference_split(op, en, u, rho, None)
    e_none = relerr(rhs_new, rhs_ref)

    # loss_rates must still mirror split(), channel by channel.
    photo, plasma = op.loss_rates(u, rho)
    _, alpha_split = op.split(u, rho, None)
    e_loss = relerr(photo + plasma, alpha_split)

    print(f"cases compared            : {cases}")
    print(f"bit-identical             : {n_exact} / {cases}")
    print(f"worst relative diff, rhs  : {worst_rhs:.3e}")
    print(f"worst relative diff, alpha: {worst_alpha:.3e}")
    print(f"rho_s=None path           : {e_none:.3e}")
    print(f"loss_rates vs alpha       : {e_loss:.3e}")
    if worst_case is not None and worst_rhs > 0:
        ste_on, en = worst_case
        on = [k.replace("enable_", "") for k, v in en.items() if v]
        print(f"worst case                : enable_ste={ste_on}, on={on or ['nothing']}")

    ok = max(worst_rhs, worst_alpha, e_none, e_loss) <= tol
    print("\nRESULT:", "PASS" if ok else "FAIL", f"(tolerance {tol:.0e})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
