"""
================================================================================
 Femtosecond filamentation in bulk fused silica (SiO2) -- MAIN ENTRY POINT
 Faithful to Couairon, Sudrie, Franco, Prade, Mysyrowicz,
   PRB 71, 125435 (2005).  SI units throughout.

 This module is the façade: it wires the pieces together and exposes run().
 It is what the notebook imports, and importing it gives access to everything
 (Config, build_grids, the operators, the Integrator), so existing code and
 notebooks keep working unchanged after the split into modules.

   config.py      Config dataclass (every tunable) + code_fingerprint()
   keldysh.py     Keldysh photoionization rate, its analytic limits, Sellmeier
   kernels.py     CUDA kernel for the carrier rate equations (eq. 6-7)
   grids.py       Hankel/time/frequency grids, dispersion, LUTs, envelopes
   operators.py   linear half step + nonlinear term (eq. 3)
   integrator.py  recording buffers, z-marching, result.npz / params.json
   filament_sim.py  <- you are here: run() and the public surface

 The nonlinear term is split into six independently switchable physics
 channels so that ablation studies ("what does this term actually do to the
 filament?") can be run from a notebook without touching the solver internals:

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

import sys
from pathlib import Path

# Package modules live beside this file; inserting the directory explicitly
# means the imports work regardless of the caller's sys.path/cwd setup.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config, code_fingerprint                    # noqa: E402,F401
from keldysh import (SELLMEIER_B, SELLMEIER_L2, n_sellmeier,   # noqa: E402,F401
                     KeldyshSiO2, keldysh_multiphoton, keldysh_tunnel)
from kernels import rate_eq_kernel                             # noqa: E402,F401
from grids import build_grids, envelope_gaussian_focused, ENVELOPES  # noqa: E402,F401
from operators import LinearOperator, NonlinearOperator        # noqa: E402,F401
from integrator import Integrator                              # noqa: E402,F401

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


__all__ = [
    "run", "FIELD_TOGGLES", "Config", "code_fingerprint",
    "build_grids", "envelope_gaussian_focused", "ENVELOPES",
    "LinearOperator", "NonlinearOperator", "Integrator",
    "n_sellmeier", "KeldyshSiO2", "keldysh_multiphoton", "keldysh_tunnel",
    "rate_eq_kernel", "SELLMEIER_B", "SELLMEIER_L2",
]
