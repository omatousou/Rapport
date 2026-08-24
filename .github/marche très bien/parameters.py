"""
parameters.py -- every number of the experiment, in one editable place.

The solver takes all its parameters as arguments, so nothing here is magic.
These are named bundles of values, and `simulate_kwargs()` turns a set of
bundles into the keyword arguments of `simulate()`.

    from parameters import MATERIAL, LASER, BOX, GRID, simulate_kwargs
    from run_filament import simulate

    res = simulate(**simulate_kwargs(MATERIAL, LASER, BOX, GRID))

Edit a value by editing it here. Nothing else reads these numbers, so a change
here is the only change needed.

Override without editing, for a one-off:

    res = simulate(**simulate_kwargs(MATERIAL, LASER, BOX, GRID,
                                     energy_incident_uJ=2.0,
                                     enable_kerr_raman=False))

Make a variant, since the bundles are frozen dataclasses:

    import dataclasses
    softer = dataclasses.replace(MATERIAL, n2_m2W=2.4e-20)

Run `python parameters.py` to print every value with where it comes from, and
to check that the bundles still match the solver's signature.


WHY THE SOURCES ARE IN HERE
===========================
A number in a simulation is worth little without knowing where it came from.
Each bundle carries a `sources` dict keyed by field name. Three of the silica
values are genuinely disputed in the literature, and the code has to pick one:
see the notes on `tau_c_s`, `meff_drude_rel` and `n2_m2W`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace   # noqa: F401  (replace is for callers)
from typing import Dict, Optional, Tuple

import numpy as np


# ================================================================================
#  MATERIAL
# ================================================================================
@dataclass(frozen=True)
class Material:
    """Everything that changes when you change the sample.

    `sellmeier_*` is carried here because it IS a property of the material, but
    be aware that it is not yet plumbed through to the solver: the dispersion
    is still a module constant in sim/keldysh.py. `simulate_kwargs()` refuses
    to build arguments for a material whose Sellmeier fit differs from the one
    the solver has compiled in, rather than silently running the wrong
    dispersion. See the note at the bottom of this file.
    """

    name: str

    # ---- linear dispersion -------------------------------------------------------
    # n^2 - 1 = sum_i B_i lam^2 / (lam^2 - L_i^2),  lam in micrometres
    sellmeier_B: Tuple[float, ...]
    sellmeier_L_um: Tuple[float, ...]
    sellmeier_range_um: Tuple[float, float]

    # ---- Kerr ---------------------------------------------------------------------
    n2_m2W: float          # nonlinear index
    f_R: float             # fraction of the Kerr response that is delayed
    tau_d_s: float         # Raman damping time
    tau_s_s: float         # Raman oscillation period

    # ---- photoionization ------------------------------------------------------------
    Ui_eV: float           # band gap, sets the Keldysh rate W_PI
    meff_rel: float        # reduced mass in the Keldysh rate, in units of m_e
    rho_max_cm3: float     # N_at, density of ionizable units

    # ---- Drude response of the free carriers ------------------------------------------
    meff_drude_rel: float  # effective mass in sigma_w, in units of m_e
    tau_c_s: float         # electron collision time

    # ---- self-trapped excitons ----------------------------------------------------------
    has_ste: bool
    Us_eV: Optional[float] = None       # gap seen by a trapped exciton
    E_tr_eV: Optional[float] = None     # STE resonance, sets the pump-side index
    tau_r_s: Optional[float] = None     # trapping time, N -> N_STE
    tau_ste_s: Optional[float] = None   # STE decay to the ground state

    # ---- entrance face -------------------------------------------------------------------
    n_fresnel: float = 1.45

    sources: Dict[str, str] = field(default_factory=dict)
    notes: str = ""

    @property
    def sellmeier_L2(self) -> Tuple[float, ...]:
        return tuple(float(x) ** 2 for x in self.sellmeier_L_um)

    def n(self, lam_m: float) -> float:
        """Linear index at `lam_m`, from this material's Sellmeier fit."""
        lam_um = lam_m * 1e6
        lo, hi = self.sellmeier_range_um
        if not (lo <= lam_um <= hi):
            raise ValueError(
                f"{self.name}: {lam_um:.3f} um is outside the Sellmeier window "
                f"[{lo}, {hi}] um, the index would be extrapolated")
        n2m1 = sum(B * lam_um**2 / (lam_um**2 - L2)
                   for B, L2 in zip(self.sellmeier_B, self.sellmeier_L2))
        return float(np.sqrt(1.0 + n2m1))


FUSED_SILICA = Material(
    name="fused silica (SiO2)",

    sellmeier_B=(0.6961663, 0.4079426, 0.8974794),
    sellmeier_L_um=(0.0684043, 0.1162414, 9.896161),
    sellmeier_range_um=(0.18, 5.0),

    n2_m2W=2.74e-20,
    f_R=0.18,
    tau_d_s=32e-15,
    tau_s_s=12e-15,

    Ui_eV=9.0,
    meff_rel=0.64,
    rho_max_cm3=2.1e22,

    meff_drude_rel=1.0,
    tau_c_s=1.7e-15,

    has_ste=True,
    Us_eV=6.0,
    E_tr_eV=4.2,
    tau_r_s=330e-15,
    tau_ste_s=1e-12,

    n_fresnel=1.45,

    sources={
        "sellmeier_B": "Malitson, JOSA 55, 1205 (1965)",
        "n2_m2W": "used by the runs of this repository; Couairon 2005 uses 3.54e-20 at 800 nm",
        "f_R": "Couairon et al., PRB 71, 125435 (2005), delayed Raman fraction",
        "tau_d_s": "Couairon 2005, Raman damping time",
        "tau_s_s": "Couairon 2005, Raman oscillation period",
        "Ui_eV": "Couairon 2005, band gap",
        "meff_rel": "Couairon 2005, reduced mass in the Keldysh rate",
        "rho_max_cm3": "molecular density of SiO2",
        "meff_drude_rel": "bare electron mass; Martin et al. 1997 Table II give 0.5",
        "tau_c_s": "Bulgakova convention; Couairon 2005 fit 10 fs, Martin 1997 give 0.67 fs",
        "Us_eV": "re-ionization of a trapped exciton, Mao et al., Appl. Phys. A 79, 1695 (2004)",
        "E_tr_eV": "STE resonance, Martin et al., PRB 55, 5799 (1997), Table II",
        "tau_r_s": "trapping into the STE state",
        "tau_ste_s": "non-radiative STE decay, Sakurai et al.",
        "n_fresnel": "index at 1030 nm, for the entrance-face transmission",
    },
    notes="""
Every published figure in this repository was produced with these numbers, so
changing one changes results you may want to compare against.

Three of them are disputed and the code simply has to pick one.

tau_c_s is 1.7 fs here, 10 fs in Couairon 2005 and 0.67 fs in Martin 1997. All
three fit the same physical quantity to a different measurement. It sets
sigma_w, so it drives both plasma absorption and defocusing.

meff_drude_rel is the bare mass here, 0.5 in Martin 1997. Do not confuse it
with meff_rel, which is the reduced mass in the Keldysh rate and is a
different quantity.

n2_m2W is 2.74e-20 here against Couairon's 3.54e-20 at 800 nm.
""",
)


# ================================================================================
#  LASER
# ================================================================================
@dataclass(frozen=True)
class Laser:
    """Pump pulse and probe wavelengths."""
    name: str
    wavelength_m: float
    energy_incident_uJ: float      # energy BEFORE the entrance face
    spot_sx_um: float              # measured spot sizes, w0 = sqrt(sx*sy)
    spot_sy_um: float
    delta_t_s: float               # FWHM of the intensity
    probe_wavelengths_nm: Tuple[float, ...]
    apply_fresnel: bool = True
    sources: Dict[str, str] = field(default_factory=dict)

    @property
    def w0_m(self) -> float:
        return float(np.sqrt(self.spot_sx_um * self.spot_sy_um) * 1e-6)


PUMP_1030_4UJ = Laser(
    name="1030 nm, 4 uJ, 263 fs",
    wavelength_m=1030e-9,
    energy_incident_uJ=4.0,
    spot_sx_um=11.5,
    spot_sy_um=11.0,
    delta_t_s=263e-15,
    probe_wavelengths_nm=(490.0, 620.0, 690.0),
    apply_fresnel=True,
    sources={
        "spot_sx_um": "measured beam profile, w0 = sqrt(sx*sy) = 11.25 um",
        "delta_t_s": "FWHM of the intensity; the solver uses tp = FWHM/sqrt(2 ln2)",
        "energy_incident_uJ": "before the sample; apply_fresnel removes the reflected part",
        "probe_wavelengths_nm": "probe crosses the pump at 90 degrees, Nomarski interferometry",
    },
)


# ================================================================================
#  BOX AND GRID
# ================================================================================
@dataclass(frozen=True)
class Geometry:
    """Where the simulation box starts and ends, in the medium."""
    name: str
    begin_m: float
    end_m: float
    z_focus_air_um: Optional[float] = None   # overrides begin_m when set


BOX_350UM = Geometry(name="0 to 350 um", begin_m=0.0, end_m=350e-6)


@dataclass(frozen=True)
class Numerics:
    """Grid and output sampling. No physics here.

    tmax_factor sets the comoving window, tmax = tmax_factor * tp. At the
    historical 5.0 it covers only about +/-1.1 ps for a 263 fs pulse, too short
    to see the STE decay of roughly 1 ps: past tmax the recorded cube has no
    data and the HTML cursor plateaus. Raise Nt with it to keep the same dt.
    """
    name: str
    Nz: int
    Nt: int
    Nr: int
    R_factor: float          # R_max = R_factor * w0
    tmax_factor: float       # tmax = tmax_factor * tp
    save_stride: int
    rho_t_stride: int        # 0 disables the (z, r, t) cube entirely
    rho_r_stride: int
    ckpt_every: int = 200


GRID_PRODUCTION = Numerics(
    name="production", Nz=9000, Nt=4096, Nr=1024, R_factor=8.0,
    tmax_factor=10.0, save_stride=20, rho_t_stride=8, rho_r_stride=2)

GRID_QUICK = Numerics(
    name="quick look", Nz=1500, Nt=2048, Nr=512, R_factor=8.0,
    tmax_factor=10.0, save_stride=10, rho_t_stride=16, rho_r_stride=4)


# ================================================================================
#  HTML OUTPUT
# ================================================================================
@dataclass(frozen=True)
class HtmlOptions:
    """Options of the interactive explorer pages.

    t_step_fs is the delay cursor step, NOT the resolution of the recorded
    cube. None would put every cube instant in the page, several hundred of
    them, each carrying a phase map and a density map, and the file would run
    into gigabytes.
    """
    t_step_fs: float = 67.0
    coarsen_z: int = 4
    coarsen_r: int = 1
    phase_clip: float = 0.2
    t_min: Optional[float] = None   # None = transmittance colorbar floor auto
    z_lim_um: Tuple[float, float] = (0.0, 350.0)
    x_lim_um: Tuple[float, float] = (-50.0, 50.0)


HTML_DEFAULT = HtmlOptions()


# ================================================================================
#  PHYSICS SWITCHES
# ================================================================================
@dataclass(frozen=True)
class Physics:
    """Which terms of the two equations are integrated.

    The field flags act only on the propagation equation. Switching one off
    stops that channel acting on the beam, it does not stop carriers being
    created: that is what makes an ablation study possible, and it also means
    such a run does not conserve energy by construction.
    """
    # field equation
    enable_kerr_instantaneous: bool = True
    enable_kerr_raman: bool = True
    enable_self_steepening: bool = True
    enable_photoionization_loss: bool = True
    enable_plasma_absorption: bool = True
    enable_plasma_defocusing: bool = True
    enable_ste_index: bool = True
    enable_space_time_focusing: bool = True
    enable_spectral_filter: bool = True
    # carrier equations
    enable_avalanche: bool = True
    enable_recombination: bool = True
    enable_ste: bool = True


PHYSICS_ALL_ON = Physics()


# ================================================================================
#  Turning bundles into simulate() arguments
# ================================================================================
def simulate_kwargs(material: Material = FUSED_SILICA,
                    laser: Laser = PUMP_1030_4UJ,
                    geometry: Geometry = BOX_350UM,
                    numerics: Numerics = GRID_PRODUCTION,
                    html: HtmlOptions = HTML_DEFAULT,
                    physics: Physics = PHYSICS_ALL_ON,
                    check_dispersion: bool = True,
                    **overrides) -> dict:
    """Flatten the bundles into the keyword arguments of `simulate()`.

    Anything passed as a keyword wins over the bundles, so a one-off variation
    needs no new bundle.

    `check_dispersion` compares the material's Sellmeier fit against the one
    compiled into the solver and raises if they differ, because the solver
    would otherwise run the wrong dispersion without saying so. Pass False only
    if you have wired the new coefficients in yourself.
    """
    if check_dispersion:
        _assert_dispersion_matches(material)

    kw = dict(
        # pump
        wavelength_m=laser.wavelength_m,
        energy_incident_uJ=laser.energy_incident_uJ,
        apply_fresnel=laser.apply_fresnel,
        n_glass_fresnel=material.n_fresnel,
        spot_sx_um=laser.spot_sx_um,
        spot_sy_um=laser.spot_sy_um,
        delta_t_s=laser.delta_t_s,

        # material
        n2_m2W=material.n2_m2W,
        Ui_eV=material.Ui_eV,
        meff_rel=material.meff_rel,
        meff_drude_rel=material.meff_drude_rel,
        tau_c_s=material.tau_c_s,
        rho_max_cm3=material.rho_max_cm3,
        f_R=material.f_R,
        tau_d_s=material.tau_d_s,
        tau_s_s=material.tau_s_s,

        # box
        begin_m=geometry.begin_m,
        end_m=geometry.end_m,
        z_focus_air_um=geometry.z_focus_air_um,

        # grid
        Nz=numerics.Nz, Nt=numerics.Nt, Nr=numerics.Nr,
        R_factor=numerics.R_factor, tmax_factor=numerics.tmax_factor,
        save_stride=numerics.save_stride,
        rho_t_stride=numerics.rho_t_stride,
        rho_r_stride=numerics.rho_r_stride,
        ckpt_every=numerics.ckpt_every,

        # probes and HTML
        probe_wavelengths_nm=laser.probe_wavelengths_nm,
        html_t_step_fs=html.t_step_fs,
        html_coarsen_z=html.coarsen_z,
        html_coarsen_r=html.coarsen_r,
        html_phase_clip=html.phase_clip,
        html_t_min=html.t_min,
        html_z_lim_um=html.z_lim_um,
        html_x_lim_um=html.x_lim_um,
    )

    # STE constants only when the material has the channel.
    if material.has_ste:
        kw.update(Us_eV=material.Us_eV, E_tr_eV=material.E_tr_eV,
                  tau_r_s=material.tau_r_s, tau_ste_s=material.tau_ste_s)
    else:
        # tau_r still has to be a number because it appears in the rate
        # equations, so the recombination term is switched off with the rest
        # rather than fed a made-up time constant.
        kw.update(Us_eV=material.Ui_eV, E_tr_eV=1.0,
                  tau_r_s=1.0, tau_ste_s=None)

    for f in Physics.__dataclass_fields__:
        kw[f] = getattr(physics, f)
    if not material.has_ste:
        kw.update(enable_ste=False, enable_ste_index=False,
                  enable_recombination=False)

    kw.update(overrides)
    return kw


def _assert_dispersion_matches(material: Material) -> None:
    """Refuse to run a material whose Sellmeier fit the solver does not have."""
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent / "sim"))
        import keldysh
    except ImportError:
        return    # solver not importable from here, nothing to compare against

    same = (np.allclose(np.asarray(material.sellmeier_B, float),
                        np.asarray(keldysh.SELLMEIER_B, float))
            and np.allclose(np.asarray(material.sellmeier_L2, float),
                            np.asarray(keldysh.SELLMEIER_L2, float)))
    if not same:
        raise ValueError(
            f"{material.name} carries a Sellmeier fit that the solver does not "
            f"have compiled in.\n"
            f"  parameters.py : B = {tuple(material.sellmeier_B)}\n"
            f"  sim/keldysh.py: B = {tuple(float(x) for x in keldysh.SELLMEIER_B)}\n"
            f"\n"
            f"The dispersion is still a module constant, so the solver would run\n"
            f"the wrong one without saying so. To use this material, change\n"
            f"SELLMEIER_B and SELLMEIER_L2 in sim/keldysh.py and the copy in\n"
            f"web/abel_phase_explorer.py, then pass check_dispersion=False.\n"
            f"See NOT YET WIRED at the bottom of parameters.py.")


# ================================================================================
#  Defaults, so a caller can write simulate(**simulate_kwargs())
# ================================================================================
MATERIAL = FUSED_SILICA
LASER    = PUMP_1030_4UJ
BOX      = BOX_350UM
GRID     = GRID_PRODUCTION
HTML     = HTML_DEFAULT
PHYSICS  = PHYSICS_ALL_ON


# ================================================================================
#  NOT YET WIRED
# ================================================================================
#  Three numbers of the experiment are NOT read from this file, because they
#  are still module constants inside the solver. Changing them here alone will
#  do nothing, which is why simulate_kwargs() checks the first one and raises.
#
#  1. The Sellmeier coefficients, in sim/keldysh.py lines 22-23. They are also
#     duplicated in web/abel_phase_explorer.py lines 91-92, so a change has to
#     be made twice.
#
#  2. The Sellmeier validity window, hardcoded as 0.18 to 5 um in
#     sim/grids.py, both in the omega_safe clip and in the spectral mask.
#
#  3. The probe-side dielectric model of sim/permittivity.py, whose defaults
#     are Table II of Martin et al. 1997 for SiO2: valence density 2.2e22,
#     conduction mass 0.5, electron-phonon time 0.67 fs, and the two STE bands
#     at 5.2 and 4.2 eV. That file takes them as a dataclass, so they are
#     already editable, but they live there and not here.
#
#  Wiring them through is a small change to four files. It has not been done
#  because it touches the solver, and nothing so far needed it.
# ================================================================================


def _main():
    import textwrap

    def block(title):
        print("\n" + "=" * 78)
        print(title)
        print("=" * 78)

    block("MATERIAL")
    m = MATERIAL
    print(f"  {m.name}\n")
    rows = [("n at 1030 nm", f"{m.n(1030e-9):.4f}", None),
            ("n2", f"{m.n2_m2W:.3e} m2/W", "n2_m2W"),
            ("band gap Ui", f"{m.Ui_eV:.2f} eV", "Ui_eV"),
            ("reduced mass", f"{m.meff_rel:.3f} m_e", "meff_rel"),
            ("Drude mass", f"{m.meff_drude_rel:.3f} m_e", "meff_drude_rel"),
            ("collision time", f"{m.tau_c_s*1e15:.2f} fs", "tau_c_s"),
            ("N_at", f"{m.rho_max_cm3:.3e} cm-3", "rho_max_cm3"),
            ("Raman fraction", f"{m.f_R:.3f}", "f_R"),
            ("Raman damping", f"{m.tau_d_s*1e15:.0f} fs", "tau_d_s"),
            ("Raman period", f"{m.tau_s_s*1e15:.0f} fs", "tau_s_s"),
            ("STE gap Us", f"{m.Us_eV:.2f} eV", "Us_eV"),
            ("STE resonance", f"{m.E_tr_eV:.2f} eV", "E_tr_eV"),
            ("trapping time", f"{m.tau_r_s*1e15:.0f} fs", "tau_r_s"),
            ("STE decay", f"{m.tau_ste_s*1e15:.0f} fs", "tau_ste_s")]
    for label, value, key in rows:
        print(f"  {label:16s} {value:>16s}    {m.sources.get(key, '') if key else 'derived'}")
    print(textwrap.indent(m.notes.strip(), "  "))

    block("LASER")
    L = LASER
    print(f"  {L.name}")
    print(f"  w0               = {L.w0_m*1e6:.2f} um   from sx={L.spot_sx_um}, sy={L.spot_sy_um} um")
    print(f"  energy incident  = {L.energy_incident_uJ} uJ")
    print(f"  FWHM             = {L.delta_t_s*1e15:.0f} fs")
    print(f"  probes           = {', '.join(f'{p:.0f}' for p in L.probe_wavelengths_nm)} nm")
    print("\n  index seen by each beam:")
    print(f"    pump {L.wavelength_m*1e9:.0f} nm : n = {MATERIAL.n(L.wavelength_m):.4f}")
    for p in L.probe_wavelengths_nm:
        print(f"    probe {p:.0f} nm : n = {MATERIAL.n(p*1e-9):.4f}")

    block("BOX AND GRID")
    print(f"  box   {BOX.name}: {BOX.begin_m*1e6:.0f} to {BOX.end_m*1e6:.0f} um")
    for g in (GRID_PRODUCTION, GRID_QUICK):
        tp = LASER.delta_t_s / np.sqrt(2*np.log(2))
        tmax = g.tmax_factor * tp
        print(f"  grid  {g.name:12s} Nz={g.Nz:<6d} Nt={g.Nt:<5d} Nr={g.Nr:<5d} "
              f"window=+/-{tmax*1e15:.0f} fs  dt={2*tmax/g.Nt*1e15:.2f} fs")

    block("PHYSICS SWITCHES")
    for f in Physics.__dataclass_fields__:
        print(f"  {'[ON ]' if getattr(PHYSICS, f) else '[OFF]'}  {f}")

    block("CHECK AGAINST simulate()")
    kw = simulate_kwargs()
    print(f"  simulate_kwargs() produces {len(kw)} arguments")
    try:
        import inspect
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import run_filament
        sig = set(inspect.signature(run_filament.simulate).parameters)
        bad = sorted(set(kw) - sig)
        print(f"  not in simulate() signature: {bad or 'none'}")
        print(f"  left at simulate() defaults: {sorted(sig - set(kw))}")
    except Exception as e:
        print(f"  could not import run_filament ({type(e).__name__}), "
              f"signature not checked")


if __name__ == "__main__":
    _main()
