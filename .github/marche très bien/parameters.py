"""
parameters.py -- every number of the experiment, in one place.

Two dataclasses. `Material` is everything that changes when you swap the
sample, `Experiment` is everything that changes when you change the setup.
Nothing here computes physics: this file is data, and `simulate_kwargs()`
turns a (material, experiment) pair into the call to `simulate()`.

    from parameters import FUSED_SILICA, EXP_1030_4UJ, simulate_kwargs
    from run_filament import simulate

    res = simulate(**simulate_kwargs(FUSED_SILICA, EXP_1030_4UJ))

Swap the sample by swapping one argument:

    res = simulate(**simulate_kwargs(SAPPHIRE, EXP_1030_4UJ))

Override anything for one run without editing the presets:

    res = simulate(**simulate_kwargs(FUSED_SILICA, EXP_1030_4UJ,
                                     energy_incident_uJ=8.0,
                                     enable_kerr_raman=False))


HOW SURE ARE THESE NUMBERS
==========================
Not equally. Every field carries a provenance tag in `SOURCES`, and
`Material.show()` prints it next to the value. The tags:

    measured   a direct measurement, reproduced in several places
    fitted     obtained by fitting a model to data, so it depends on that model
    derived    computed here from other quantities, no new information
    assumed    carried over from another material or a plausible guess
    unknown    a placeholder. Results that depend on it are not predictions

`Material.audit()` lists everything not measured, fitted or derived. Run it
before trusting a run in a material other than fused silica.

Fused silica is the calibrated case: it reproduces the published figures. The
sapphire entry is a starting point, not a validated parameter set, and its
weak fields are tagged accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace, fields as _fields
from typing import Dict, Optional, Sequence, Tuple


# ================================================================================
#  Material
# ================================================================================
@dataclass(frozen=True)
class Material:
    """Everything that changes when the sample changes."""

    name: str

    # ---- linear dispersion -----------------------------------------------
    # Sellmeier: n^2 - 1 = sum_j B_j lam^2 / (lam^2 - L2_j), lam in micrometres.
    # L2 is the SQUARE of the pole positions.
    sellmeier_B: Tuple[float, ...] = ()
    sellmeier_L2: Tuple[float, ...] = ()
    # Where that fit means anything, in micrometres. The solver clips its
    # frequency axis to this window before evaluating the fit, and puts the
    # edges of its spectral mask here. See the note at the end of this file.
    sellmeier_range_um: Tuple[float, float] = (0.18, 5.0)

    # ---- nonlinear response ------------------------------------------------
    n2_m2W: float = 0.0                  # Kerr index
    f_R: float = 0.0                     # fraction of the Kerr that is Raman
    tau_d_s: float = 32e-15              # Raman damping time
    tau_s_s: float = 12e-15              # Raman oscillation period

    # ---- ionization and carriers -------------------------------------------
    Ui_eV: float = 9.0                   # band gap, drives the Keldysh rate
    Us_eV: float = 6.0                   # gap seen by a trapped exciton
    meff_rel: float = 0.64               # reduced mass in the Keldysh rate
    meff_drude_rel: float = 1.0          # effective mass in sigma_w
    tau_c_s: float = 1.7e-15             # electron collision time in sigma_w
    tau_r_s: float = 330e-15             # trapping, conduction band -> STE
    tau_ste_s: Optional[float] = 1e-12   # STE decay to the ground state
    rho_max_cm3: float = 2.1e22          # N_at, saturation density
    E_tr_eV: float = 4.2                 # STE resonance seen by the pump

    # ---- probe-side dielectric model (sim/permittivity.py) -----------------
    # Not used by the propagation. Turns the densities into the phase and
    # transmittance maps of the HTML pages.
    valence_N0_cm3: float = 2.2e22       # density of ionizable units
    n_valence_per_unit: float = 8.0      # valence ELECTRONS per unit, see below
    probe_meff_rel: float = 0.5          # conduction mass in the probe Drude term
    tau_ep_s: float = 1.0 / 1.5e15       # electron-phonon collisions
    ste_bands: Tuple[Tuple[float, float, float], ...] = ()   # (E_eV, f, gamma_eV)

    # ---- interfaces --------------------------------------------------------
    n_fresnel: float = 1.45              # index used for the entrance Fresnel loss

    # ---- is the STE channel meaningful in this material? -------------------
    enable_ste: bool = True

    # ---- provenance --------------------------------------------------------
    sources: Dict[str, str] = field(default_factory=dict)
    notes: str = ""

    # ------------------------------------------------------------------
    def with_(self, **changes) -> "Material":
        """A copy with some fields changed. Materials are frozen on purpose."""
        known = {f.name for f in _fields(self)}
        bad = set(changes) - known
        if bad:
            raise TypeError(f"not Material fields: {sorted(bad)}")
        return replace(self, **changes)

    def provenance(self, field_name: str) -> str:
        return self.sources.get(field_name, "unknown  (no source recorded)")

    def show(self) -> None:
        """Print every value with where it comes from."""
        print("=" * 78)
        print(f"MATERIAL: {self.name}")
        print("=" * 78)
        for f in _fields(self):
            if f.name in ("name", "sources", "notes"):
                continue
            v = getattr(self, f.name)
            if isinstance(v, float):
                vs = f"{v:.6g}"
            elif isinstance(v, tuple):
                vs = "(" + ", ".join(f"{x:.6g}" if isinstance(x, float) else str(x)
                                     for x in v) + ")"
            else:
                vs = str(v)
            if len(vs) > 34:
                vs = vs[:31] + "..."
            print(f"  {f.name:22s} {vs:36s} {self.provenance(f.name)}")
        if self.notes:
            print("\nNOTES")
            for line in self.notes.strip().split("\n"):
                print("  " + line)
        print("=" * 78)

    def audit(self, quiet: bool = False):
        """Field names whose value is assumed, unknown, or unsourced."""
        weak = []
        for f in _fields(self):
            if f.name in ("name", "sources", "notes"):
                continue
            tag = self.provenance(f.name).split()[0]
            if tag not in ("measured", "fitted", "derived"):
                weak.append((f.name, self.provenance(f.name)))
        if not quiet:
            if not weak:
                print(f"{self.name}: every field is measured, fitted or derived.")
            else:
                print(f"{self.name}: {len(weak)} field(s) NOT on solid ground.")
                for n, s in weak:
                    print(f"  {n:22s} {s}")
        return [n for n, _ in weak]


# ================================================================================
#  Fused silica -- the calibrated case
# ================================================================================
FUSED_SILICA = Material(
    name="fused silica (SiO2)",

    sellmeier_B=(0.6961663, 0.4079426, 0.8974794),
    sellmeier_L2=(0.0684043**2, 0.1162414**2, 9.896161**2),
    sellmeier_range_um=(0.18, 5.0),

    n2_m2W=2.74e-20,
    f_R=0.18,
    tau_d_s=32e-15,
    tau_s_s=12e-15,

    Ui_eV=9.0,
    Us_eV=6.0,
    meff_rel=0.64,
    meff_drude_rel=1.0,
    tau_c_s=1.7e-15,
    tau_r_s=330e-15,
    tau_ste_s=1e-12,
    rho_max_cm3=2.1e22,
    E_tr_eV=4.2,

    valence_N0_cm3=2.2e22,
    n_valence_per_unit=8.0,
    probe_meff_rel=0.5,
    tau_ep_s=1.0 / 1.5e15,
    ste_bands=((5.2, 0.40, 1.5), (4.2, 0.15, 1.0)),

    n_fresnel=1.45,
    enable_ste=True,

    sources={
        "sellmeier_B":        "measured  Malitson, JOSA 55, 1205 (1965)",
        "sellmeier_L2":       "measured  Malitson, JOSA 55, 1205 (1965)",
        "sellmeier_range_um": "derived   the range Malitson's fit covers",
        "n2_m2W":             "measured  standard value for fused silica near 1 um",
        "f_R":                "fitted    Couairon et al., PRB 71, 125435 (2005)",
        "tau_d_s":            "fitted    Couairon et al., PRB 71, 125435 (2005)",
        "tau_s_s":            "fitted    Couairon et al., PRB 71, 125435 (2005)",
        "Ui_eV":              "measured  band gap of fused silica",
        "Us_eV":              "fitted    STE gap, Mao et al., Appl. Phys. A 79, 1695 (2004)",
        "meff_rel":           "fitted    Couairon et al., PRB 71, 125435 (2005)",
        "meff_drude_rel":     "assumed   bare mass in sigma_w, the solver's convention",
        "tau_c_s":            "fitted    Couairon et al., PRB 71, 125435 (2005)",
        "tau_r_s":            "measured  trapping time, several fs-pump-probe studies",
        "tau_ste_s":          "measured  Sakurai et al., ~1 ps in fused silica",
        "rho_max_cm3":        "derived   molecular density of SiO2",
        "E_tr_eV":            "measured  STE first excited level, Mao et al. (2004)",
        "valence_N0_cm3":     "derived   molecular density of SiO2",
        "n_valence_per_unit": "derived   4 Si-O bonds x 2 electrons, see the note",
        "probe_meff_rel":     "fitted    Martin et al., PRB 55, 5799 (1997), Table II",
        "tau_ep_s":           "fitted    Martin et al. (1997) Table II, 1/tau = 1.5e15 /s",
        "ste_bands":          "fitted    Martin et al. (1997) Table II",
        "n_fresnel":          "derived   n at the pump, rounded",
        "enable_ste":         "measured  STEs are well established in a-SiO2",
    },

    notes="""
This is the calibrated material. It reproduces the published figures, and the
other entries in this file should be read against it.

n_valence_per_unit deserves its own note, because N0 carries two different
meanings in Martin et al. under one symbol.

  (a) The density of ionizable centres, in the source term N0 sigma_K F^K.
      That is the MOLECULAR density, 2.2e22 cm^-3 for SiO2.
  (b) The number of valence oscillators, in the depletion factor
      (N0 - N_CB - N_tr). That is the number of ELECTRONS carrying the
      polarizability, which is 8 per SiO2: four Si-O bonds of two electrons.

Removing one electron removes 1/N_oscillators of the total polarizability, so
the depletion term needs (b). Using (a) overestimates it eightfold and flips
the sign of the long-delay plateau, which contradicts Fig. 6 of the paper
itself. Hence valence_N0_cm3 * n_valence_per_unit = 1.76e23 as the denominator.
Set n_valence_per_unit to 1.0 to reproduce the older, wrong behaviour.
""",
)


# ================================================================================
#  Sapphire -- a starting point, NOT a validated parameter set
# ================================================================================
SAPPHIRE = Material(
    name="sapphire (Al2O3), ordinary ray",

    # Ordinary ray. Sapphire is birefringent and this file has no notion of
    # that: the solver is scalar. Fine for o-ray propagation along the c axis,
    # wrong for anything else.
    sellmeier_B=(1.4313493, 0.65054713, 5.3414021),
    sellmeier_L2=(0.0726631**2, 0.1193242**2, 18.028251**2),
    sellmeier_range_um=(0.20, 5.0),

    n2_m2W=3.1e-20,
    f_R=0.0,
    tau_d_s=32e-15,
    tau_s_s=12e-15,

    Ui_eV=8.8,
    Us_eV=6.0,
    meff_rel=0.4,
    meff_drude_rel=1.0,
    tau_c_s=1.7e-15,
    tau_r_s=330e-15,
    tau_ste_s=None,
    rho_max_cm3=2.35e22,
    E_tr_eV=4.2,

    valence_N0_cm3=2.35e22,
    n_valence_per_unit=12.0,
    probe_meff_rel=0.4,
    tau_ep_s=1.0 / 1.5e15,
    ste_bands=(),

    n_fresnel=1.7551,
    enable_ste=False,

    sources={
        "sellmeier_B":        "measured  Malitson & Dodge (1972), ordinary ray",
        "sellmeier_L2":       "measured  Malitson & Dodge (1972), ordinary ray",
        "sellmeier_range_um": "derived   the range that fit covers",
        "n2_m2W":             "measured  reported 2.8-3.2e-20 near 800 nm, spread ~15%",
        "f_R":                "assumed   set to zero, see the note",
        "tau_d_s":            "unknown   silica's value, meaningless while f_R = 0",
        "tau_s_s":            "unknown   silica's value, meaningless while f_R = 0",
        "Ui_eV":              "measured  8.8 eV common in damage work, 9.9 also quoted",
        "Us_eV":              "unknown   no STE channel here, see the note",
        "meff_rel":           "assumed   ~0.4 is quoted, but not a Keldysh fit",
        "meff_drude_rel":     "assumed   bare mass, the solver's convention",
        "tau_c_s":            "unknown   carried over from silica, not measured here",
        "tau_r_s":            "unknown   no STE channel here, see the note",
        "tau_ste_s":          "unknown   no STE channel here, see the note",
        "rho_max_cm3":        "derived   3.98 g/cm3 / 101.96 g/mol x N_A",
        "E_tr_eV":            "unknown   no STE channel here, see the note",
        "valence_N0_cm3":     "derived   same formula-unit density",
        "n_valence_per_unit": "assumed   6 Al-O bonds x 2 electrons, by analogy",
        "probe_meff_rel":     "assumed   same as meff_rel above",
        "tau_ep_s":           "unknown   carried over from silica",
        "ste_bands":          "assumed   empty, no STE absorption modelled",
        "n_fresnel":          "derived   n(1030 nm) from the Sellmeier fit above",
        "enable_ste":         "assumed   switched off deliberately, see the note",
    },

    notes="""
Do not read a sapphire run as a prediction. Run SAPPHIRE.audit() and look at
what it lists before drawing any conclusion.

What is solid: the dispersion. The Malitson and Dodge fit is the standard
reference and gives n(1030 nm) = 1.7551, and the formula-unit density follows
from the density and the molar mass. Those two are as good as silica's.

What is not: everything in the carrier model. The numbers below are either
carried over from silica or quoted from a range.

The STE channel is switched off, and that is a physics decision rather than
missing data. The self-trapped exciton of a-SiO2 is a specific defect, a
dangling Si-O bond pair, with a measured trapping time and measured absorption
bands. Crystalline Al2O3 traps carriers through a different set of colour
centres, mostly F and F+ centres, on different timescales. Reusing silica's
tau_r, tau_ste, E_tr and band table would produce plausible-looking curves
built on nothing. If you need trapping in sapphire, put in the F-centre
parameters and turn the channel back on:

    SAPPHIRE.with_(enable_ste=True, tau_r_s=..., E_tr_eV=...,
                   ste_bands=((..., ..., ...),))

f_R is set to zero for the same reason. Silica's 0.18 was fitted to silica, and
sapphire's Raman response is not the same. Zero means a purely electronic Kerr,
which is a defensible approximation rather than a borrowed number.

Sapphire is birefringent and this solver is scalar. These are ordinary-ray
values, so they apply to propagation along the c axis and not otherwise.
""",
)


MATERIALS = {m.name: m for m in (FUSED_SILICA, SAPPHIRE)}


# ================================================================================
#  Experiment
# ================================================================================
@dataclass(frozen=True)
class Experiment:
    """Everything that changes when the setup changes, not the sample."""

    name: str

    # ---- pump --------------------------------------------------------------
    wavelength_m: float = 1030e-9
    energy_incident_uJ: float = 4.0      # BEFORE the entrance face
    apply_fresnel: bool = True
    spot_sx_um: float = 11.5
    spot_sy_um: float = 11.0
    delta_t_s: float = 263e-15           # FWHM in intensity
    w0_m: Optional[float] = None         # None: sqrt(sx*sy)

    # ---- probe, 90 degrees from the pump (Nomarski / Abel) -----------------
    probe_wavelengths_nm: Tuple[float, ...] = (490.0, 620.0, 690.0)

    # ---- box ---------------------------------------------------------------
    begin_m: float = 0.0
    end_m: float = 350e-6
    z_focus_air_um: Optional[float] = None

    # ---- numerical grid ----------------------------------------------------
    Nz: int = 9000
    Nt: int = 4096
    Nr: int = 1024
    R_factor: float = 8.0
    tmax_factor: float = 10.0
    save_stride: int = 20
    rho_t_stride: int = 8
    rho_r_stride: int = 2

    # ---- HTML --------------------------------------------------------------
    html_t_step_fs: float = 67.0
    html_coarsen_z: int = 4
    html_phase_clip: float = 0.2
    html_t_min: Optional[float] = None
    html_z_lim_um: Tuple[float, float] = (0.0, 350.0)
    html_x_lim_um: Tuple[float, float] = (-50.0, 50.0)

    notes: str = ""

    def with_(self, **changes) -> "Experiment":
        known = {f.name for f in _fields(self)}
        bad = set(changes) - known
        if bad:
            raise TypeError(f"not Experiment fields: {sorted(bad)}")
        return replace(self, **changes)

    def show(self) -> None:
        print("=" * 78)
        print(f"EXPERIMENT: {self.name}")
        print("=" * 78)
        for f in _fields(self):
            if f.name in ("name", "notes"):
                continue
            v = getattr(self, f.name)
            vs = f"{v:.6g}" if isinstance(v, float) else str(v)
            print(f"  {f.name:22s} {vs}")
        if self.notes:
            print("\nNOTES")
            for line in self.notes.strip().split("\n"):
                print("  " + line)
        print("=" * 78)


EXP_1030_4UJ = Experiment(
    name="1030 nm, 4 uJ, z0 to 350 um, probes at 490/620/690 nm",
    notes="""
The pump energy is what arrives at the sample. apply_fresnel takes off the
reflection at the entrance face, using the material's n_fresnel, so the solver
gets the energy actually inside.

tmax_factor sets the comoving window, tmax = tmax_factor * tp. At the historical
5.0 it spans only about +/-1.1 ps for a 263 fs pulse, too short to see the STE
decay of roughly 1 ps: past tmax the recorded cube has no data and the HTML
cursor plateaus. Raising it to 10 and Nt to 4096 together keeps the same dt and
doubles the window, for roughly twice the run time.

html_t_step_fs is the cursor step, not the resolution of the cube. Leaving it
at None puts every recorded instant in the page, several hundred of them, each
carrying a phase map and a density map, and the file runs into gigabytes.
""",
)

EXPERIMENTS = {e.name: e for e in (EXP_1030_4UJ,)}


# ================================================================================
#  Turning a pair into a call
# ================================================================================
def simulate_kwargs(material: Material, experiment: Experiment,
                    **overrides) -> dict:
    """Arguments for run_filament.simulate(), from a material and a setup.

    `overrides` go in last and win, so a one-off change needs no edit to the
    presets. An override that is not an argument of simulate() raises there,
    not silently here.
    """
    if not material.sellmeier_B or not material.sellmeier_L2:
        raise ValueError(f"{material.name}: no Sellmeier fit, cannot propagate")

    kw = dict(
        # dispersion, installed into the solver before anything reads an index
        material_name=material.name,
        sellmeier_B=tuple(material.sellmeier_B),
        sellmeier_L2=tuple(material.sellmeier_L2),
        sellmeier_range_um=tuple(material.sellmeier_range_um),

        # material, solver side
        n2_m2W=material.n2_m2W,
        f_R=material.f_R, tau_d_s=material.tau_d_s, tau_s_s=material.tau_s_s,
        Ui_eV=material.Ui_eV, Us_eV=material.Us_eV, E_tr_eV=material.E_tr_eV,
        meff_rel=material.meff_rel, meff_drude_rel=material.meff_drude_rel,
        tau_c_s=material.tau_c_s, tau_r_s=material.tau_r_s,
        tau_ste_s=material.tau_ste_s, rho_max_cm3=material.rho_max_cm3,
        enable_ste=material.enable_ste,

        # material, probe side
        valence_N0_cm3=material.valence_N0_cm3,
        n_valence_per_unit=material.n_valence_per_unit,
        probe_meff_rel=material.probe_meff_rel,
        tau_ep_s=material.tau_ep_s,
        ste_bands=tuple(tuple(b) for b in material.ste_bands),

        # interface
        n_glass_fresnel=material.n_fresnel,

        # setup
        wavelength_m=experiment.wavelength_m,
        energy_incident_uJ=experiment.energy_incident_uJ,
        apply_fresnel=experiment.apply_fresnel,
        spot_sx_um=experiment.spot_sx_um, spot_sy_um=experiment.spot_sy_um,
        delta_t_s=experiment.delta_t_s, w0_m=experiment.w0_m,
        probe_wavelengths_nm=tuple(experiment.probe_wavelengths_nm),
        begin_m=experiment.begin_m, end_m=experiment.end_m,
        z_focus_air_um=experiment.z_focus_air_um,
        Nz=experiment.Nz, Nt=experiment.Nt, Nr=experiment.Nr,
        R_factor=experiment.R_factor, tmax_factor=experiment.tmax_factor,
        save_stride=experiment.save_stride,
        rho_t_stride=experiment.rho_t_stride,
        rho_r_stride=experiment.rho_r_stride,
        html_t_step_fs=experiment.html_t_step_fs,
        html_coarsen_z=experiment.html_coarsen_z,
        html_phase_clip=experiment.html_phase_clip,
        html_t_min=experiment.html_t_min,
        html_z_lim_um=tuple(experiment.html_z_lim_um),
        html_x_lim_um=tuple(experiment.html_x_lim_um),
    )
    kw.update(overrides)
    return kw


# ================================================================================
#  A note on sellmeier_range_um, since it is the least obvious field here
# ================================================================================
#  It is not a physical property of the material, it is where the FIT is
#  meaningful, and the solver needs it for two separate reasons.
#
#  The time grid gives the solver a frequency axis far wider than the pulse. At
#  Nt = 4096 and 1030 nm the absolute frequency omega0 + Omega runs from about
#  -170 to +750 THz, so roughly a fifth of the bins sit at NEGATIVE absolute
#  frequency, which is an artifact of writing an envelope equation on a
#  discrete grid.
#
#  First, the Sellmeier form has poles, at lambda^2 = L2_j. Silica's are at
#  0.068, 0.116 and 9.90 um. Evaluating the fit near one returns a divergence
#  and beyond one returns a negative index. grids.py therefore clips the
#  frequency axis to this window before evaluating the fit, which freezes the
#  index at the edge value instead of letting it explode. The 0.18 and 5 um
#  bounds for silica sit safely between the second and third pole.
#
#  Second, the self-steepening operator is T^ = omega/omega0, which changes
#  SIGN wherever the absolute frequency is negative, down to -0.58 on this
#  grid. An operator standing for d/dt that changes sign is meaningless. At the
#  other end T^ reaches 2.58, so T^2 reaches 6.6, and anything that aliases up
#  there is amplified by that factor at every step, which is a feedback loop.
#  The spectral mask kills both regions, about a quarter of the grid, with tanh
#  edges rather than a hard cut so the truncation does not ring.
#
#  So the window is material-dependent through the fit, not through the physics
#  of the sample. A material whose published fit covers a different range needs
#  a different window here.


if __name__ == "__main__":
    for m in MATERIALS.values():
        m.show()
        print()
        m.audit()
        print()
    EXP_1030_4UJ.show()
