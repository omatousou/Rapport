"""
run_filament.py -- one entry point for the SiO2 filamentation simulation.

WHAT THIS FILE IS FOR
=====================
The solver itself lives in sim/ and is split across seven modules, because each
term of the propagation equation is handled in the basis where it is diagonal.
That split is good for the solver and bad for the reader: to find out which
physics is actually switched on, you currently have to open config.py,
grids.py and operators.py at the same time.

This file is the opposite trade-off. It exposes ONE function, `simulate()`,
whose signature lists EVERY physical parameter explicitly, with the equation
each one belongs to written next to it. Nothing is hidden behind **kwargs, and
a misspelled parameter raises immediately instead of being silently ignored.

    from run_filament import simulate
    res = simulate(energy_incident_uJ=4.0, meff_drude_rel=1.0)

`simulate()` runs the propagation, then builds the interactive HTML pages for
the probe wavelengths asked for. Running this file with no arguments
reproduces the 4 uJ run of notebooks/filament_1030nm_4uJ.ipynb.

    python run_filament.py


THE EQUATIONS THE SOLVER INTEGRATES
===================================
Field envelope u(r, t, z), Couairon et al., PRB 71, 125435 (2005), Eq. (2).
Each line carries the flag that switches it off, so the equation below and the
signature of `simulate()` can be read side by side.

  U^ du/dz =   (i / 2k0) grad_perp^2 u                      always on
             + i D^ U^ u                                    always on
             + i T^^2 (3 w0^2 chi3 / 8 k0 c^2) (1-f_R) |u|^2 u
                                                            enable_kerr_instantaneous
             + i T^^2 (3 w0^2 chi3 / 8 k0 c^2) f_R (R*|u|^2) u
                                                            enable_kerr_raman
             - T^ (Ui W_PI / n0 c eps0 |u|^2) (1 - N/N_at) u
                                                            enable_photoionization_loss
             - (sigma_w / 2) N u                            enable_plasma_absorption
             - i (sigma_w w0 tau_c / 2) N u                 enable_plasma_defocusing
             + i (w0 / 2 n0 c rho_c) f_STE N_STE u          enable_ste_index

  T^ = 1 + (i/w0) d/dt                 enable_self_steepening   (T^ = 1 when off)
  U^ = 1 + (i k'/k0) d/dt              enable_space_time_focusing (U^ = 1 when off)
  D^ = k(w) - k0 - k' Omega            Sellmeier, no Taylor truncation
  f_STE = w0^2 / (w_tr^2 - w0^2)       Lorentz factor at the STE level w_tr
  rho_c = eps0 m_e w0^2 / q^2          critical density at the pump, bare mass

Note on U^. In half_linear the factor U^-1 multiplies the diffraction term
only, not D^, which is why D^ picks up a U^ once U^ is moved to the left. That
is the standard form: diffraction goes as 1/k(w) and dispersion does not.

Note on the Kerr prefactor. 3 w0^2 chi3 / 8 k0 c^2 with chi3 = (4/3) eps0 n0^2
c n2 is algebraically equal to (w0/c) n2 I / |u|^2, so the Kerr term is
i (w0/c) n2 I u, the same k_vac * dn form as the STE term above.

Carrier populations, solved on the GPU before every propagation step
(sim/kernels.py). These are NOT gated by the flags above: switching off a
field term stops it acting on the beam, it does not stop carriers being made.

  dN/dt     = W_PI(I) (1 - N/N_at)                          always on
            + beta_g I N (1 - N/N_at)                       enable_avalanche
            + (W_STE + beta_s I N) N_STE/N_at               enable_ste
            - N / tau_r                                     enable_recombination

  dN_STE/dt = N / tau_r                                     enable_ste
            - (W_STE + beta_s I N) N_STE/N_at               enable_ste
            - N_STE / tau_ste                               enable_ste and tau_ste

W_PI is the Keldysh rate at the band gap Ui, W_STE the same formula evaluated
at the shallower self-trapped exciton gap Us. Both are tabulated once before
propagation and read by interpolation inside the CUDA kernel.

beta_g = sigma_w / Ui and beta_s = sigma_w / Us are the avalanche coefficients
for the valence band and for the trapped excitons. Both are zeroed by
enable_avalanche, so switching avalanche off also removes the beta_s I N part
of the STE re-ionization, leaving only its W_STE part.


WHAT THE EQUATIONS ABOVE LEAVE OUT
==================================
Four things the solver does that are not in the equations. They are all
deliberate, but none of them can be found by reading the equations alone.

1. A radial absorbing boundary. Every z step ends with u multiplied by
   exp(-(r / 0.9R)^20), built in grids.py as mask_r and applied at the end of
   Integrator.step. It keeps light that reaches the edge of the box from
   wrapping around through the Hankel transform. It also removes energy, so a
   run whose beam gets close to R does not conserve energy for a physical
   reason.

2. A joint saturation clamp on the populations. If N + N_STE exceeds N_at at
   any time step, the CUDA kernel scales both down so that they sum to N_at.
   Neither rate equation contains this, and it becomes active exactly in the
   strongly ionized regime one usually cares about.

3. The spectral mask is not a separate factor. It is folded into T^ and into
   U^-1, so the Kerr term carries it squared while the ionization term carries
   it once. More importantly, when enable_self_steepening is off T^ becomes a
   plain 1 with no mask at all, and likewise for U^-1 when space time focusing
   is off. Turning those two flags off therefore also removes the spectral
   filter from the nonlinear step, even with enable_spectral_filter left on.
   The linear step keeps its own mask either way.

4. D^ is exact only inside the Sellmeier window. It is built from a frequency
   axis clipped to lambda in [0.18, 5] um, so outside that band it is frozen at
   the edge value rather than extrapolated. The spectral mask has normally
   killed the field there already.


HOW TO CHECK ANY OF THIS YOURSELF
=================================
The equations above were re-derived from the code, not copied from comments.
The derivation is short enough to repeat, and worth repeating after any change.

Start from Integrator.step. Over one z step the field is multiplied by
exp(-alpha dz/2), integrated by RK4 on split(), multiplied by exp(-alpha dz/2)
again, and passed through half_linear twice. The two exponentials contribute
-alpha u to du/dz and split() ends with a + alpha * u that cancels them, so the
whole nonlinear contribution to du/dz is exactly

    ifft(NL_freq * inv_U_nl)

and nothing else. Read NL_freq in split() as the equation, read alpha as a
numerical device for applying the stiff part exponentially, and the field
equation falls out directly. Then multiply through by U to compare with the
form written above.

For the linear part, read the phase built in half_linear, remember that rho^2
in the Hankel basis is minus the transverse Laplacian, and do the same.

For prefactors, do not read them, recompute them. Each one should reduce to a
textbook expression: the Kerr prefactor to (w0/c) n2 I / |u|^2, sigma_w to the
Drude cross section, ste_pref to k_vac times the Lorentz index change. A
prefactor that does not reduce to something recognizable is the first place to
look for a transcription error.

To change what the solver integrates rather than only which terms are on, see
MODIFYING_THE_EQUATIONS.md next to this file. It maps every term above to the
line that computes it and explains the one piece of bookkeeping that is easy to
get wrong, namely that split() does not return du/dz.
"""

from __future__ import annotations

import json
import sys
from dataclasses import fields as _dataclass_fields
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from scipy.constants import c as c_SI


# ================================================================================
#  Locating sim/ and web/
# ================================================================================
def _add_package_dirs(sim_dir: Optional[str] = None) -> dict:
    """Put sim/ and web/ on sys.path and return where they were found.

    Walks up from this file first, then from the working directory, looking for
    a folder that actually contains sim/filament_sim.py. Explicit `sim_dir`
    wins over both.
    """
    roots = []
    if sim_dir:
        p = Path(sim_dir).expanduser().resolve()
        roots.append(p.parent if p.name == "sim" else p)
    for base in (Path(__file__).resolve().parent, Path.cwd().resolve()):
        roots += [base, *base.parents]

    for root in roots:
        if (root / "sim" / "filament_sim.py").is_file():
            found = {n: root / n for n in ("sim", "web") if (root / n).is_dir()}
            for d in found.values():
                if str(d) not in sys.path:
                    sys.path.insert(0, str(d))
            return found
    raise ModuleNotFoundError(
        "sim/filament_sim.py not found. Pass sim_dir=... explicitly.")


# ================================================================================
#  Reporting which physics is active
# ================================================================================
#  (flag name, term as it appears in the equation, which equation)
#
#  The six terms of the field equation are NOT listed here. They are read from
#  operators.FIELD_TERMS, the registry the solver itself loops over, so this
#  listing cannot drift from what split() actually assembles. Only the things
#  that are not terms of that sum are listed below: the two operators, the
#  spectral mask, and the carrier equation flags.
_EXTRA_TERMS = [
    ("enable_self_steepening",    "T^ = 1 + (i/w0) d/dt   (else T^ = 1)", "field"),
    ("enable_space_time_focusing", "U^ = 1 + (i k'/k0) d/dt   (else U^ = 1)", "field"),
    ("enable_spectral_filter",    "spectral mask on the Sellmeier window", "field"),
    ("enable_avalanche",          "+ beta_g I N (1 - N/N_at)", "carriers"),
    ("enable_recombination",      "- N / tau_r   (trapping into STE)", "carriers"),
    ("enable_ste",                "STE channel: trapping, re-ionization, index", "carriers"),
]


def _term_table():
    """The listing, with the field terms taken from the solver's own registry."""
    from operators import FIELD_TERMS
    return [(t.flag, t.equation, "field") for t in FIELD_TERMS] + _EXTRA_TERMS


def print_active_physics(cfg_kwargs: dict, tau_ste: Optional[float]) -> None:
    """Print each term of both equations with ON or OFF next to it."""
    table = _term_table()
    print("\n" + "=" * 74)
    print("PHYSICS ACTUALLY INTEGRATED")
    print("=" * 74)
    for eq_name, header in (("field", "Field envelope, Couairon 2005 Eq. (2)"),
                            ("carriers", "Carrier populations, Eqs. (6)-(7)")):
        print(f"\n{header}")
        for flag, term, which in table:
            if which != eq_name:
                continue
            on = bool(cfg_kwargs.get(flag, True))
            print(f"   [{'ON ' if on else 'OFF'}]  {term}")
        if eq_name == "carriers":
            on = tau_ste is not None
            print(f"   [{'ON ' if on else 'OFF'}]  - N_STE / tau_ste"
                  + (f"   (tau_ste = {tau_ste*1e15:.0f} fs)" if on else "   (tau_ste = None)"))
    print("\nAlways on, not switchable: diffraction, Sellmeier dispersion to all")
    print("orders, Keldysh photoionization as the carrier source.")
    print("=" * 74 + "\n")


# ================================================================================
#  The one function
# ================================================================================
def simulate(
    # ---------------- pump laser ----------------
    wavelength_m: float = 1030e-9,
    energy_incident_uJ: float = 4.0,
    apply_fresnel: bool = True,
    n_glass_fresnel: float = 1.45,
    w0_m: Optional[float] = None,
    spot_sx_um: float = 11.5,
    spot_sy_um: float = 11.0,
    delta_t_s: float = 263e-15,
    peak_power_W: Optional[float] = None,

    # ---------------- dispersion ----------------
    # Pass both, or neither. Given, they are installed into the solver through
    # keldysh.set_dispersion() before anything reads an index, and recorded in
    # params.json so the HTML pages use the same fit. None keeps fused silica.
    material_name: str = "fused silica (SiO2)",
    sellmeier_B: Optional[Sequence[float]] = None,
    sellmeier_L2: Optional[Sequence[float]] = None,   # squared pole positions
    sellmeier_range_um: Optional[Sequence[float]] = None,

    # ---------------- probe-side dielectric model ----------------
    # Not used by the propagation. Turns the densities into what the probe
    # sees, i.e. the phase and transmittance maps. Martin et al. 1997 Table II.
    valence_N0_cm3: float = 2.2e22,
    n_valence_per_unit: float = 8.0,
    probe_meff_rel: float = 0.5,
    tau_ep_s: float = 1.0 / 1.5e15,
    ste_bands: Sequence = ((5.2, 0.40, 1.5), (4.2, 0.15, 1.0)),

    # ---------------- material, SiO2 ----------------
    n2_m2W: float = 2.74e-20,          # nonlinear index, Kerr term
    Ui_eV: float = 9.0,                # band gap, Keldysh rate W_PI
    Us_eV: float = 6.0,                # STE gap, Keldysh rate W_STE
    E_tr_eV: float = 4.2,              # STE resonance, pump-side index
    meff_rel: float = 0.64,            # reduced mass in the Keldysh rate
    meff_drude_rel: float = 1.0,       # effective mass in sigma_w (Drude)
    tau_c_s: float = 1.7e-15,          # electron collision time, sigma_w
    tau_r_s: float = 330e-15,          # trapping time, N -> N_STE
    tau_ste_s: Optional[float] = 1e-12,  # STE decay to ground state
    rho_max_cm3: float = 2.1e22,       # N_at, saturation density
    f_R: float = 0.18,                 # Raman fraction of the Kerr response
    tau_d_s: float = 32e-15,           # Raman damping time
    tau_s_s: float = 12e-15,           # Raman oscillation period

    # ---------------- geometry of the box ----------------
    begin_m: float = 0.0,
    end_m: float = 350e-6,
    z_focus_air_um: Optional[float] = None,

    # ---------------- numerical grid ----------------
    Nz: int = 9000,
    Nt: int = 4096,
    Nr: int = 1024,
    R_factor: float = 8.0,
    tmax_factor: float = 10.0,
    save_stride: int = 20,
    rho_t_stride: int = 8,
    rho_r_stride: int = 2,
    ckpt_every: int = 200,

    # ---------------- field equation, term by term ----------------
    enable_kerr_instantaneous: bool = True,
    enable_kerr_raman: bool = True,
    enable_self_steepening: bool = True,
    enable_photoionization_loss: bool = True,
    enable_plasma_defocusing: bool = True,
    enable_plasma_absorption: bool = True,
    enable_space_time_focusing: bool = True,
    enable_spectral_filter: bool = True,
    enable_ste_index: bool = True,

    # ---------------- carrier equations, term by term ----------------
    enable_avalanche: bool = True,
    enable_recombination: bool = True,
    enable_ste: bool = True,

    # ---------------- probe and outputs ----------------
    probe_wavelengths_nm: Sequence[float] = (490.0, 620.0, 690.0),
    out_root: str = "runs_z0_probe_sweeps",
    run_tag: Optional[str] = None,
    make_html: bool = True,
    html_t_step_fs: float = 67.0,
    html_coarsen_z: int = 4,
    html_coarsen_r: int = 1,
    html_phase_clip: float = 0.2,
    html_t_min: Optional[float] = None,
    html_z_lim_um: Sequence[float] = (0.0, 350.0),
    html_x_lim_um: Sequence[float] = (-50.0, 50.0),

    # ---------------- housekeeping ----------------
    reuse_cached: bool = True,
    verbose: bool = True,
    sim_dir: Optional[str] = None,
):
    """Run the filamentation simulation and build the probe HTML pages.

    Every physical parameter is a named argument, so the whole configuration of
    a run is visible in one call and a typo raises instead of being ignored.

    A few arguments deserve a note.

    `w0_m` is the 1/e intensity radius in the medium. Left at None it is taken
    as sqrt(sx*sy) from the two measured spot sizes, which is what the notebook
    does. Paraxial refraction at a flat interface leaves the waist size
    unchanged, so a value measured in air can be used directly.

    `apply_fresnel` multiplies the incident energy by the transmission of the
    air-glass interface before it reaches the solver. Set it to False to pass
    the energy already inside the glass.

    `meff_rel` and `meff_drude_rel` are two different masses and are easy to
    confuse. The first is the reduced mass entering the Keldysh rate, the
    second is the effective mass in the Drude conductivity sigma_w, which sets
    both plasma absorption and defocusing.

    `tau_r_s` is called a recombination time in Couairon 2005 and a trapping
    time in the STE literature. It is the same process, namely a conduction
    band electron falling into a self-trapped state, and it feeds N_STE.

    `tmax_factor` sets the comoving time window, tmax = tmax_factor * tp. At
    the historical 5.0 the window is about +/-1.1 ps for a 263 fs pulse, too
    short to see the STE decay. Raising it widens the window, and Nt should be
    raised with it to keep the same dt.

    Returns a dict with the result arrays, the output directory and the paths
    of the HTML pages that were written.
    """
    dirs = _add_package_dirs(sim_dir)

    import keldysh
    import abel_phase_explorer as _explorer
    from config import Config
    from keldysh import n_sellmeier
    import figures_filament as ff
    from abel_phase_explorer import build_explorer_html, probe_optics

    # ---- dispersion, before anything reads an index --------------------------
    # keldysh is the single source of truth; grids.py reads it at call time and
    # Integrator records it in params.json. The explorer keeps its own copy so
    # it can be used without the solver, so it is put in step here too, which
    # matters for the probe_optics table printed below.
    if (sellmeier_B is None) != (sellmeier_L2 is None):
        raise ValueError("pass both sellmeier_B and sellmeier_L2, or neither")
    if sellmeier_B is not None:
        keldysh.set_dispersion(sellmeier_B, sellmeier_L2, sellmeier_range_um)
    elif sellmeier_range_um is not None:
        keldysh.set_dispersion(keldysh.SELLMEIER_B, keldysh.SELLMEIER_L2,
                               sellmeier_range_um)
    _explorer.set_dispersion(*keldysh.get_dispersion()[:2])

    # ---- derived laser quantities -------------------------------------------
    if w0_m is None:
        w0_m = float(np.sqrt(spot_sx_um * spot_sy_um) * 1e-6)
    fresnel_T = 1.0 - ((n_glass_fresnel - 1.0) / (n_glass_fresnel + 1.0)) ** 2
    energy_in_glass_uJ = energy_incident_uJ * (fresnel_T if apply_fresnel else 1.0)

    n0 = float(n_sellmeier(wavelength_m))
    k0 = 2.0 * np.pi * n0 / wavelength_m
    zR = k0 * w0_m ** 2 / 2.0
    tp = delta_t_s / np.sqrt(2.0 * np.log(2.0))
    tmax = tmax_factor * tp
    dt = 2.0 * tmax / Nt
    dz = (end_m - begin_m) / Nz
    R_max = R_factor * w0_m
    dr = R_max / (Nr - 1)

    n_saves = Nz // save_stride + 1
    Nt_sub = (Nt - 1) // rho_t_stride + 1
    Nr_sub = (Nr - 2) // rho_r_stride + 1
    dt_cube_fs = 2.0 * tmax * 1e15 / (Nt_sub - 1)

    # ---- everything that goes to the solver ---------------------------------
    # Assembled as a dict so it can be checked against Config before use. The
    # stock run() swallows unknown names through **material, which turns a
    # typo into a silently ignored parameter.
    cfg_kwargs = dict(
        nz=Nz, Nt=Nt, N=Nr, R_factor=R_factor,
        begin=begin_m, end=end_m, z_focus_air_um=z_focus_air_um,
        wavelength=wavelength_m, energy_uJ=energy_in_glass_uJ,
        peak_power_W=peak_power_W, w0=w0_m, delta_t=delta_t_s,
        tmax_factor=tmax_factor,
        material_name=material_name,
        valence_N0_cm3=valence_N0_cm3, n_valence_per_unit=n_valence_per_unit,
        probe_meff_rel=probe_meff_rel, tau_ep_s=tau_ep_s,
        ste_bands=tuple(tuple(b) for b in ste_bands),
        n2=n2_m2W, Ui_eV=Ui_eV, Us_eV=Us_eV, E_tr_eV=E_tr_eV,
        meff_rel=meff_rel, meff_drude_rel=meff_drude_rel,
        tau_c=tau_c_s, tau_r=tau_r_s, tau_ste=tau_ste_s,
        rho_max=rho_max_cm3, f_R=f_R, tau_d=tau_d_s, tau_s=tau_s_s,
        enable_kerr_instantaneous=enable_kerr_instantaneous,
        enable_kerr_raman=enable_kerr_raman,
        enable_self_steepening=enable_self_steepening,
        enable_photoionization_loss=enable_photoionization_loss,
        enable_plasma_defocusing=enable_plasma_defocusing,
        enable_plasma_absorption=enable_plasma_absorption,
        enable_space_time_focusing=enable_space_time_focusing,
        enable_spectral_filter=enable_spectral_filter,
        enable_ste_index=enable_ste_index,
        enable_avalanche=enable_avalanche,
        enable_recombination=enable_recombination,
        enable_ste=enable_ste,
        lambda_probe=float(probe_wavelengths_nm[0]) * 1e-9,
        rho_t_stride=rho_t_stride, rho_r_stride=rho_r_stride,
        save_stride=save_stride, ckpt_every=ckpt_every, verbose=verbose,
    )
    known = {f.name for f in _dataclass_fields(Config)}
    unknown = set(cfg_kwargs) - known
    if unknown:
        raise TypeError(f"not Config fields: {sorted(unknown)}")

    # ---- report -------------------------------------------------------------
    if verbose:
        print(f"modules: {dirs}")
        print(f"pump      = {wavelength_m*1e9:.0f} nm")
        print(f"w0        = {w0_m*1e6:.2f} um        z_R = {zR*1e6:.0f} um")
        print(f"energy    = {energy_incident_uJ:g} uJ incident"
              + (f" -> {energy_in_glass_uJ:.3f} uJ in glass (T={fresnel_T:.3f})"
                 if apply_fresnel else " (no Fresnel correction)"))
        print(f"z         = Nz={Nz}  dz={dz*1e9:.1f} nm  |  {n_saves} planes, "
              f"dz_save={dz*save_stride*1e6:.2f} um")
        print(f"r         = R_max={R_max*1e6:.0f} um  dr={dr*1e9:.0f} nm  "
              f"({w0_m/dr:.0f} pts in w0)")
        print(f"t         = Nt={Nt}  dt={dt*1e15:.2f} fs  window +/-{tmax*1e15:.0f} fs  "
              f"f_Nyq/f0={1.0/(2*dt)/(c_SI/wavelength_m):.2f}")
        print(f"cube      = {3*n_saves*Nr_sub*Nt_sub*4/1e6:.0f} Mo   "
              f"dt_cube={dt_cube_fs:.1f} fs")
        print("\nprobe wavelengths:")
        for probe_nm in probe_wavelengths_nm:
            n0p, ncp = probe_optics(probe_nm)
            print(f"  {probe_nm:.0f} nm : n0={n0p:.4f}, n_c={ncp:.3e} cm-3")
        print_active_physics(cfg_kwargs, tau_ste_s)

        P_cr = ff.critical_power(n2_m2W, wavelength_m, n0)
        P_in = energy_in_glass_uJ * 1e-6 / (tp * np.sqrt(np.pi / 2.0))
        ratio, _, L_c, _ = ff.marburger_collapse(P_in, P_cr, w0_m, wavelength_m, n0)
        print(f"P_cr = {P_cr*1e-6:.2f} MW    P/P_cr = {ratio:.1f}    L_c = {L_c*1e6:.0f} um")
        ff.check_entrance_intensity(energy_in_glass_uJ, w0_m, delta_t_s,
                                    begin_m, wavelength_m, n0)

    # ---- run, or reuse a cached result --------------------------------------
    out_root_p = Path(out_root)
    html_dir = out_root_p / "html"
    out_root_p.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)

    if run_tag is None:
        # The material goes in the tag. Without it a cached result made in one
        # material would be reused for another: code_fingerprint() only covers
        # the source files, and swapping material changes no source file.
        slug = "".join(ch if ch.isalnum() else "_" for ch in material_name).strip("_")
        while "__" in slug:
            slug = slug.replace("__", "_")
        run_tag = (f"{slug}_z0_{(end_m-begin_m)*1e6:.0f}um_{energy_incident_uJ:g}uJ"
                   f"_pump{wavelength_m*1e9:.0f}nm")
    out_dir = out_root_p / run_tag
    cfg_kwargs["out_dir"] = str(out_dir)

    res = ff.load_scenario_npz(out_dir) if reuse_cached else None
    if res is None:
        if verbose:
            print(f"\n=== running {run_tag} ===")
        from filament_sim import run
        # run() renames three of them and passes the rest through to Config.
        kw = dict(cfg_kwargs)
        res = run(Nz=kw.pop("nz"), Nt=kw.pop("Nt"), Nr=kw.pop("N"),
                  envelope="gaussian_focused", **kw)
    elif verbose:
        print(f"cached result reused: {out_dir}")

    params_path = out_dir / "params.json"
    if params_path.exists():
        pump_nm = float(json.loads(params_path.read_text()).get("wavelength_nm", np.nan))
        if not np.isclose(pump_nm, wavelength_m * 1e9):
            raise ValueError(f"params.json says pump = {pump_nm} nm, "
                             f"expected {wavelength_m*1e9:.0f} nm")

    if verbose:
        ff.run_health_check(res, out_dir=out_dir, label=run_tag, rho_max=rho_max_cm3)

    # ---- interactive HTML, one page per probe wavelength --------------------
    html_files = {}
    if make_html:
        sim_dirs = {run_tag: str(out_dir)}
        for probe_nm in probe_wavelengths_nm:
            save = html_dir / f"abel_{run_tag}_probe_{probe_nm:.0f}nm.html"
            if verbose:
                print(f"\n=== HTML, probe {probe_nm:.0f} nm ===")
            build_explorer_html(
                sim_dirs=sim_dirs, save=str(save), raw_dir=None,
                energy_uJ=energy_incident_uJ, lmd_nm=probe_nm,
                t_step_fs=html_t_step_fs, apply_na_filter=True,
                phase_clip=html_phase_clip, t_min=html_t_min,
                xlim=list(html_z_lim_um), ylim=list(html_x_lim_um),
                coarsen_z=html_coarsen_z, coarsen_r=html_coarsen_r,
            )
            html_files[probe_nm] = save
        if verbose:
            print("\nHTML written:")
            for probe_nm, path in html_files.items():
                print(f"  {probe_nm:.0f} nm -> {path}")

    return dict(result=res, out_dir=out_dir, html=html_files,
                config=cfg_kwargs, n0=n0, w0_m=w0_m,
                energy_in_glass_uJ=energy_in_glass_uJ)


# ================================================================================
#  Default run: the 4 uJ case of notebooks/filament_1030nm_4uJ.ipynb
# ================================================================================
if __name__ == "__main__":
    simulate()
