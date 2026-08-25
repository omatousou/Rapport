"""
Config: every tunable of the solver, plus the source fingerprint used to
detect a stale cached result.

Split out of the former single-file filament_sim.py. Imports only keldysh
(for n_sellmeier in __post_init__), so nothing else in the package can create
an import cycle through it.
"""

import sys
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.constants import c, epsilon_0, m_e
from scipy.constants import elementary_charge as q_e

sys.path.insert(0, str(Path(__file__).resolve().parent))
from keldysh import n_sellmeier  # noqa: E402


# Every source file of the solver. code_fingerprint() hashes all of them, so
# editing ANY module invalidates cached result.npz files -- listing only some
# would let a change in e.g. operators.py slip through unnoticed.
_SOURCE_FILES = ("config.py", "kernels.py", "grids.py", "operators.py",
                 "integrator.py", "filament_sim.py", "keldysh.py")


def code_fingerprint():
    """Short hash of every solver source file.

    Stamped into each params.json so a cached result.npz can be recognised as
    stale after the physics changes -- otherwise load_scenario_npz happily
    reloads a run computed by an older solver and the change appears to have
    had no effect."""
    import hashlib
    h = hashlib.sha256()
    here = Path(__file__).resolve().parent
    for name in _SOURCE_FILES:
        f = here / name
        if f.exists():
            h.update(f.read_bytes())
    return h.hexdigest()[:12]


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
    # Demi-largeur de la fenetre temporelle comobile, en unites de tp (largeur
    # 1/e du pulse), donc tmax = tmax_factor * tp et la fenetre totale = 2 tmax.
    # A 5.0 (defaut historique) elle ne couvre qu'environ +/-1.1 ps pour un
    # pulse de 263 fs -- trop court pour voir la decroissance des STE (~1 ps)
    # ou le plateau tardif : au-dela de tmax le cube n'a simplement plus de
    # donnees, rho_rzt/I_rzt se figent sur la derniere tranche simulee.
    # Augmenter ce facteur (ex. 12-15) elargit la fenetre ; penser a augmenter
    # Nt en proportion pour garder le meme pas dt (le cout par pas de z croit
    # avec Nt log Nt, la taille du cube (z,r,t) EXPORTE n'est pas affectee
    # tant que rho_t_stride est ajuste en consequence).
    tmax_factor: float = 5.0

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

    material_name: str = "fused silica (SiO2)"

    # ---- probe-side dielectric model (sim/permittivity.py) ----
    # These do NOT affect the propagation. They describe what the PROBE sees in
    # the excited medium, and are used after the run to turn rho_e and rho_s
    # into the phase and transmittance maps of the HTML pages.
    #
    # They live here so that _dump_params writes them into params.json and the
    # explorer builds the same model, instead of falling back to its own copy
    # of the defaults. Defaults are Table II of Martin, Guizard, Daguzan,
    # Petite et al., PRB 55, 5799 (1997), for SiO2.
    valence_N0_cm3: float = 2.2e22        # molecular density, ionizable units
    # Valence ELECTRONS per formula unit. N0 plays two different roles in the
    # paper under one symbol: the density of ionizable centres (2.2e22 for
    # SiO2) and the number of valence oscillators carrying the polarizability,
    # which is 8 per SiO2, four Si-O bonds of two electrons. The depletion
    # factor needs the second. Using the first overestimates that term
    # eightfold and flips the sign of the long-delay plateau.
    n_valence_per_unit: float = 8.0
    probe_meff_rel: float = 0.5           # conduction mass in the probe Drude term
    tau_ep_s: float = 1.0 / 1.5e15        # electron-phonon collisions, 0.67 fs
    # STE absorption bands: (resonance_eV, oscillator_strength, width_eV)
    ste_bands: tuple = ((5.2, 0.40, 1.5), (4.2, 0.15, 1.0))

    # ---- physics switch: linear step ----
    # The L^^2/2k0 piece of the dispersion bracket of Couairon 2005 Eq. (2),
    # (U^ + L^/2k0) L^. Fourth order in Omega and small: over a 350 um box it
    # is under 1e-5 rad between 900 and 1200 nm and 2e-3 rad at 700 nm, but it
    # reaches 0.74 rad at 400 nm, so it matters for the blue edge of a
    # supercontinuum and not for the pump. Omitted before this flag existed;
    # set False to reproduce runs made then.
    enable_dispersion_l2: bool = True

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
    # Sonde de l'experience Nomarski : 515 nm (2e harmonique du 1030 nm).
    # Valait 490e-9, ce qui n'etait la longueur d'onde d'aucune manip du
    # depot et se propageait silencieusement dans params.json puis dans tout
    # le post-traitement quand l'appelant oubliait de la passer.
    lambda_probe: float = 515e-9

    # ---- time-resolved rho snapshot ----
    rho_t_stride: int = 15
    # Radial subsampling of the same (z, r, t) cube. The cube is by far the
    # largest thing written (98% of a production result.npz: 8.3 GB of 8.4 GB
    # at Nz=6000/Nt=2048/Nr=3000/rho_t_stride=5), and the web explorer does
    # not need 3000 radial points for a filament ~100 points wide. 1 = keep
    # every radius, as before.
    rho_r_stride: int = 1
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


