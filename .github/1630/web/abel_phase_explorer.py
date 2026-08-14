"""
web/abel_phase_explorer.py
===========================
Generalisation of `unified_filament_slider_v3.py` (Abel-forward phase slider)
for the term-ablation study:

  - the Delta n(r, z, t) that feeds the Abel transform is split into THREE
    independently togglable channels -- Drude (free electrons), Lorentz/STE
    (self-trapped excitons), Kerr (n2 I) -- each Abel-transformed and NA-filtered
    SEPARATELY in Python (both operations are linear, so summing channels
    commutes with both the line-of-sight integral and the low-pass filter).
    The browser only needs to sum whichever channels are checked: no FFT /
    matrix multiply in JS.

  - several simulation scenario (e.g. the ablation-loop outputs from
    `notebooks/term_ablation_study.ipynb`: full physics, no-Kerr-Raman,
    no-plasma-defocusing, ...) can be loaded side by side and switched with a
    dropdown, to see how disabling a term in the solver changes the predicted
    interferometric phase shift.

  - the experimental side-phase preprocessing (holographic reconstruction +
    rotation + baseline fit) from `unified_filament_slider_v3.py` is kept
    UNCHANGED so real shots can still be overlaid; if `raw_dir=None` or the
    expected files are not found, the experiment panels are simply left empty
    with a note, and the page still works in sim-only mode.

Usage
-----
    from abel_phase_explorer import build_explorer_html
    build_explorer_html(
        sim_dirs={"full": "runs_ablation/full", "no_kerr_raman": "runs_ablation/no_kerr_raman"},
        save="abel_phase_explorer.html",
        raw_dir=None,               # or a path to the experimental npz shots
    )
"""

import json
from pathlib import Path

import numpy as np

try:
    from scipy.ndimage import rotate
except ImportError:  # experiment panel becomes unavailable, sim panel still works
    rotate = None

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sim"))
try:
    from permittivity import MaterialResponse, SIO2_MARTIN1997, XPM
except ImportError:      # module absent -> on retombe sur le calcul a la main
    MaterialResponse = SIO2_MARTIN1997 = None
    XPM = 2.0


# =============================================================================
# == CONFIG (experiment preprocessing -- identical to unified_filament_slider_v3) =
# =============================================================================

NA_REF                = 0.23
NA_REF_LMD_NM         = 515.0
RECONSTRUCTION_LMD_NM = None
PXSIZE_CAM_UM         = 3.45
M_SIDE                = 10.0
ROTATION_ANGLE_DEG    = -135.3
INVERT_SIDE_PHASE     = False
SIDE_BASELINE_RECTS = (
    ((100, 500), (130, 530)),
    ((100, 400), (130, 420)),
    ((650, 400), (720, 420)),
    ((600, 500), (700, 530)),
)

# --- Canal STE (Lorentz) ---------------------------------------------------
# ANCIENNE VALEUR : une bande unique a 5.8 eV avec une force d'oscillateur
# implicite de 1. Elle ne correspondait ni au solveur (Config.E_tr_eV = 4.2)
# ni a la litterature. Martin, Guizard, Daguzan, Petite et al., PRB 55, 5799
# (1997), Table II, donnent pour SiO2 DEUX bandes STE :
#     5.2 eV, f = 0.40, largeur 1.5 eV
#     4.2 eV, f = 0.15, largeur 1.0 eV
# soit, a 515 nm, un facteur effectif 0.177 la ou une bande unique f=1 a 4.2 eV
# donnait 0.489 : le canal STE etait 2.8 fois trop fort.
STE_LEVEL_EV = 4.2          # conserve pour le chemin historique uniquement
USE_PERMITTIVITY_MODEL = True   # False -> ancien calcul a la main

# --- Grille x de l'integrale Abel -------------------------------------------
X_SIM_HALF_UM = 50.0
DX_SIM_UM     = 0.2

# --- Pump group index (Sellmeier, silice) -----------------------------------
_SELLMEIER_B  = np.array([0.6961663, 0.4079426, 0.8974794])
_SELLMEIER_L2 = np.array([0.0684043, 0.1162414, 9.896161]) ** 2
_C_UM_FS = 299792458.0 * 1e-9  # um/fs
_C_M_S = 299792458.0
_EPS0 = 8.8541878128e-12
_M_E = 9.1093837139e-31
_Q_E = 1.602176634e-19


def _n_sellmeier(lam_um):
    n2m1 = sum(B * lam_um**2 / (lam_um**2 - L2) for B, L2 in zip(_SELLMEIER_B, _SELLMEIER_L2))
    return np.sqrt(1.0 + n2m1)


def group_index(lam_um, d=1e-4):
    n0 = _n_sellmeier(lam_um)
    dn = (_n_sellmeier(lam_um + d) - _n_sellmeier(lam_um - d)) / (2 * d)
    return n0 - lam_um * dn


def calc_NA(lmd_nm):
    return float(NA_REF) * float(lmd_nm) / float(NA_REF_LMD_NM)


def probe_optics(lmd_nm):
    """n0 et densite critique de la sonde a `lmd_nm`.

    Le HTML peut etre genere a plusieurs longueurs d'onde de sonde depuis un
    meme run de pompe. Dans ce cas il faut recalculer ces deux grandeurs, et ne
    pas relire aveuglement la valeur historisee dans params.json.
    """
    lmd_nm = float(lmd_nm)
    lmd_m = lmd_nm * 1e-9
    omega = 2.0 * np.pi * _C_M_S / lmd_m
    n0_probe = float(_n_sellmeier(lmd_nm * 1e-3))
    nc_probe_cm3 = float(_EPS0 * _M_E * omega**2 / _Q_E**2 * 1e-6)
    return n0_probe, nc_probe_cm3


# =============================================================================
# == PREPROCESS (experience, 1:1 depuis unified_filament_slider_v3.py) =======
# =============================================================================

def fmt_delay(p):
    x = float(p); r = round(x)
    return f"{int(r):+d}" if np.isclose(x, r) else f"{x:+g}"


def fmt_probe(lmd):
    return f"{float(lmd):g}"


def raw_filename(energy_uJ, delay_pulse, lmd_nm):
    return f"{float(energy_uJ):.1f}uJ_{fmt_delay(delay_pulse)}pulse_{fmt_probe(lmd_nm)}nm.npz"


def find_1storder_peak(img, remove_dc=100):
    img_fft = np.fft.fftshift(np.fft.fft2(img))
    ny, nx = img_fft.shape
    cy0, cx0 = ny // 2, nx // 2
    mag = np.abs(img_fft).copy()
    Y, X = np.ogrid[:ny, :nx]
    mag[np.sqrt((X - cx0)**2 + (Y - cy0)**2) < remove_dc] = 0
    mag[:cy0, :] = 0
    py, px = np.unravel_index(np.argmax(mag), mag.shape)
    return int(py), int(px)


def reconstruct(img, peak_y, peak_x, radius_px):
    img_fft = np.fft.fftshift(np.fft.fft2(img))
    ny, nx = img_fft.shape
    Y, X = np.ogrid[:ny, :nx]
    mask = np.sqrt((X - peak_x)**2 + (Y - peak_y)**2) <= radius_px
    img_fft = img_fft * mask
    img_fft_cropped = img_fft[
        peak_y - radius_px: peak_y + radius_px,
        peak_x - radius_px: peak_x + radius_px,
    ]
    field = np.fft.ifft2(np.fft.ifftshift(img_fft_cropped))
    _, nx_rec = img_fft_cropped.shape
    pxsize_rec_um = PXSIZE_CAM_UM * nx / nx_rec
    return field, pxsize_rec_um


def reconstruction_radius_px(img_nx, magnification, lmd_um, rec_NA):
    radius_umi = float(rec_NA) / (magnification * lmd_um)
    return int(radius_umi / (1 / (img_nx * PXSIZE_CAM_UM)))


def baseline_rects_to_mask(shape, rects):
    h, w = shape
    mask = np.zeros(shape, dtype=bool)
    for (x1, y1), (x2, y2) in rects:
        x0 = max(0, int(round(min(x1, x2))))
        y0 = max(0, int(round(min(y1, y2))))
        x3 = min(w, int(round(max(x1, x2))))
        y3 = min(h, int(round(max(y1, y2))))
        if x3 > x0 and y3 > y0:
            mask[y0:y3, x0:x3] = True
    return mask


def fit_plane_from_mask(img, mask):
    h, w = img.shape
    Y, X = np.indices((h, w))
    x = X[mask].ravel(); y = Y[mask].ravel(); z = img[mask].ravel()
    if len(z) < 3:
        raise ValueError("baseline mask has too few points")
    A = np.c_[x, y, np.ones_like(x)]
    coeff, *_ = np.linalg.lstsq(A, z, rcond=None)
    a, b, c = coeff
    return a * X + b * Y + c


def preprocess_side_full(npz_path, lmd_nm):
    if rotate is None:
        raise RuntimeError("scipy.ndimage.rotate unavailable -- experiment panel disabled")
    data = np.load(npz_path)
    img_side    = data["side_sig"][:2048, :2048]
    img_side_bg = data["side_bg"] [:2048, :2048]

    lmd_rec_nm = float(lmd_nm) if RECONSTRUCTION_LMD_NM is None else float(RECONSTRUCTION_LMD_NM)
    rec_NA = calc_NA(lmd_rec_nm)
    lmd_rec_um = lmd_rec_nm * 1e-3

    py, px = find_1storder_peak(img_side)
    _, nx = img_side.shape
    radius_px = reconstruction_radius_px(nx, M_SIDE, lmd_rec_um, rec_NA)

    rec,    pxsize_rec_um = reconstruct(img_side,    py, px, radius_px)
    rec_bg, _              = reconstruct(img_side_bg, py, px, radius_px)

    field = rec / rec_bg
    field_rot = rotate(field, ROTATION_ANGLE_DEG, reshape=True, order=1, cval=0, mode="constant")
    phase = np.angle(field_rot)

    mask  = baseline_rects_to_mask(phase.shape, SIDE_BASELINE_RECTS)
    plane = fit_plane_from_mask(phase, mask)
    phase = phase - plane

    if INVERT_SIDE_PHASE:
        phase = -phase

    s_um_per_px = pxsize_rec_um / M_SIDE
    return phase, s_um_per_px, rec_NA


# =============================================================================
# == SIM : chargement + canaux Delta n separes + Abel forward =================
# =============================================================================

def load_sim(sim_dir):
    d = np.load(str(Path(sim_dir) / "result.npz"), allow_pickle=True)
    if "rho_rzt" not in d.files or np.asarray(d["rho_rzt"]).shape == ():
        raise RuntimeError(
            f"{sim_dir}/result.npz n'a pas de rho_rzt exploitable "
            "(relance filament_sim.run(..., rho_t_stride>0)).")
    # Le cube (z,r,t) peut etre sous-echantillonne radialement (rho_r_stride) :
    # dans ce cas le solveur ecrit r_sub, et c'est CETTE grille qui indexe
    # rho_rzt/I_rzt, pas rlist.
    if "r_sub" in d.files and np.asarray(d["r_sub"]).shape != ():
        rlist_m = np.asarray(d["r_sub"], dtype=np.float64)
    elif "rlist" in d.files:
        rlist_m = np.asarray(d["rlist"], dtype=np.float64)
    else:
        rlist_m = np.asarray(d["r"], dtype=np.float64)[len(d["r"]) // 2:]
    sim = dict(
        rlist_um   = rlist_m * 1e6,
        z_sim_um   = np.asarray(d["z"], dtype=np.float64) * 1e6,
        rho_rzt    = np.asarray(d["rho_rzt"],   dtype=np.float32),
        rho_s_rzt  = (np.asarray(d["rho_s_rzt"], dtype=np.float32)
                      if "rho_s_rzt" in d.files and np.asarray(d["rho_s_rzt"]).shape != ()
                      else None),
        I_rzt      = np.asarray(d["I_rzt"], dtype=np.float32),
        t_sub_fs   = np.asarray(d["t_sub_fs"], dtype=np.float64),
    )
    with open(Path(sim_dir) / "params.json", "r") as f:
        sim["params"] = json.load(f)
    return sim


def build_abel_matrix(rlist_um, x_um):
    rlist_um = np.asarray(rlist_um, dtype=np.float64)
    x_um     = np.asarray(x_um,     dtype=np.float64)
    n_r = len(rlist_um); n_x = len(x_um)

    r_mid = 0.5 * (rlist_um[:-1] + rlist_um[1:])
    r_lo  = np.concatenate(([0.0], r_mid))
    last_hi = rlist_um[-1] + (rlist_um[-1] - r_mid[-1])
    r_hi  = np.concatenate((r_mid, [last_hi]))

    A = np.zeros((n_x, n_r), dtype=np.float64)
    abs_x = np.abs(x_um)[:, None]
    r_hi_b = r_hi[None, :]
    r_lo_b = r_lo[None, :]

    upper = r_hi_b**2 - abs_x**2
    lower_r = np.maximum(r_lo_b, abs_x)
    lower = lower_r**2 - abs_x**2

    valid = r_hi_b >= abs_x
    A[valid] = 2.0 * (np.sqrt(np.maximum(upper[valid], 0.0)) - np.sqrt(np.maximum(lower[valid], 0.0)))
    return A


def lowpass_NA_2d(phi, d_axis0_um, d_axis1_um, NA, lmd_um):
    phi = np.asarray(phi, dtype=float)
    nan_mask = ~np.isfinite(phi)
    fill = np.where(nan_mask, 0.0, phi)
    n0, n1 = fill.shape
    k0 = 2 * np.pi * np.fft.fftfreq(n0, d=d_axis0_um)
    k1 = 2 * np.pi * np.fft.fftfreq(n1, d=d_axis1_um)
    K0, K1 = np.meshgrid(k0, k1, indexing="ij")
    k_max = 2 * np.pi * float(NA) / float(lmd_um)
    mask = (K0**2 + K1**2) <= k_max**2
    out = np.real(np.fft.ifft2(np.fft.fft2(fill) * mask))
    out[nan_mask] = np.nan
    return out


def auto_t0_ref_um(sim):
    """z (repere labo, meme convention que z_sim_um) ou la pompe est la plus
    intense sur l'axe -- le point de collapse.

    C'est LA reference physiquement sensee pour t=0 : "delai nul = coincidence
    des maxima pompe/sonde" (comme dans virtual_experiment.py), pas
    l'entree de la boite, qui n'a rien de particulier une fois le faisceau
    focalise a l'interieur.
    """
    I_rzt = sim["I_rzt"]
    if I_rzt is None or np.asarray(I_rzt).shape == ():
        return float(sim["z_sim_um"][0])
    I_onaxis_max_z = np.asarray(I_rzt[:, 0, :], dtype=np.float64).max(axis=1)
    return float(sim["z_sim_um"][int(np.argmax(I_onaxis_max_z))])


def channel_phases_2d(sim, t_exp_fs, *,
                       apply_na_filter=True, NA_eff=None, lmd_um=None,
                       probe_lmd_nm=None, t0_ref_um=None,
                       mask_before_interface=True,
                       x_sim_half_um=X_SIM_HALF_UM, dx_sim_um=DX_SIM_UM):
    """
    Retourne (z_lab_um, x_um, {"drude":phi, "kerr":phi, "ste":phi})
    -- une carte de phase (Nz, Nx) deja transformee par Abel et filtree NA,
    PAR CANAL, prete a etre sommee lineairement cote navigateur.

    `t0_ref_um` : position z (repere simulation, avant decalage vers le
    labo) ou t_exp_fs=0 signifie "pompe et sonde coincident". Par defaut
    (None) c'est le point de collapse (voir `auto_t0_ref_um`), pas l'entree
    de la boite -- la sonde n'a aucune raison physique de coincider avec la
    pompe a z=0 plutot qu'au point ou l'intensite est maximale.
    """
    rlist_um = sim["rlist_um"]; z_sim_um = sim["z_sim_um"]; t_sub_fs = sim["t_sub_fs"]
    params   = sim["params"]

    lmd_probe_nm = float(probe_lmd_nm if probe_lmd_nm is not None
                         else params.get("lambda_probe_nm", 515.0))
    if probe_lmd_nm is not None or "n0_probe" not in params or "nc_probe_cm3" not in params:
        n0_probe, nc_probe_cm3 = probe_optics(lmd_probe_nm)
    else:
        n0_probe     = float(params["n0_probe"])
        nc_probe_cm3 = float(params["nc_probe_cm3"])
    n2_m2W       = float(params.get("n2", 2.4e-20))
    z_focus_glass_dist_um = -float(params.get("begin_um", 0.0))

    lam_pump_um   = float(params.get("wavelength_nm", 1030.0)) * 1e-3
    n_group_pump  = group_index(lam_pump_um)
    v_g_pump_um_fs = _C_UM_FS / n_group_pump

    z_ref_um = float(t0_ref_um if t0_ref_um is not None else auto_t0_ref_um(sim))
    t_local = t_exp_fs - (z_sim_um - z_ref_um) / v_g_pump_um_fs
    k_t = np.clip(np.searchsorted(t_sub_fs, t_local), 1, len(t_sub_fs) - 1)
    left  = np.abs(t_sub_fs[k_t - 1] - t_local)
    right = np.abs(t_sub_fs[k_t]     - t_local)
    k_t = np.where(left < right, k_t - 1, k_t)

    Nz = len(z_sim_um)
    iz = np.arange(Nz)

    rho_e_rz = sim["rho_rzt"][iz, :, k_t].astype(np.float64)
    has_ste  = sim["rho_s_rzt"] is not None
    rho_s_rz = sim["rho_s_rzt"][iz, :, k_t].astype(np.float64) if has_ste else np.zeros_like(rho_e_rz)
    I_rz     = sim["I_rzt"][iz, :, k_t].astype(np.float64)

    lmd_probe_m = lmd_probe_nm * 1e-9
    alpha_cm_rz = None
    if USE_PERMITTIVITY_MODEL and SIO2_MARTIN1997 is not None:
        mat = MaterialResponse(n2_m2W=n2_m2W)
        z0 = np.zeros_like(rho_e_rz)

        def _dn(rho_e=z0, rho_s=z0, I=z0, inc=()):
            return mat.response(lmd_probe_m, n0_probe, rho_e_cm3=rho_e,
                                rho_s_cm3=rho_s, I_Wcm2=I, xpm_factor=XPM,
                                include=inc)

        r_drude = _dn(rho_e=rho_e_rz, inc=("drude",))
        r_depl = _dn(rho_e=rho_e_rz + (rho_s_rz if has_ste else z0),
                     inc=("depletion",))
        r_ste = _dn(rho_s=rho_s_rz, inc=("ste",)) if has_ste else None
        r_kerr = _dn(I=I_rz, inc=("kerr",))

        dn_channels = {
            "drude": np.asarray(r_drude["dn"]) + np.asarray(r_depl["dn"]),
            "ste": (np.asarray(r_ste["dn"]) if has_ste else z0),
            "kerr": np.asarray(r_kerr["dn"]),
        }
        alpha_cm_rz = (np.asarray(r_drude["alpha_cm"])
                       + (np.asarray(r_ste["alpha_cm"]) if has_ste else 0.0))
        f_STE = mat.f_ste_effective(lmd_probe_m)
    else:
        E_probe = 1240.0 / lmd_probe_nm
        f_STE = E_probe**2 / (STE_LEVEL_EV**2 - E_probe**2)
        dn_channels = {
            "drude":   -rho_e_rz / (2.0 * n0_probe * nc_probe_cm3),
            "ste":     (f_STE * rho_s_rz / (2.0 * n0_probe * nc_probe_cm3)) if has_ste else np.zeros_like(rho_e_rz),
            "kerr":    n2_m2W * I_rz * 1.0e4,
        }

    x_max = float(min(x_sim_half_um, rlist_um[-1]))
    n_x   = int(2 * x_max / dx_sim_um) + 1
    x_um  = np.linspace(-x_max, x_max, n_x)
    A = build_abel_matrix(rlist_um, x_um)

    lmd_p_um = lmd_probe_nm * 1e-3
    z_lab_um = z_sim_um + z_focus_glass_dist_um
    air = z_lab_um < 0.0 if mask_before_interface else np.zeros_like(z_lab_um, dtype=bool)

    dz_sim = float(np.mean(np.diff(z_lab_um)))
    dx_sim = float(np.mean(np.diff(x_um)))
    NA_eff = NA_eff if NA_eff is not None else calc_NA(lmd_probe_nm)
    lmd_um = lmd_um if lmd_um is not None else lmd_p_um

    phases = {}
    for name, dn in dn_channels.items():
        phi = (2.0 * np.pi / lmd_p_um) * (dn @ A.T)
        if apply_na_filter:
            phi = lowpass_NA_2d(phi, dz_sim, dx_sim, NA_eff, lmd_um)
        phi[air, :] = np.nan
        phases[name] = phi

    # L'experience mesure la partie REELLE et la partie IMAGINAIRE de l'indice
    # (dephasage et contraste de frange). Jusqu'ici cette page ne calculait que
    # la premiere. La transmittance vient du meme eps que la phase, donc les
    # deux ne peuvent plus se contredire.
    if alpha_cm_rz is not None:
        tau = (alpha_cm_rz @ A.T) * 1e-4            # cm^-1 integre sur des µm
        T = np.exp(-np.clip(tau, 0.0, None))
        if apply_na_filter:
            T = lowpass_NA_2d(T, dz_sim, dx_sim, NA_eff, lmd_um)
        T[air, :] = np.nan
        phases["transmittance"] = T

    # Densites de colonne (Abel-projetees, filtrees NA comme la phase) : ce
    # qu'un imageur "verrait" s'il pouvait voir la densite -- PAS la densite
    # reelle en un point, une integrale de ligne de visee en cm^-3.um. Sert
    # de pendant, en (z, x), a la carte (z, r) brute de density_maps_2d :
    # meme convention d'axes que le panneau de phase, au choix cote HTML.
    def _col_log(rho_rz, floor=1.0):
        if rho_rz is None:
            return None
        col = rho_rz @ A.T
        if apply_na_filter:
            col = lowpass_NA_2d(col, dz_sim, dx_sim, NA_eff, lmd_um)
        out = np.full_like(col, np.nan)
        m = col >= floor
        out[m] = np.log10(col[m])
        out[air, :] = np.nan
        return out

    phases["rho_e_col"] = _col_log(rho_e_rz)
    phases["rho_s_col"] = _col_log(rho_s_rz) if has_ste else None

    return z_lab_um, x_um, phases


# =============================================================================
# == Helpers communs ==========================================================
# =============================================================================

def _to_json_array(a, decimals=4):
    a = np.asarray(a)
    if np.issubdtype(a.dtype, np.floating):
        a = np.round(a.astype(np.float64), decimals)
        out = np.where(np.isfinite(a), a, None)
        return out.tolist()
    return a.tolist()


def density_maps_2d(sim, t_exp_fs, t0_ref_um=None, log_floor_cm3=1e12):
    """rho_e(z, r) et rho_STE(z, r) [log10 cm^-3] AU DELAI `t_exp_fs`.

    Meme selection temporelle que `channel_phases_2d` (t=0 au collapse, pas a
    l'entree de la boite), mais SANS transformee d'Abel : c'est la densite
    reelle dans le plan meridien (z, r) -- un instantane, pas un historique
    complet -- de sorte que ce panneau reagisse au curseur de delai exactement
    comme le panneau de phase, au lieu de montrer tout le temps a la fois avec
    une simple ligne indicatrice.
    """
    z_sim_um = sim["z_sim_um"]; r_um = sim["rlist_um"]; t_sub_fs = sim["t_sub_fs"]
    params = sim["params"]

    lam_pump_um = float(params.get("wavelength_nm", 1030.0)) * 1e-3
    v_g_pump_um_fs = _C_UM_FS / group_index(lam_pump_um)
    z_focus_glass_dist_um = -float(params.get("begin_um", 0.0))
    z_ref_um = float(t0_ref_um if t0_ref_um is not None else auto_t0_ref_um(sim))

    t_local = t_exp_fs - (z_sim_um - z_ref_um) / v_g_pump_um_fs
    k_t = np.clip(np.searchsorted(t_sub_fs, t_local), 1, len(t_sub_fs) - 1)
    left  = np.abs(t_sub_fs[k_t - 1] - t_local)
    right = np.abs(t_sub_fs[k_t]     - t_local)
    k_t = np.where(left < right, k_t - 1, k_t)
    iz = np.arange(len(z_sim_um))

    def _log(cube):
        if cube is None or np.asarray(cube).shape == ():
            return None
        a = np.asarray(cube[iz, :, k_t], dtype=np.float64)   # (Nz, Nr)
        out = np.full_like(a, np.nan)
        m = a >= float(log_floor_cm3)
        out[m] = np.log10(a[m])
        return out

    z_lab_um = z_sim_um + z_focus_glass_dist_um
    return (z_lab_um, r_um,
            _log(sim.get("rho_rzt")), _log(sim.get("rho_s_rzt")))


# =============================================================================
# == HTML output ===============================================================
# =============================================================================

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>Abel phase explorer -- ablation des termes</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  body { font-family: -apple-system, Segoe UI, sans-serif; margin: 16px; color: #222; }
  .controls { padding: 10px 14px; background: #f4f4f7; border-radius: 6px;
              margin-bottom: 10px; display: flex; flex-wrap: wrap;
              gap: 18px; align-items: center; }
  .controls label { font-size: 14px; }
  .controls input[type=range]  { width: 280px; vertical-align: middle; }
  .channels label { margin-right: 10px; }
  #status { font-size: 13px; color: #555; margin-left: auto; }
  #plot { width: 100%; height: __FIGURE_HEIGHT__px; }
  #densityPlot { width: 100%; height: 540px; margin-top: 12px; }
  #note { font-size: 12px; color: #888; }
</style>
</head>
<body>

<h3 style="margin:6px 0 12px 0">Abel phase explorer — canaux de Δn togglables, scénarios d'ablation</h3>

<div class="controls">
  <label>Scénario:
    <select id="scenario"></select></label>
  <label>Pulse:
    <input type="range" id="pulseSlider" min="0" max="__PULSE_MAX__" value="0">
    <span id="pulseLabel" style="display:inline-block; width:180px;">__INIT_LABEL__</span>
  </label>
  <span id="status"></span>
</div>
<div class="controls channels">
  <b>Canaux Δn :</b>
  <label><input type="checkbox" id="ch_drude" checked> Drude (électrons libres, &lt;0)</label>
  <label><input type="checkbox" id="ch_kerr" checked> Kerr (n2·I, &gt;0)</label>
  <label><input type="checkbox" id="ch_ste" checked> STE / Lorentz (excitons piégés)</label>
  <b style="margin-left: 20px;">Densités :</b>
  <label><input type="checkbox" id="cb_rho_e" checked> Électrons</label>
  <label><input type="checkbox" id="cb_rho_s" checked> STE</label>
  <label style="margin-left: 10px;">
    <input type="checkbox" id="cb_rho_abel"> vue colonne Abel (comme la phase, axe x)
  </label>
</div>

<div id="plot"></div>
<div id="densityPlot"></div>
<p id="note">Δφ = Abel-forward de Δn, filtré NA de la sonde (pas de recalcul FFT côté navigateur : chaque canal est déjà transformé par Abel + filtré côté Python ; le navigateur ne fait que sommer les canaux cochés).</p>

<script>
const DATA   = __DATA_JSON__;
const LAYOUT = __LAYOUT_JSON__;
const DENSITY_LAYOUT = __DENSITY_LAYOUT_JSON__;
const META   = __META_JSON__;
const HAS_EXP = __HAS_EXP__;
const HAS_DENSITY = __HAS_DENSITY__;

const scenarioSel = document.getElementById('scenario');
Object.keys(DATA.scenarios).forEach(name => {
  const opt = document.createElement('option');
  opt.value = name; opt.textContent = name;
  scenarioSel.appendChild(opt);
});

let pulseIdx = 0;

function sumChannels(pulse) {
  const chosen = [];
  if (document.getElementById('ch_drude').checked && pulse.channels.drude) chosen.push(pulse.channels.drude);
  if (document.getElementById('ch_kerr').checked && pulse.channels.kerr) chosen.push(pulse.channels.kerr);
  if (document.getElementById('ch_ste').checked && pulse.channels.ste) chosen.push(pulse.channels.ste);
  if (chosen.length === 0) return null;
  const n0 = chosen[0].length, n1 = chosen[0][0].length;
  const out = new Array(n0);
  for (let i = 0; i < n0; i++) {
    out[i] = new Array(n1).fill(0);
    for (const arr of chosen) {
      const row = arr[i];
      for (let j = 0; j < n1; j++) {
        const v = row[j];
        out[i][j] = (out[i][j] === null || v === null) ? null : out[i][j] + v;
      }
    }
  }
  return out;
}

function lineoutOnAxis(phi2d, xCoord, bandUm) {
  if (!phi2d) return [];
  const N1 = phi2d.length, N2 = phi2d[0].length;
  const line = new Array(N1).fill(null);
  const jMid = Math.floor(N2 / 2);
  const jLo = Math.max(0, jMid - bandUm), jHi = Math.min(N2, jMid + bandUm + 1);
  for (let i = 0; i < N1; i++) {
    let s = 0, c = 0;
    for (let j = jLo; j < jHi; j++) {
      const v = phi2d[i][j];
      if (v !== null && isFinite(v)) { s += v; c++; }
    }
    line[i] = c > 0 ? s / c : null;
  }
  return line;
}

function buildTraces() {
  const scen = DATA.scenarios[scenarioSel.value];
  const pulse = scen.pulses[Math.min(pulseIdx, scen.pulses.length - 1)];
  const phi2d = sumChannels(pulse);
  const nz = phi2d ? phi2d.length : 0;
  const nx = phi2d ? phi2d[0].length : 0;
  const phiT = phi2d ? Array.from({length: nx}, (_, j) => phi2d.map(row => row[j])) : [];
  const lineSim = phi2d ? phi2d.map(row => {
    const jm = Math.floor(row.length / 2);
    return row[jm];
  }) : [];

  const traces = [
    { type: 'heatmap', x: scen.z_sim, y: scen.x_sim, z: phiT,
      colorscale: 'RdBu', reversescale: false, zmid: 0,
      zmin: -META.clip, zmax: META.clip,
      colorbar: { title: 'δφ (rad)', len: 0.85 },
      hovertemplate: 'z=%{x:.0f} µm<br>x=%{y:.0f} µm<br>δφ=%{z:.3f} rad<extra></extra>',
      xaxis: 'x', yaxis: 'y' },
    { type: 'scatter', x: scen.z_sim, y: lineSim, mode: 'lines',
      line: { color: '#1f77b4', width: 2 }, showlegend: false,
      xaxis: 'x2', yaxis: 'y2',
      hovertemplate: 'z=%{x:.0f} µm<br>δφ on-axis=%{y:.3f} rad<extra></extra>' },
  ];
  if (pulse.channels.transmittance) {
    const T = pulse.channels.transmittance;
    const nxT = T[0].length;
    const TT = Array.from({length: nxT}, (_, j) => T.map(row => row[j]));
    traces.push({ type: 'heatmap', x: scen.z_sim, y: scen.x_sim, z: TT,
      colorscale: 'Greys', reversescale: true, zmin: META.tmin, zmax: 1.0,
      colorbar: { title: 'T', len: 0.28, y: 0.14 },
      hovertemplate: 'z=%{x:.0f} µm<br>x=%{y:.0f} µm<br>T=%{z:.3f}<extra></extra>',
      xaxis: 'x3', yaxis: 'y3' });
  }
  return traces;
}

function transposeZr(arr) {
  // (Nz, Nr) -> (Nr, Nz), pour heatmap(x=z, y=r)
  if (!arr) return [];
  const nr = arr[0].length;
  return Array.from({length: nr}, (_, j) => arr.map(row => row[j]));
}

function buildDensityTraces() {
  // Instantane au delai DU PULSE COURANT -- pas un historique fixe : ce
  // panneau doit reagir au curseur exactement comme le panneau de phase.
  //
  // Deux vues possibles (case a cocher "vue colonne Abel") :
  //   - (z, r) brut : la densite REELLE dans le plan meridien, telle que
  //     calculee par le solveur -- pas ce qu'une camera peut voir.
  //   - (z, x) Abel-projete, filtre NA : une integrale de ligne de visee en
  //     cm^-3.µm, comme la phase -- ce qu'un imageur "verrait" s'il pouvait
  //     voir la densite, axes coherents avec le panneau du haut.
  const scen = DATA.scenarios[scenarioSel.value];
  const pulse = scen.pulses[Math.min(pulseIdx, scen.pulses.length - 1)];
  const useAbel = document.getElementById('cb_rho_abel').checked;

  const show_rho_e = document.getElementById('cb_rho_e').checked;
  const show_rho_s = document.getElementById('cb_rho_s').checked;
  const traces = [];

  if (useAbel) {
    if (!pulse.rho_e_col) return [];
    if (show_rho_e) {
      traces.push(
        { type: 'heatmap', x: scen.z_sim, y: scen.x_sim, z: transposeZr(pulse.rho_e_col),
          colorscale: 'Blues', reversescale: false, zmin: META.rho_col_log_min, zmax: META.rho_col_log_max,
          colorbar: { title: 'log10 ∫ρe dx', len: 0.42, y: 0.76 },
          hovertemplate: 'z=%{x:.0f} µm<br>x=%{y:.0f} µm<br>log10 ∫ρe dx=%{z:.2f}<extra></extra>',
          xaxis: 'x', yaxis: 'y' }
      );
    }
    if (show_rho_s && pulse.rho_s_col) {
      traces.push(
        { type: 'heatmap', x: scen.z_sim, y: scen.x_sim, z: transposeZr(pulse.rho_s_col),
          colorscale: 'Greens', reversescale: false, zmin: META.rho_col_log_min, zmax: META.rho_col_log_max,
          colorbar: { title: 'log10 ∫ρSTE dx', len: 0.42, y: 0.23 },
          hovertemplate: 'z=%{x:.0f} µm<br>x=%{y:.0f} µm<br>log10 ∫ρSTE dx=%{z:.2f}<extra></extra>',
          xaxis: 'x2', yaxis: 'y2' }
      );
    }
    return traces;
  }

  if (!scen.r_dens || !pulse.rho_e_map) return [];
  if (show_rho_e) {
    traces.push(
      { type: 'heatmap', x: scen.z_sim, y: scen.r_dens, z: transposeZr(pulse.rho_e_map),
        colorscale: 'Blues', reversescale: false, zmin: META.rho_log_min, zmax: META.rho_log_max,
        colorbar: { title: 'log10 ρe', len: 0.42, y: 0.76 },
        hovertemplate: 'z=%{x:.0f} µm<br>r=%{y:.0f} µm<br>log10 ρe=%{z:.2f}<extra></extra>',
        xaxis: 'x', yaxis: 'y' }
    );
  }
  if (show_rho_s && pulse.rho_s_map) {
    traces.push(
      { type: 'heatmap', x: scen.z_sim, y: scen.r_dens, z: transposeZr(pulse.rho_s_map),
        colorscale: 'Greens', reversescale: false, zmin: META.rho_log_min, zmax: META.rho_log_max,
        colorbar: { title: 'log10 ρSTE', len: 0.42, y: 0.23 },
        hovertemplate: 'z=%{x:.0f} µm<br>r=%{y:.0f} µm<br>log10 ρSTE=%{z:.2f}<extra></extra>',
        xaxis: 'x2', yaxis: 'y2' }
    );
  }
  return traces;
}

function densityLayoutFor(useAbel) {
  // (z, r) brut : r >= 0 seul, titre "r". (z, x) Abel : x symetrique, titre "x".
  const scen = DATA.scenarios[scenarioSel.value];
  const L = JSON.parse(JSON.stringify(DENSITY_LAYOUT));
  if (useAbel && scen.x_sim && scen.x_sim.length) {
    const xMax = Math.max(...scen.x_sim.map(Math.abs));
    L.yaxis.range = [-xMax, xMax]; L.yaxis.title = 'x (µm) — colonne ρe';
    L.yaxis2.range = [-xMax, xMax]; L.yaxis2.title = 'x (µm) — colonne ρSTE';
  }
  return L;
}

function render() {
  const scen = DATA.scenarios[scenarioSel.value];
  const pulse = scen.pulses[Math.min(pulseIdx, scen.pulses.length - 1)];
  Plotly.react('plot', buildTraces(), LAYOUT, {responsive: true});

  const show_rho_e = document.getElementById('cb_rho_e').checked;
  const show_rho_s = document.getElementById('cb_rho_s').checked;
  const useAbel = document.getElementById('cb_rho_abel').checked;

  if (HAS_DENSITY && (show_rho_e || show_rho_s)) {
    Plotly.react('densityPlot', buildDensityTraces(), densityLayoutFor(useAbel), {responsive: true});
    document.getElementById('densityPlot').style.display = 'block';
  } else {
    document.getElementById('densityPlot').style.display = 'none';
  }

  const sign = pulse.p > 0 ? '+' : '';
  document.getElementById('pulseLabel').textContent =
    `pulse ${sign}${pulse.p}  (Δt = ${pulse.t_exp.toFixed(0)} fs)`;
  document.getElementById('status').textContent =
    `scénario = ${scenarioSel.value} | sonde = ${META.probe_nm[scenarioSel.value].toFixed(0)} nm | ` +
    `z_foyer_gauss = ${META.z_focus[scenarioSel.value].toFixed(0)} µm | ` +
    `t=0 (collapse) a z = ${META.t0_ref[scenarioSel.value].toFixed(0)} µm`;
}

scenarioSel.addEventListener('change', () => {
  pulseIdx = Math.min(pulseIdx, DATA.scenarios[scenarioSel.value].pulses.length - 1);
  document.getElementById('pulseSlider').max = DATA.scenarios[scenarioSel.value].pulses.length - 1;
  render();
});
document.getElementById('pulseSlider').addEventListener('input', e => { pulseIdx = +e.target.value; render(); });
['ch_drude', 'ch_kerr', 'ch_ste', 'cb_rho_e', 'cb_rho_s', 'cb_rho_abel'].forEach(id =>
  document.getElementById(id).addEventListener('change', render));

render();
</script>

</body>
</html>
"""


def build_layout(xlim, ylim, with_transmittance=True):
    """Trois lignes quand la transmittance est disponible : carte de phase,
    coupe on-axis, carte de transmittance. L'experience mesure les deux parties
    de l'indice, la page doit montrer les deux."""
    if not with_transmittance:
        return {
            "template": "plotly_white",
            "margin": {"l": 70, "r": 30, "t": 20, "b": 50},
            "height": 700,
            "xaxis":  {"domain": [0.0, 1.0], "anchor": "y", "range": xlim, "matches": "x2"},
            "yaxis":  {"domain": [0.42, 1.0], "anchor": "x", "range": list(ylim), "title": "x (µm)"},
            "xaxis2": {"domain": [0.0, 1.0], "anchor": "y2", "range": xlim,
                       "title": "Propagation z (µm) — lab frame (0 = interface)"},
            "yaxis2": {"domain": [0.0, 0.38], "anchor": "x2", "title": "δφ on-axis (rad)"},
        }
    return {
        "template": "plotly_white",
        "margin": {"l": 70, "r": 30, "t": 20, "b": 50},
        "height": 900,
        "xaxis":  {"domain": [0.0, 1.0], "anchor": "y", "range": xlim, "matches": "x3"},
        "yaxis":  {"domain": [0.62, 1.0], "anchor": "x", "range": list(ylim), "title": "x (µm) — δφ"},
        "xaxis2": {"domain": [0.0, 1.0], "anchor": "y2", "range": xlim, "matches": "x3"},
        "yaxis2": {"domain": [0.34, 0.56], "anchor": "x2", "title": "δφ on-axis (rad)"},
        "xaxis3": {"domain": [0.0, 1.0], "anchor": "y3", "range": xlim,
                   "title": "Propagation z (µm) — lab frame (0 = interface)"},
        "yaxis3": {"domain": [0.0, 0.28], "anchor": "x3", "range": list(ylim),
                   "title": "x (µm) — T"},
    }

def build_density_layout(xlim, rlim):
    """Layout du panneau densites : instantane (z, r) au delai courant,
    electrons en haut / STE en bas -- pas un historique (z, t) fixe."""
    return {
        "template": "plotly_white",
        "margin": {"l": 70, "r": 30, "t": 20, "b": 50},
        "height": 600,
        "xaxis":  {"domain": [0.0, 1.0], "anchor": "y", "range": xlim, "matches": "x2", "title": "Propagation z (µm) — lab frame"},
        "yaxis":  {"domain": [0.55, 1.0], "anchor": "x", "range": list(rlim), "title": "r (µm) — ρe"},
        "xaxis2": {"domain": [0.0, 1.0], "anchor": "y2", "range": xlim, "matches": "x"},
        "yaxis2": {"domain": [0.0, 0.45], "anchor": "x2", "range": list(rlim), "title": "r (µm) — ρSTE"},
    }
def run_slider_scenario(sim_dir, pmin, pmax, fs_per_pulse, lmd_nm, apply_na_filter,
                         coarsen_z=1, coarsen_r=1):
    sim = load_sim(sim_dir)
    if lmd_nm is None:
        lmd_nm = float(sim["params"].get("lambda_probe_nm", 515.0))

    if coarsen_z > 1:
        sim["z_sim_um"]  = sim["z_sim_um"][::coarsen_z]
        sim["rho_rzt"]   = sim["rho_rzt"][::coarsen_z]
        if sim["rho_s_rzt"] is not None:
            sim["rho_s_rzt"] = sim["rho_s_rzt"][::coarsen_z]
        sim["I_rzt"]      = sim["I_rzt"][::coarsen_z]

    NA_eff = calc_NA(lmd_nm); lmd_um = lmd_nm * 1e-3
    z_focus = -float(sim["params"].get("begin_um", 0.0))

    # t=0 = coincidence pompe/sonde AU POINT DE COLLAPSE, pas a l'entree de la
    # boite (voir auto_t0_ref_um) -- calcule une fois, reutilise pour tous
    # les pulses.
    t0_ref_um = auto_t0_ref_um(sim)
    t0_ref_lab_um = t0_ref_um + z_focus

    r_dens_um = sim["rlist_um"][::coarsen_r]

    pulses = []
    z_sim_ref = x_sim_ref = None
    for p in range(pmin, pmax + 1):
        t_exp = p * fs_per_pulse
        z_lab, x_um, phases = channel_phases_2d(
            sim, t_exp,
            apply_na_filter=apply_na_filter, NA_eff=NA_eff, lmd_um=lmd_um,
            probe_lmd_nm=lmd_nm, t0_ref_um=t0_ref_um)
        if z_sim_ref is None:
            z_sim_ref = z_lab; x_sim_ref = x_um
        rho_e_col = phases.pop("rho_e_col", None)
        rho_s_col = phases.pop("rho_s_col", None)
        # Instantane de densite (z, r) A CE DELAI -- pas un historique complet
        # comme avant : reagit au curseur exactement comme le panneau de phase.
        _, _, rho_e_log, rho_s_log = density_maps_2d(sim, t_exp, t0_ref_um=t0_ref_um)
        pulses.append(dict(
            p=int(p), t_exp=float(t_exp),
            channels={k: (_to_json_array(v) if v is not None else None) for k, v in phases.items()},
            rho_e_map=(_to_json_array(rho_e_log[:, ::coarsen_r], 3) if rho_e_log is not None else None),
            rho_s_map=(_to_json_array(rho_s_log[:, ::coarsen_r], 3) if rho_s_log is not None else None),
            rho_e_col=(_to_json_array(rho_e_col, 3) if rho_e_col is not None else None),
            rho_s_col=(_to_json_array(rho_s_col, 3) if rho_s_col is not None else None),
        ))

    return dict(z_sim=_to_json_array(z_sim_ref, 3), x_sim=_to_json_array(x_sim_ref, 3),
                r_dens=_to_json_array(r_dens_um, 3),
                pulses=pulses, z_focus=z_focus, probe_nm=float(lmd_nm),
                t0_ref_lab_um=t0_ref_lab_um)

def build_explorer_html(sim_dirs, save="abel_phase_explorer.html", *,
                         raw_dir=None, energy_uJ=4.0,
                         pmin=-40, pmax=40,
                         fs_per_pulse=67.0, lmd_nm=None,
                         apply_na_filter=True,
                         phase_clip=0.2, t_min=0.75, xlim=None, ylim=(-50.0, 50.0),
                         coarsen_z=1, coarsen_r=1, rho_log_min=12.0, rho_log_max=21.0,
                         rho_col_log_min=14.0, rho_col_log_max=23.0):
    """
    sim_dirs : dict {scenario_name: path_to_dir_containing_result.npz+params.json}
               (exactly what the ablation loop in the notebook produces).
    raw_dir  : optional path to experimental npz shots (unified_filament_slider_v3
               naming convention). Left as None => sim-only page.
    """
    scenarios = {}
    z_focus_by_scenario = {}
    probe_nm_by_scenario = {}
    t0_ref_by_scenario = {}

    for name, sim_dir in sim_dirs.items():
        try:
            scenarios[name] = run_slider_scenario(
                sim_dir, pmin, pmax, fs_per_pulse, lmd_nm, apply_na_filter,
                coarsen_z=coarsen_z, coarsen_r=coarsen_r)
            z_focus_by_scenario[name] = scenarios[name].pop("z_focus")
            probe_nm_by_scenario[name] = scenarios[name].pop("probe_nm")
            t0_ref_by_scenario[name] = scenarios[name].pop("t0_ref_lab_um")
            print(f"[{name}] {len(scenarios[name]['pulses'])} pulses depuis {sim_dir} "
                  f"(t=0 au collapse, z={t0_ref_by_scenario[name]:.0f} µm lab)")
        except Exception as e:
            print(f"[{name}] indisponible ({e})")

    if not scenarios:
        raise RuntimeError("Aucun scénario chargé -- vérifie sim_dirs (result.npz + params.json).")

    has_exp = False
    if raw_dir is not None and rotate is not None:
        exp_lmd_nm = float(lmd_nm) if lmd_nm is not None else float(next(iter(probe_nm_by_scenario.values())))
        test_file = Path(raw_dir) / raw_filename(energy_uJ, 0, exp_lmd_nm)
        has_exp = test_file.exists()
        if not has_exp:
            print(f"[exp] pas de fichiers trouvés dans {raw_dir} -- panneau expérience désactivé")

    xlim_max = max(float(s["z_sim"][-1]) for s in scenarios.values()) + 20
    xlim_eff = list(xlim) if xlim is not None else [-50.0, xlim_max]

    # Axe r pour le panneau densites : instantane (z, r) a CHAQUE delai (pas
    # un historique (z, t) fixe) -- meme principe que le panneau de phase.
    r_max_dens = 0.0
    has_density = False
    for sc in scenarios.values():
        if sc.get("r_dens") and any(p.get("rho_e_map") is not None for p in sc["pulses"]):
            r_max_dens = max(r_max_dens, float(sc["r_dens"][-1]))
            has_density = True

    density_layout = build_density_layout(xlim_eff, [0.0, r_max_dens]) if has_density else {}

    data_obj = dict(scenarios=scenarios)
    has_T = any(p["channels"].get("transmittance") is not None
                for sc in scenarios.values() for p in sc["pulses"])
    meta = dict(clip=float(phase_clip), z_focus=z_focus_by_scenario,
                probe_nm=probe_nm_by_scenario,
                t0_ref=t0_ref_by_scenario,
                tmin=float(t_min),
                rho_log_min=float(rho_log_min), rho_log_max=float(rho_log_max),
                rho_col_log_min=float(rho_col_log_min), rho_col_log_max=float(rho_col_log_max))
    layout = build_layout(xlim_eff, ylim, with_transmittance=has_T)

    first = next(iter(scenarios.values()))
    p0 = first["pulses"][0]
    init_label = f"pulse {'+' if p0['p'] >= 0 else ''}{p0['p']}  (Δt = {p0['t_exp']:.0f} fs)"

    html = (HTML_TEMPLATE
        .replace("__FIGURE_HEIGHT__", "700")
        .replace("__PULSE_MAX__", str(len(first["pulses"]) - 1))
        .replace("__INIT_LABEL__", init_label)
        .replace("__DATA_JSON__", json.dumps(data_obj))
        .replace("__LAYOUT_JSON__", json.dumps(layout))
        .replace("__META_JSON__", json.dumps(meta))
        .replace("__HAS_EXP__", "true" if has_exp else "false")
        .replace("__DENSITY_LAYOUT_JSON__", json.dumps(density_layout))
        .replace("__HAS_DENSITY__", "true" if has_density else "false"))

    with open(save, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"-> Sauvegardé : {save}  ({len(html)/1e6:.1f} MB)")
    return save


if __name__ == "__main__":
    print("Import ce module et appelle build_explorer_html(sim_dirs=...).")
