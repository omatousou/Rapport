"""
web/abel_phase_explorer.py
===========================
Generalisation of `unified_filament_slider_v3.py` (Abel-forward phase slider)
for the term-ablation study:

  - the Delta n(r, z, t) that feeds the Abel transform is split into FOUR
    independently togglable channels -- Drude (free electrons), Lorentz/STE
    (self-trapped excitons), Kerr (n2 I), thermal (phonon heating from STE
    trapping) -- each Abel-transformed and NA-filtered SEPARATELY in Python
    (both operations are linear, so summing channels commutes with both the
    line-of-sight integral and the low-pass filter). The browser only needs
    to sum whichever channels are checked: no FFT / matrix multiply in JS.

  - several simulation SCENARIOS (e.g. the ablation-loop outputs from
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

# --- Canal thermique (chaleur deposee au piegeage STE) ----------------------
DN_DT_K         = 1.1e-5     # dn/dT silice [K^-1]
RHO_CP_J_CM3_K  = 1.628      # rho*Cp = 2200 kg/m^3 x 740 J/(kg K)  [J cm^-3 K^-1]
E_KIN_MEAN_EV   = 2.0
TAU_PH_FS       = 1000.0

# --- Grille x de l'integrale Abel -------------------------------------------
X_SIM_HALF_UM = 50.0
DX_SIM_UM     = 0.2

# --- Pump group index (Sellmeier, silice) -----------------------------------
_SELLMEIER_B  = np.array([0.6961663, 0.4079426, 0.8974794])
_SELLMEIER_L2 = np.array([0.0684043, 0.1162414, 9.896161]) ** 2
_C_UM_FS = 299792458.0 * 1e-9  # um/fs


def _n_sellmeier(lam_um):
    n2m1 = sum(B * lam_um**2 / (lam_um**2 - L2) for B, L2 in zip(_SELLMEIER_B, _SELLMEIER_L2))
    return np.sqrt(1.0 + n2m1)


def group_index(lam_um, d=1e-4):
    n0 = _n_sellmeier(lam_um)
    dn = (_n_sellmeier(lam_um + d) - _n_sellmeier(lam_um - d)) / (2 * d)
    return n0 - lam_um * dn


def calc_NA(lmd_nm):
    return float(NA_REF) * float(lmd_nm) / float(NA_REF_LMD_NM)


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


def precompute_dn_thermal(sim, tau_ph_fs=TAU_PH_FS, e_kin_mean_eV=E_KIN_MEAN_EV):
    rho_s = sim.get("rho_s_rzt")
    if rho_s is None:
        return None
    p = sim["params"]
    e_dep_eV = float(p.get("U_g_eV", 9.0)) - STE_LEVEL_EV + e_kin_mean_eV
    fac = np.float32(DN_DT_K * e_dep_eV * 1.602176634e-19 / RHO_CP_J_CM3_K)
    t = sim["t_sub_fs"]
    a = np.float32(np.exp(-float(t[1] - t[0]) / float(tau_ph_fs)))

    dn = np.zeros_like(rho_s)
    for j in range(1, rho_s.shape[-1]):
        eq = fac * rho_s[..., j]
        dn[..., j] = eq + (dn[..., j - 1] - eq) * a

    sim["dn_th_rzt"] = dn
    return dn


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


def channel_phases_2d(sim, t_exp_fs, *,
                       include_thermal=True,
                       apply_na_filter=True, NA_eff=None, lmd_um=None,
                       mask_before_interface=True,
                       x_sim_half_um=X_SIM_HALF_UM, dx_sim_um=DX_SIM_UM):
    """
    Retourne (z_lab_um, x_um, {"drude":phi, "kerr":phi, "ste":phi, "thermal":phi|None})
    -- une carte de phase (Nz, Nx) deja transformee par Abel et filtree NA,
    PAR CANAL, prete a etre sommee lineairement cote navigateur.
    """
    rlist_um = sim["rlist_um"]; z_sim_um = sim["z_sim_um"]; t_sub_fs = sim["t_sub_fs"]
    params   = sim["params"]

    n0_probe     = float(params.get("n0_probe", 1.46))
    nc_probe_cm3 = float(params["nc_probe_cm3"])
    lmd_probe_nm = float(params.get("lambda_probe_nm", 490.0))
    n2_m2W       = float(params.get("n2", 2.4e-20))
    z_focus_glass_dist_um = -float(params.get("begin_um", 0.0))

    lam_pump_um   = float(params.get("wavelength_nm", 1030.0)) * 1e-3
    n_group_pump  = group_index(lam_pump_um)
    v_g_pump_um_fs = _C_UM_FS / n_group_pump

    t_local = t_exp_fs - z_sim_um / v_g_pump_um_fs
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
        # Martin et al. 1997 Eq. (2) : deux bandes STE avec leurs forces
        # d'oscillateur et leurs largeurs, masse effective 0.5 m_e dans le
        # terme Drude, deplation de la bande de valence, et n = sqrt(eps) au
        # lieu du developpement au premier ordre. Chaque canal est evalue seul
        # (les autres densites mises a zero) pour rester sommable cote
        # navigateur ; c'est exact tant qu'on reste dans le regime lineaire et
        # `overdense` signale ou ce n'est plus le cas.
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
    has_thermal = include_thermal and sim.get("dn_th_rzt") is not None
    dn_channels["thermal"] = (sim["dn_th_rzt"][iz, :, k_t].astype(np.float64)
                              if has_thermal else np.zeros_like(rho_e_rz))

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
        phases[name] = phi if (name != "thermal" or has_thermal) else None

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
  <label><input type="checkbox" id="ch_thermal"> Thermique (chauffage phonons)</label>
</div>

<div id="plot"></div>
<p id="note">Δφ = Abel-forward de Δn, filtré NA de la sonde (pas de recalcul FFT côté navigateur : chaque canal est déjà transformé par Abel + filtré côté Python ; le navigateur ne fait que sommer les canaux cochés).</p>

<script>
const DATA   = __DATA_JSON__;
const LAYOUT = __LAYOUT_JSON__;
const META   = __META_JSON__;
const HAS_EXP = __HAS_EXP__;

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
  if (document.getElementById('ch_thermal').checked && pulse.channels.thermal) chosen.push(pulse.channels.thermal);
  // `transmittance` n'est PAS un canal de Delta n : c'est la partie imaginaire
  // de l'indice, tracee dans son propre panneau. Ne jamais l'ajouter ici.
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
  // phi2d shape (Nz, Nx) -> heatmap wants z=x-axis(z), y-axis(x); transpose for imshow-like orientation
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

function render() {
  const scen = DATA.scenarios[scenarioSel.value];
  const pulse = scen.pulses[Math.min(pulseIdx, scen.pulses.length - 1)];
  Plotly.react('plot', buildTraces(), LAYOUT, {responsive: true});
  const sign = pulse.p > 0 ? '+' : '';
  document.getElementById('pulseLabel').textContent =
    `pulse ${sign}${pulse.p}  (Δt = ${pulse.t_exp.toFixed(0)} fs)`;
  document.getElementById('status').textContent =
    `scénario = ${scenarioSel.value} | z_foyer_gauss = ${META.z_focus[scenarioSel.value].toFixed(0)} µm`;
}

scenarioSel.addEventListener('change', () => {
  pulseIdx = Math.min(pulseIdx, DATA.scenarios[scenarioSel.value].pulses.length - 1);
  document.getElementById('pulseSlider').max = DATA.scenarios[scenarioSel.value].pulses.length - 1;
  render();
});
document.getElementById('pulseSlider').addEventListener('input', e => { pulseIdx = +e.target.value; render(); });
['ch_drude', 'ch_kerr', 'ch_ste', 'ch_thermal'].forEach(id =>
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


def run_slider_scenario(sim_dir, pmin, pmax, fs_per_pulse, lmd_nm, include_thermal, apply_na_filter,
                         coarsen_z=1):
    sim = load_sim(sim_dir)
    if include_thermal:
        precompute_dn_thermal(sim)

    if coarsen_z > 1:
        sim["z_sim_um"]  = sim["z_sim_um"][::coarsen_z]
        sim["rho_rzt"]   = sim["rho_rzt"][::coarsen_z]
        if sim["rho_s_rzt"] is not None:
            sim["rho_s_rzt"] = sim["rho_s_rzt"][::coarsen_z]
        sim["I_rzt"]      = sim["I_rzt"][::coarsen_z]
        if sim.get("dn_th_rzt") is not None:
            sim["dn_th_rzt"] = sim["dn_th_rzt"][::coarsen_z]

    NA_eff = calc_NA(lmd_nm); lmd_um = lmd_nm * 1e-3
    z_focus = -float(sim["params"].get("begin_um", 0.0))

    pulses = []
    z_sim_ref = x_sim_ref = None
    for p in range(pmin, pmax + 1):
        t_exp = p * fs_per_pulse
        z_lab, x_um, phases = channel_phases_2d(
            sim, t_exp, include_thermal=include_thermal,
            apply_na_filter=apply_na_filter, NA_eff=NA_eff, lmd_um=lmd_um)
        if z_sim_ref is None:
            z_sim_ref = z_lab; x_sim_ref = x_um
        pulses.append(dict(
            p=int(p), t_exp=float(t_exp),
            channels={k: (_to_json_array(v) if v is not None else None) for k, v in phases.items()},
        ))
    return dict(z_sim=_to_json_array(z_sim_ref, 3), x_sim=_to_json_array(x_sim_ref, 3),
                pulses=pulses, z_focus=z_focus)


def build_explorer_html(sim_dirs, save="abel_phase_explorer.html", *,
                         raw_dir=None, energy_uJ=13.0, pmin=-20, pmax=19,
                         fs_per_pulse=67.0, lmd_nm=490.0,
                         include_thermal=True, apply_na_filter=True,
                         phase_clip=0.2, t_min=0.75, xlim=None, ylim=(-50.0, 50.0),
                         coarsen_z=1):
    """
    sim_dirs : dict {scenario_name: path_to_dir_containing_result.npz+params.json}
               (exactly what the ablation loop in the notebook produces).
    raw_dir  : optional path to experimental npz shots (unified_filament_slider_v3
               naming convention). Left as None => sim-only page.
    """
    scenarios = {}
    z_focus_by_scenario = {}
    for name, sim_dir in sim_dirs.items():
        try:
            scenarios[name] = run_slider_scenario(
                sim_dir, pmin, pmax, fs_per_pulse, lmd_nm, include_thermal, apply_na_filter,
                coarsen_z=coarsen_z)
            z_focus_by_scenario[name] = scenarios[name].pop("z_focus")
            print(f"[{name}] {len(scenarios[name]['pulses'])} pulses depuis {sim_dir}")
        except Exception as e:
            print(f"[{name}] indisponible ({e})")

    if not scenarios:
        raise RuntimeError("Aucun scénario chargé -- vérifie sim_dirs (result.npz + params.json).")

    has_exp = False
    if raw_dir is not None and rotate is not None:
        test_file = Path(raw_dir) / raw_filename(energy_uJ, 0, lmd_nm)
        has_exp = test_file.exists()
        if not has_exp:
            print(f"[exp] pas de fichiers trouvés dans {raw_dir} -- panneau expérience désactivé")

    xlim_max = max(float(s["z_sim"][-1]) for s in scenarios.values()) + 20
    xlim_eff = list(xlim) if xlim is not None else [-50.0, xlim_max]

    data_obj = dict(scenarios=scenarios)
    has_T = any(p["channels"].get("transmittance") is not None
                for sc in scenarios.values() for p in sc["pulses"])
    meta = dict(clip=float(phase_clip), z_focus=z_focus_by_scenario,
                tmin=float(t_min))
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
        .replace("__HAS_EXP__", "true" if has_exp else "false"))

    with open(save, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"-> Sauvegardé : {save}  ({len(html)/1e6:.1f} MB)")
    return save


if __name__ == "__main__":
    print("Import ce module et appelle build_explorer_html(sim_dirs=...).")
