"""
unified_filament_slider_v3.py
=============================
Slider interactif comparant l'expérience (PREPROCESS STRICTEMENT IDENTIQUE
au pipeline 01_preprocess_npz_v23 sur la vue side, SANS crop) à la simulation
`nlprop_filament` (Abel forward seulement, sans gradient / sans shear Wollaston).

v3 : ajout du canal THERMIQUE (chaleur déposée au piégeage STE) comme 4e
contribution à Δn, à côté de Drude (rho_e), Lorentz STE (rho_s) et Kerr (I).
Blocs ajoutés grepables via [thermal].

CONVENTION DE COORDONNÉES
-------------------------
z = 0 dans le repère labo  <->  air-glass interface (point où le laser
                                entre dans le SiO2).
La position de l'interface dans l'image expérimentale (en µm dans le
système de coordonnées de l'image rotée + baseline-corrigée) est éditable
DIRECTEMENT depuis la page HTML via deux champs input :
    - Interface z (le long de la propagation)
    - Axe laser y (transverse, pour centrer l'axe à y = 0)

Géométrie du focus :
    z_focus_glass = N_GLASS * Z_FOCUS_AIR_DIST_UM
où Z_FOCUS_AIR_DIST_UM est la distance interface->foyer si la lentille
focalisait dans le vide (pas de réfraction). C'est le foyer GAUSSIEN, donc
pulse 0 expé <-> pump à z = z_focus_glass <-> t_sim = 0.
"""

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import rotate


# =============================================================================
# == CONFIG ====================================================================
# =============================================================================

# --- Fichiers ---
RAW_DIR        = r'C:/Users/anato/Desktop/20260618_inside300um'
SIM_DIR        = r'C:/Users/anato/Desktop/24juin/export/sim_filament_13uJ_w4'
RESULT_NPZ     = str(Path(SIM_DIR) / "result.npz")
PARAMS_JSON    = str(Path(SIM_DIR) / "params.json")
OUTPUT_HTML    = "unified_filament_slider_v3.html"

# --- Conditions ---
ENERGY_UJ      = 13.0
LMD_NM         = 490.0
FS_PER_PULSE   = 67.0
PMIN, PMAX     = -20, 19

# --- Preprocess (IDENTIQUE 01_preprocess_npz_v23.py, side phase seulement) ---
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

# --- Géométrie : z = 0 à l'interface ---
# Z_FOCUS_AIR_DIST_UM = distance interface -> foyer GAUSSIEN dans le vide
# (avant réfraction par l'interface)
Z_FOCUS_AIR_DIST_UM = 272
N_GLASS             = 1.45
Z_FOCUS_GLASS_DIST_UM = N_GLASS * Z_FOCUS_AIR_DIST_UM      # ≈ 488.65 µm

# --- Pump group velocity (1030 nm dans SiO2) ---
N_GROUP_PUMP    = 1.4628
V_G_PUMP_UM_FS  = 0.2998 / N_GROUP_PUMP                    # ≈ 0.2049 µm/fs
# Niveau STE (Lorentz). Probe à 490 nm = 2.53 eV.
#   STE_LEVEL_EV = 5.8   -> PCM E' center, facteur ~ 0.235  (STE faible et positif)
#   STE_LEVEL_EV = 2.74  -> FUM phosphorescence, facteur ~ 5.77 (STE très fort, attention)
# Doit rester > E_probe pour avoir le signe positif attendu.
STE_LEVEL_EV = 5.8
# --- Sim ---
INCLUDE_KERR              = True
INCLUDE_STE               = True
X_SIM_HALF_UM             = 50.0
DX_SIM_UM                 = 0.1
APPLY_NA_FILTER           = True
MASK_SIM_BEFORE_INTERFACE = True   # masque z_sim_lab < 0 (avant l'interface)

# --- [thermal] Canal thermique : chaleur déposée au piégeage STE -------------
# Post-traitement pur : dn_th est esclave de rho_s_rzt, rien dans le solveur.
#   dn_eq(t) = (dn/dT) * E_dep * q_e * rho_s(t) / (rho*Cp)      [isochore]
#   d(dn_th)/dt = (dn_eq - dn_th) / tau_ph                       [montée phonons]
# E_DEP_EV = énergie relâchée en phonons PAR porteur piégé, encadrée par :
#   borne basse : (U_g - STE_LEVEL_EV) + eps_k ~ 5.2 eV
#                 (énergie stockée dans le STE = transition optique 5.8 eV)
#   borne haute : U_s + eps_k ~ 8 eV
#                 (énergie stockée = niveau d'émission 2.74 eV ; le shift de
#                  Stokes 5.8 - 2.74 part alors en phonons dès le piégeage)
# La vérité est entre les deux (relaxation Franck-Condon). None -> borne basse
# auto : (U_g[params] - STE_LEVEL_EV) + E_KIN_MEAN_EV.
INCLUDE_THERMAL = True
DN_DT_K         = 1.1e-5     # dn/dT silice [K^-1], quasi-achromatique
RHO_CP_J_CM3_K  = 1.628      # rho*Cp = 2200 kg/m^3 x 740 J/(kg K)  [J cm^-3 K^-1]
E_KIN_MEAN_EV   = 2.0        # <eps_k> du porteur au moment du piégeage
E_DEP_EV        = None       # None -> (U_g - STE_LEVEL_EV) + E_KIN_MEAN_EV
TAU_PH_FS       = 1000.0     # thermalisation phonons (constante de montée)

# --- Position initiale de l'interface (l'utilisateur ajuste dans l'HTML) ---
# En µm dans le système de coord. de l'image rotée + baseline-corrigée
# (origine = pixel (0, 0) de cette image, axes alignés avec ses lignes/colonnes).
INIT_IFX_UM = 178.0
INIT_IFY_UM = 497.0

# --- Affichage ---
PHASE_CLIP          = 0.2
PHASE_SIGN          = +1.0
XLIM_PLOT           = (-50.0, 800.0)
YLIM_EXP            = (-50.0, 50.0)
YLIM_SIM            = (-50.0, 50.0)
FIGURE_HEIGHT       = 950
COARSEN_EXP         = 1
COARSEN_SIM         = 1
LASER_AXIS_BAND_UM  = 1.0


# =============================================================================
# == PREPROCESS (extraits 1:1 de 01_preprocess_npz_v23.py, side phase) =========
# =============================================================================

def fmt_delay(p):
    x = float(p); r = round(x)
    return f"{int(r):+d}" if np.isclose(x, r) else f"{x:+g}"


def fmt_probe(lmd):
    return f"{float(lmd):g}"


def raw_filename(energy_uJ, delay_pulse, lmd_nm):
    return f"{float(energy_uJ):.1f}uJ_{fmt_delay(delay_pulse)}pulse_{fmt_probe(lmd_nm)}nm.npz"


def calc_NA(lmd_nm):
    return float(NA_REF) * float(lmd_nm) / float(NA_REF_LMD_NM)


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


def preprocess_side_full(npz_path, lmd_nm=LMD_NM):
    """
    Pipeline IDENTIQUE à preprocess_npz_v23.process_one_file (vue side
    seulement), sauf qu'on RENVOIE la phase rotée + baseline-corrigée
    SANS APPLIQUER LE CROP final.

    Retour
    ------
    phase : ndarray (H_rot, W_rot)  [rad]
    s_um_per_px : pas physique d'un pixel [µm]
    rec_NA : NA effective de la reconstruction
    """
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
    field_rot = rotate(field, ROTATION_ANGLE_DEG,
                       reshape=True, order=1, cval=0, mode="constant")
    phase = np.angle(field_rot)

    mask  = baseline_rects_to_mask(phase.shape, SIDE_BASELINE_RECTS)
    plane = fit_plane_from_mask(phase, mask)
    phase = phase - plane

    if INVERT_SIDE_PHASE:
        phase = -phase

    s_um_per_px = pxsize_rec_um / M_SIDE
    return phase, s_um_per_px, rec_NA


# =============================================================================
# == SIM : Abel forward seulement (PAS de shear) ==============================
# =============================================================================

def load_sim(result_npz_path=RESULT_NPZ, params_json_path=PARAMS_JSON):
    d = np.load(result_npz_path, allow_pickle=True)
    if "rho_rzt" not in d.files or d["rho_rzt"].shape == ():
        raise RuntimeError(
            "result.npz ne contient pas de rho_rzt utilisable. "
            "Relance la sim avec rho_t_stride > 0.")
    sim = dict(
        rlist_um   = np.asarray(d["rlist"], dtype=np.float64) * 1e6,
        z_sim_um   = np.asarray(d["z"],     dtype=np.float64) * 1e6,
        rho_rzt    = np.asarray(d["rho_rzt"],   dtype=np.float32),
        rho_s_rzt  = (np.asarray(d["rho_s_rzt"], dtype=np.float32)
                      if "rho_s_rzt" in d.files else None),
        I_rzt      = np.asarray(d["I_rzt"],     dtype=np.float32),
        t_sub_fs   = np.asarray(d["t_sub_fs"],  dtype=np.float64),
    )
    with open(params_json_path, "r") as f:
        sim["params"] = json.load(f)
    return sim


# --- [thermal] --------------------------------------------------------------
def precompute_dn_thermal(sim, tau_ph_fs=TAU_PH_FS):
    """
    Précalcule dn_th(r, z, t) sur tout le cube (float32, même forme que
    rho_s_rzt, ~1.2 GB pour 497x3000x200). Appelé UNE fois après load_sim ;
    abel_phase_2d indexe ensuite dn_th_rzt exactement comme rho_rzt.

    Modèle : chaque porteur piégé dépose E_dep en phonons localement.
    Chauffage isochore (l'acoustique part en w/c_s ~ 0.3 ns >> fenêtre) :
        dn_eq(t)   = (dn/dT) * E_dep * q_e * rho_s(t) / (rho*Cp)
        dn_th'(t)  = (dn_eq - dn_th) / tau_ph
    intégré en pas exact-exponentiel (source constante par morceaux).

    NB : l'axe t_sub stocké (repère co-mobile) est le temps physique local à
    z fixé, à un décalage constant z/v_g près -> la récurrence temporelle est
    légitime plan par plan, sans re-mapping.
    """
    rho_s = sim.get('rho_s_rzt')
    if rho_s is None:
        return None
    p = sim['params']
    e_dep_eV = (E_DEP_EV if E_DEP_EV is not None
                else float(p.get('U_g_eV', 9.0)) - STE_LEVEL_EV + E_KIN_MEAN_EV)
    fac = np.float32(DN_DT_K * e_dep_eV * 1.602176634e-19
                     / RHO_CP_J_CM3_K)                     # [cm^3] ; dn = fac*rho_s
    t = sim['t_sub_fs']
    a = np.float32(np.exp(-float(t[1] - t[0]) / float(tau_ph_fs)))

    dn = np.zeros_like(rho_s)                              # float32
    for j in range(1, rho_s.shape[-1]):
        eq = fac * rho_s[..., j]
        dn[..., j] = eq + (dn[..., j - 1] - eq) * a

    sim['dn_th_rzt'] = dn
    sim['thermal_meta'] = dict(E_dep_eV=float(e_dep_eV),
                               fac_cm3=float(fac),
                               tau_ph_fs=float(tau_ph_fs))
    return dn


def build_abel_matrix(rlist_um, x_um):
    """
    Matrice Abel forward shell-based sur grille r non-uniforme :
        phi(x_i) = sum_j A[i, j] * f_j
                 ~= 2 * int_|x_i|^inf f(r) * r / sqrt(r^2 - x^2) dr.
    """
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
    A[valid] = 2.0 * (np.sqrt(np.maximum(upper[valid], 0.0))
                      - np.sqrt(np.maximum(lower[valid], 0.0)))
    return A


def abel_phase_2d(sim, t_exp_fs,
                  include_kerr=INCLUDE_KERR,
                  include_ste=INCLUDE_STE,
                  include_thermal=INCLUDE_THERMAL,
                  mask_before_interface=MASK_SIM_BEFORE_INTERFACE):
    """
    Phase intégrée ligne-de-visée phi(x, z_lab) au temps expé t_exp_fs.
    PAS de différentiel Nomarski : juste l'Abel forward de Δn.

    Δn = Drude (rho_e, <0) + Lorentz STE (rho_s, >0 sous 5.8 eV)
       + Kerr (n2*I, >0) + [thermal] thermique (dn_th_rzt, >0).

    Convention :
        z_lab = 0 à l'interface ;
        gaussian focus à z_lab = Z_FOCUS_GLASS_DIST_UM ;
        z_sim_lab = z_sim_um + Z_FOCUS_GLASS_DIST_UM ;
        t_local(z_sim) = t_exp - z_sim_um / V_G_PUMP_UM_FS.

    """

    rlist_um = sim['rlist_um']
    z_sim_um = sim['z_sim_um']
    t_sub_fs = sim['t_sub_fs']
    params   = sim['params']

    n0_probe     = float(params.get('n0_probe', 1.46))
    nc_probe_cm3 = float(params['nc_probe_cm3'])
    lmd_probe_nm = float(params.get('lambda_probe_nm', 490.0))
    n2_m2W       = float(params.get('n2', 3.54e-20))

    # Temps local au z_sim (nearest neighbour dans t_sub_fs)
    t_local = t_exp_fs - z_sim_um / V_G_PUMP_UM_FS
    k_t = np.clip(np.searchsorted(t_sub_fs, t_local), 1, len(t_sub_fs) - 1)
    left  = np.abs(t_sub_fs[k_t - 1] - t_local)
    right = np.abs(t_sub_fs[k_t]     - t_local)
    k_t = np.where(left < right, k_t - 1, k_t)



    Nz = len(z_sim_um); Nr = len(rlist_um)
    iz = np.arange(Nz)

    # Populations séparées au temps local de chaque z
    rho_e_rz = sim['rho_rzt'][iz, :, k_t].astype(np.float64)
    rho_s_rz = (sim['rho_s_rzt'][iz, :, k_t].astype(np.float64)
                if (include_ste and sim['rho_s_rzt'] is not None)
                else np.zeros_like(rho_e_rz))


    E_probe = 1240.0 / lmd_probe_nm
    I_rz = sim['I_rzt'][iz, :, k_t].astype(np.float64)

    delta_n = - rho_e_rz / (2.0 * n0_probe * nc_probe_cm3)
    if include_ste and sim['rho_s_rzt'] is not None:
        f_STE = E_probe**2 / (STE_LEVEL_EV**2 - E_probe**2)
        delta_n = delta_n + f_STE * rho_s_rz / (2.0 * n0_probe * nc_probe_cm3)
    if include_kerr:
        delta_n = delta_n + n2_m2W * I_rz * 1.0e4
    # [thermal] 4e canal : même indexation (iz, :, k_t) que les autres cubes
    if include_thermal and sim.get('dn_th_rzt') is not None:
        delta_n = delta_n + sim['dn_th_rzt'][iz, :, k_t].astype(np.float64)


    # Grille x sim
    x_max = float(min(X_SIM_HALF_UM, rlist_um[-1]))
    n_x   = int(2 * x_max / DX_SIM_UM) + 1
    x_um  = np.linspace(-x_max, x_max, n_x)

    A = build_abel_matrix(rlist_um, x_um)

    lmd_p_um = lmd_probe_nm * 1e-3
    phi_xz = (2.0 * np.pi / lmd_p_um) * (delta_n @ A.T)        # (Nz, Nx) rad

    # Mapping z_sim -> z_lab (z = 0 à l'interface)
    z_lab_um = z_sim_um + Z_FOCUS_GLASS_DIST_UM
    if mask_before_interface:
        air = z_lab_um < 0.0
        phi_xz[air, :] = np.nan

    return z_lab_um, x_um, phi_xz


# =============================================================================
# == Helpers communs ===========================================================
# =============================================================================

def lowpass_NA_2d(phi, d_axis0_um, d_axis1_um, NA, lmd_um):
    phi = np.asarray(phi, dtype=float)
    nan_mask = ~np.isfinite(phi)
    fill = np.where(nan_mask, 0.0, phi)
    n0, n1 = fill.shape
    k0 = 2 * np.pi * np.fft.fftfreq(n0, d=d_axis0_um)
    k1 = 2 * np.pi * np.fft.fftfreq(n1, d=d_axis1_um)
    K0, K1 = np.meshgrid(k0, k1, indexing='ij')
    k_max = 2 * np.pi * float(NA) / float(lmd_um)
    mask = (K0**2 + K1**2) <= k_max**2
    out = np.real(np.fft.ifft2(np.fft.fft2(fill) * mask))
    out[nan_mask] = np.nan
    return out


def _to_json_array(a, decimals=4):
    """Sérialise un ndarray en liste compacte : floats arrondis, NaN -> None."""
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
<title>Unified filament slider</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  body { font-family: -apple-system, Segoe UI, sans-serif; margin: 16px;
         color: #222; }
  .controls { padding: 10px 14px; background: #f4f4f7; border-radius: 6px;
              margin-bottom: 10px; display: flex; flex-wrap: wrap;
              gap: 18px; align-items: center; }
  .controls label { font-size: 14px; }
  .controls input[type=number] { width: 80px; margin-left: 6px; }
  .controls input[type=range]  { width: 320px; vertical-align: middle; }
  #status { font-size: 13px; color: #555; margin-left: auto; }
  #plot { width: 100%; height: __FIGURE_HEIGHT__px; }
</style>
</head>
<body>

<h3 style="margin:6px 0 12px 0">Unified filament slider — z = 0 à l'interface</h3>

<div class="controls">
  <label>z-interface [µm in the image]:
    <input type="number" id="ifX" value="__INIT_IFX__" step="1"></label>
  <label>Laser axis y [µm in the image]: :
    <input type="number" id="ifY" value="__INIT_IFY__" step="1"></label>
  <label>Band y avg. [µm] :
    <input type="number" id="bandY" value="__INIT_BAND__" step="0.5" min="0.5"></label>
  <label>Pulse :
    <input type="range" id="pulseSlider" min="0" max="__PULSE_MAX__" value="0">
    <span id="pulseLabel" style="display:inline-block; width:180px;">__INIT_LABEL__</span>
  </label>
  <span id="status"></span>
</div>

<div id="plot"></div>

<script>
const DATA   = __DATA_JSON__;
const LAYOUT = __LAYOUT_JSON__;
const META   = __META_JSON__;

let pulseIdx = 0;
let ifX   = +document.getElementById('ifX').value;
let ifY   = +document.getElementById('ifY').value;
let bandY = +document.getElementById('bandY').value;

// -------- helpers -----------------------------------------------------------

// x_lab = x_local - shift
function shiftArr(arr, delta) {
  const out = new Array(arr.length);
  for (let i = 0; i < arr.length; i++) out[i] = arr[i] - delta;
  return out;
}

// masque toutes les colonnes de phi2d où xCoord[j] < threshold (-> null)
// phi2d shape [Ny][Nx], xCoord shape [Nx]
function maskBefore(phi2d, xCoord, threshold) {
  return phi2d.map(row => row.map((v, j) => xCoord[j] < threshold ? null : v));
}

// moyenne d'une image [N1][N2] le long du 1er axe, sur les lignes i telles
// que |axisCoords[i]| < bandUm. Renvoie un vecteur de longueur N2.
function bandMean(phi2d, axisCoords, bandUm) {
  const N1 = phi2d.length;
  if (N1 === 0) return [];
  const N2 = phi2d[0].length;
  const line = new Array(N2).fill(0);
  const cnt  = new Array(N2).fill(0);
  for (let i = 0; i < N1; i++) {
    if (Math.abs(axisCoords[i]) < bandUm) {
      const row = phi2d[i];
      for (let j = 0; j < N2; j++) {
        const v = row[j];
        if (v !== null && isFinite(v)) { line[j] += v; cnt[j]++; }
      }
    }
  }
  for (let j = 0; j < N2; j++) line[j] = cnt[j] > 0 ? line[j] / cnt[j] : null;
  return line;
}

// -------- traces builder ----------------------------------------------------

function buildTraces(p) {
  // Axes labo (dépendent des inputs)
  const xLab = shiftArr(DATA.x_local, ifX);
  const yLab = shiftArr(DATA.y_local, ifY);

  // Masque tout ce qui est avant l'interface côté expé (x_lab < 0)
  const phiExpMasked = maskBefore(p.phi_exp, xLab, 0.0);

  // Lineouts : moyenne sur la bande |y| < bandY (expé) et |x| < bandY (sim)
  const lineExp = bandMean(phiExpMasked, yLab, bandY);
  const lineSim = bandMean(p.phi_sim_2d, DATA.x_sim, bandY);

  return [
    // rangée 1 : heatmap expé masquée
    { type: 'heatmap', x: xLab, y: yLab, z: phiExpMasked,
      colorscale: 'RdBu', reversescale: false, zmid: 0,
      zmin: -META.clip, zmax: META.clip,
      colorbar: { title: 'δφ Exp (rad)', len: 0.22, y: 0.88, yanchor: 'middle' },
      xaxis: 'x', yaxis: 'y',
      hovertemplate: 'Exp: z=%{x:.0f} µm<br>y=%{y:.0f} µm<br>δφ=%{z:.3f} rad<extra></extra>' },

    // rangée 2 : heatmap sim (déjà masquée en Python via MASK_SIM_BEFORE_INTERFACE)
    { type: 'heatmap', x: DATA.z_sim, y: DATA.x_sim, z: p.phi_sim_2d,
      colorscale: 'RdBu', reversescale: false, zmid: 0,
      zmin: -META.clip, zmax: META.clip,
      colorbar: { title: 'δφ Sim (rad)', len: 0.22, y: 0.62, yanchor: 'middle' },
      xaxis: 'x2', yaxis: 'y2',
      hovertemplate: 'Sim: z=%{x:.0f} µm<br>x=%{y:.0f} µm<br>δφ=%{z:.3f} rad<extra></extra>' },

    // rangée 3 : lineout expé
    { type: 'scatter', x: xLab, y: lineExp, mode: 'lines',
      line: { color: 'crimson', width: 1.4 }, name: 'Exp',
      xaxis: 'x3', yaxis: 'y3', showlegend: false,
      hovertemplate: 'Exp: z=%{x:.0f} µm<br>δφ=%{y:.3f} rad<extra></extra>' },

    // rangée 4 : lineout sim
    { type: 'scatter', x: DATA.z_sim, y: lineSim, mode: 'lines',
      line: { color: '#1f77b4', width: 2.2 }, name: 'Sim',
      xaxis: 'x4', yaxis: 'y4', showlegend: false,
      hovertemplate: 'Sim: z=%{x:.0f} µm<br>δφ=%{y:.3f} rad<extra></extra>' }
  ];
}

// -------- render ------------------------------------------------------------

function render() {
  const p = DATA.pulses[pulseIdx];
  Plotly.react('plot', buildTraces(p), LAYOUT, {responsive: true});

  const sign = p.p > 0 ? '+' : '';
  document.getElementById('pulseLabel').textContent =
    `pulse ${sign}${p.p}  (Δt = ${p.t_exp.toFixed(0)} fs)`;
  document.getElementById('status').textContent =
    `rec_NA = ${p.rec_NA.toFixed(3)} | z_foyer_gauss = ${META.z_focus.toFixed(0)} µm | bande y = ±${bandY} µm`;
}

// -------- listeners ---------------------------------------------------------

document.getElementById('ifX').addEventListener('input', e => {
  ifX = +e.target.value; render();
});
document.getElementById('ifY').addEventListener('input', e => {
  ifY = +e.target.value; render();
});
document.getElementById('bandY').addEventListener('input', e => {
  bandY = +e.target.value; render();
});
document.getElementById('pulseSlider').addEventListener('input', e => {
  pulseIdx = +e.target.value; render();
});

// -------- init --------------------------------------------------------------

Plotly.newPlot('plot', buildTraces(DATA.pulses[0]), LAYOUT, {responsive: true});
render();
</script>

</body>
</html>
"""


def build_layout(xlim=None):
    """Construit le dict de layout plotly (axes partagés en x, 4 rangées)."""
    xlim = list(xlim) if xlim is not None else list(XLIM_PLOT)
    return {
        "template": "plotly_white",
        "margin": {"l": 70, "r": 30, "t": 20, "b": 50},
        "height": FIGURE_HEIGHT,

        "xaxis":  {"domain": [0.0, 1.0], "anchor": "y",
                   "range": xlim, "matches": "x4"},
        "yaxis":  {"domain": [0.72, 1.0], "anchor": "x",
                   "range": list(YLIM_EXP), "title": "y (µm)"},

        "xaxis2": {"domain": [0.0, 1.0], "anchor": "y2",
                   "range": xlim, "matches": "x4"},
        "yaxis2": {"domain": [0.46, 0.70], "anchor": "x2",
           "range": list(YLIM_SIM), "title": "x sim (µm)"},   # <-- scaleanchor retiré


        "xaxis3": {"domain": [0.0, 1.0], "anchor": "y3",
                   "range": xlim, "matches": "x4"},
        "yaxis3": {"domain": [0.23, 0.44], "anchor": "x3",
                   "title": "δφ Exp on-axis (rad)"},

        "xaxis4": {"domain": [0.0, 1.0], "anchor": "y4",
                   "range": xlim,
                   "title": "Propagation z (µm) — lab frame (0 = interface)"},
        "yaxis4": {"domain": [0.0, 0.21], "anchor": "x4",
                   "title": "δφ Sim on-axis (rad)"},

        "shapes": [
            {"type": "line", "x0": 0, "x1": 0,
             "y0": 0, "y1": 1, "yref": "paper",
             "line": {"color": "green", "width": 1, "dash": "dash"}},
            {"type": "line",
             "x0": Z_FOCUS_GLASS_DIST_UM, "x1": Z_FOCUS_GLASS_DIST_UM,
             "y0": 0, "y1": 1, "yref": "paper",
             "line": {"color": "purple", "width": 1, "dash": "dot"}},
        ],
        "annotations": [
            {"x": 0, "y": 1.005, "xref": "x", "yref": "paper",
             "text": "interface", "showarrow": False,
             "font": {"color": "green", "size": 11}},
            {"x": Z_FOCUS_GLASS_DIST_UM, "y": 1.005, "xref": "x", "yref": "paper",
             "text": "focus gauss.", "showarrow": False,
             "font": {"color": "purple", "size": 11}},
        ],
    }


# =============================================================================
# == MAIN ======================================================================
# =============================================================================

def run_slider(energy_uJ=ENERGY_UJ, pmin=PMIN, pmax=PMAX,
               fs_per_pulse=FS_PER_PULSE, lmd_nm=LMD_NM,
               raw_dir=RAW_DIR, sim_dir=SIM_DIR,
               coarsen_exp=COARSEN_EXP, coarsen_sim=COARSEN_SIM,
               save=OUTPUT_HTML):

    NA_eff = calc_NA(lmd_nm); lmd_um = lmd_nm * 1e-3

    print(f"z = 0 à l'interface ; z_focus_gauss = {Z_FOCUS_GLASS_DIST_UM:.1f} µm")
    print(f"NA effective sonde   : {NA_eff:.3f} à {lmd_nm:g} nm")
    print(f"V_g pump 1030 nm     : {V_G_PUMP_UM_FS:.4f} µm/fs (n_g = {N_GROUP_PUMP:.4f})")

    # --- Sim ---
    try:
        sim = load_sim(str(Path(sim_dir) / "result.npz"),
                       str(Path(sim_dir) / "params.json"))
        print(f"Sim chargée: rho_rzt {sim['rho_rzt'].shape}, "
              f"t_sub_fs ∈ [{sim['t_sub_fs'].min():+.0f}, {sim['t_sub_fs'].max():+.0f}] fs")
    except Exception as e:
        raise RuntimeError(f"Sim indisponible : {e}")

    # --- [thermal] précalcul du cube dn_th (une fois, ~1.2 GB float32) ---
    if INCLUDE_THERMAL:
        if precompute_dn_thermal(sim) is None:
            print("Canal thermique      : DÉSACTIVÉ (pas de rho_s_rzt dans le npz)")
        else:
            tm = sim['thermal_meta']
            dn_drude_1e20 = 1e20 / (2.0 * float(sim['params']['n0_probe'])
                                    * float(sim['params']['nc_probe_cm3']))
            dn_th_1e20 = tm['fac_cm3'] * 1e20
            print(f"Canal thermique      : E_dep = {tm['E_dep_eV']:.2f} eV/porteur, "
                  f"tau_ph = {tm['tau_ph_fs']:.0f} fs")
            print(f"                       dn_th(rho_s=1e20) = {dn_th_1e20:+.2e} "
                  f"= {100.0 * dn_th_1e20 / dn_drude_1e20:.1f}% du |Drude|, signe opposé")

    # Auto-ajuste XLIM au maximum de la sim (avec 20 µm de marge)
    xlim_max = float(sim['z_sim_um'][-1] + Z_FOCUS_GLASS_DIST_UM + 20)
    xlim_effective = (XLIM_PLOT[0], xlim_max)
    print(f"XLIM auto        : ({xlim_effective[0]:.0f}, {xlim_effective[1]:.0f}) µm  "
          f"(config demandait max {XLIM_PLOT[1]:.0f})")
    # --- Référence axes locales (mesurées sur le 1er tir trouvé) ---
    x_local = y_local = None
    z_sim_lab_ref = x_sim_ref = None

    # --- Loop pulses ---
    pulses = []
    se = max(1, int(coarsen_exp))
    ss = max(1, int(coarsen_sim))

    for p in range(pmin, pmax + 1):
        npz = Path(raw_dir) / raw_filename(energy_uJ, p, lmd_nm)
        if not npz.exists():
            continue
        try:
            phi_full, s_exp, rec_NA = preprocess_side_full(str(npz), lmd_nm=lmd_nm)
        except Exception as e:
            print(f"  {p:+3d} : preprocess erreur ({e})"); continue

        # axes locaux dans l'image rotée (origine = pixel (0, 0))
        H, W = phi_full.shape
        xl = (np.arange(W) * s_exp).astype(np.float32)
        yl = (np.arange(H) * s_exp).astype(np.float32)
        if x_local is None:
            x_local, y_local = xl, yl
        elif xl.shape != x_local.shape or yl.shape != y_local.shape:
            # taille différente : on pad/crop pour aligner avec le premier
            phi_pad = np.full((y_local.size, x_local.size), np.nan, dtype=np.float32)
            hh = min(phi_pad.shape[0], phi_full.shape[0])
            ww = min(phi_pad.shape[1], phi_full.shape[1])
            phi_pad[:hh, :ww] = phi_full[:hh, :ww]
            phi_full = phi_pad

        # downsample exp
        phi_exp_ds = (PHASE_SIGN * phi_full[::se, ::se]).astype(np.float32)

        # --- Sim Nomarski (Abel forward only) à t_exp = p * fs_per_pulse ---
        t_exp = p * fs_per_pulse
        z_lab_sim, x_sim, phi_LOS = abel_phase_2d(sim, t_exp)
        if APPLY_NA_FILTER and phi_LOS.size > 0:
            dz_sim = float(np.mean(np.diff(z_lab_sim)))
            dx_sim = float(np.mean(np.diff(x_sim)))
            phi_LOS = lowpass_NA_2d(phi_LOS, dz_sim, dx_sim, NA_eff, lmd_um)


        # downsample sim 2D
        phi_sim_2d_ds = phi_LOS[::ss, ::ss].T.astype(np.float32)   # (Nx, Nz) pour heatmap

        if z_sim_lab_ref is None:
            z_sim_lab_ref = z_lab_sim[::ss].astype(np.float32)
            x_sim_ref     = x_sim[::ss].astype(np.float32)

        pulses.append(dict(
            p          = int(p),
            t_exp      = float(t_exp),
            rec_NA     = float(rec_NA),
            phi_exp    = _to_json_array(phi_exp_ds, decimals=4),
            phi_sim_2d = _to_json_array(phi_sim_2d_ds, decimals=4),
        ))
        print(f"  {p:+3d} ({t_exp:+6.0f} fs)  rec_NA={rec_NA:.3f}")

    if not pulses:
        raise FileNotFoundError(f"Aucun shot {energy_uJ:.1f} µJ trouvé dans {raw_dir}")

    # --- Sérialisation ---
    data_obj = dict(
        x_local = _to_json_array(x_local[::se], decimals=3),
        y_local = _to_json_array(y_local[::se], decimals=3),
        z_sim   = _to_json_array(z_sim_lab_ref, decimals=3),
        x_sim   = _to_json_array(x_sim_ref,     decimals=3),
        pulses  = pulses,
    )
    meta = dict(
        clip     = float(PHASE_CLIP),
        band_um  = float(LASER_AXIS_BAND_UM),
        z_focus  = float(Z_FOCUS_GLASS_DIST_UM),
    )
    layout = build_layout(xlim=xlim_effective)

    p0 = pulses[0]
    init_label = f"pulse {'+' if p0['p']>=0 else ''}{p0['p']}  (Δt = {p0['t_exp']:.0f} fs)"

    html = (HTML_TEMPLATE
        .replace("__FIGURE_HEIGHT__", str(FIGURE_HEIGHT))
        .replace("__INIT_IFX__",     str(INIT_IFX_UM))
        .replace("__INIT_IFY__",     str(INIT_IFY_UM))
        .replace("__INIT_BAND__",    str(LASER_AXIS_BAND_UM))    # <-- ajoute
        .replace("__PULSE_MAX__",    str(len(pulses) - 1))
        .replace("__INIT_LABEL__",   init_label)
        .replace("__DATA_JSON__",    json.dumps(data_obj))
        .replace("__LAYOUT_JSON__",  json.dumps(layout))
        .replace("__META_JSON__",    json.dumps(meta)))

    with open(save, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"-> Sauvegardé : {save}")
    print(f"   Taille     : {len(html)/1e6:.1f} MB")


# =============================================================================
if __name__ == "__main__":
    run_slider()
