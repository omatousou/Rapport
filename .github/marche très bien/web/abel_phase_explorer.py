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

  - several simulation scenarios (e.g. the ablation-loop outputs from
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
except ImportError:      # module absent -> fall back to manual calculation
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

# --- STE channel (Lorentz) --------------------------------------------------
# OLD VALUE: a single band at 5.8 eV with an implicit oscillator strength
# of 1. It matched neither the solver (Config.E_tr_eV = 4.2) nor the
# literature. Martin, Guizard, Daguzan, Petite et al., PRB 55, 5799
# (1997), Table II, give for SiO2 TWO STE bands:
#     5.2 eV, f = 0.40, width 1.5 eV
#     4.2 eV, f = 0.15, width 1.0 eV
# i.e. at 515 nm an effective factor of 0.177 where a single band f=1 at 4.2 eV
# gave 0.489: the STE channel was 2.8x too strong.
STE_LEVEL_EV = 4.2          # kept for the historical path only
USE_PERMITTIVITY_MODEL = True   # False -> legacy manual calculation

# --- Abel integral x-grid ---------------------------------------------------
X_SIM_HALF_UM = 50.0
DX_SIM_UM     = 0.2

# --- Pump group index (Sellmeier, silica) ------------------------------------
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
    """n0 and critical density of the probe at `lmd_nm`.

    The HTML can be generated at multiple probe wavelengths from a single
    pump run. In that case these two quantities must be recomputed, rather
    than blindly reading the cached value from params.json.
    """
    lmd_nm = float(lmd_nm)
    lmd_m = lmd_nm * 1e-9
    omega = 2.0 * np.pi * _C_M_S / lmd_m
    n0_probe = float(_n_sellmeier(lmd_nm * 1e-3))
    nc_probe_cm3 = float(_EPS0 * _M_E * omega**2 / _Q_E**2 * 1e-6)
    return n0_probe, nc_probe_cm3


# =============================================================================
# == PREPROCESS (experiment, 1:1 from unified_filament_slider_v3.py) ==========
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
# == SIM: loading + separate Delta n channels + Abel forward ==================
# =============================================================================

def load_sim(sim_dir):
    d = np.load(str(Path(sim_dir) / "result.npz"), allow_pickle=True)
    if "rho_rzt" not in d.files or np.asarray(d["rho_rzt"]).shape == ():
        raise RuntimeError(
            f"{sim_dir}/result.npz has no usable rho_rzt "
            "(re-run filament_sim.run(..., rho_t_stride>0)).")
    # The (z,r,t) cube may be radially sub-sampled (rho_r_stride):
    # in that case the solver writes r_sub, and THIS grid indexes
    # rho_rzt/I_rzt, not rlist.
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
    """z (lab frame, same convention as z_sim_um) where the pump is most
    intense on axis -- the collapse point.

    This is THE physically meaningful reference for t=0: "zero delay =
    coincidence of pump/probe maxima" (as in virtual_experiment.py), not
    the box entry, which has nothing special once the beam is focused
    inside.
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
    Returns (z_lab_um, x_um, {"drude":phi, "kerr":phi, "ste":phi})
    -- a phase map (Nz, Nx) already Abel-transformed and NA-filtered,
    PER CHANNEL, ready to be linearly summed on the browser side.

    `t0_ref_um`: z position (simulation frame, before shift to lab frame)
    where t_exp_fs=0 means "pump and probe coincide". Default (None) is
    the collapse point (see `auto_t0_ref_um`), not the box entry -- the
    probe has no physical reason to coincide with the pump at z=0 rather
    than at the point of maximum intensity.
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
        # Same absorption path as probe_opl_transmittance(material="legacy")
        # in figures_filament.py: without it alpha_cm_rz stayed None and
        # transmittance simply vanished when USE_PERMITTIVITY_MODEL=False.
        from figures_filament import probe_sigma
        tau_c_s = float(params.get("tau_c_s", 1.7e-15))
        meff_drude_rel = float(params.get("meff_drude_rel", 1.0))
        sigma = probe_sigma(lmd_probe_m, tau_c_s, meff_drude_rel)
        alpha_cm_rz = sigma * rho_e_rz

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

    # The experiment measures both the REAL and IMAGINARY parts of the index
    # (phase shift and fringe contrast). Until now this page only computed
    # the former. Transmittance comes from the same eps as phase, so the
    # two can no longer contradict each other.
    if alpha_cm_rz is not None:
        tau = (alpha_cm_rz @ A.T) * 1e-4            # cm^-1 integrated over um
        T = np.exp(-np.clip(tau, 0.0, None))
        if apply_na_filter:
            T = lowpass_NA_2d(T, dz_sim, dx_sim, NA_eff, lmd_um)
        T[air, :] = np.nan
        phases["transmittance"] = T

    return z_lab_um, x_um, phases


# =============================================================================
# == Common helpers ===========================================================
# =============================================================================

def _to_json_array(a, decimals=4):
    a = np.asarray(a)
    if np.issubdtype(a.dtype, np.floating):
        a = np.round(a.astype(np.float64), decimals)
        out = np.where(np.isfinite(a), a, None)
        return out.tolist()
    return a.tolist()


def density_maps_2d(sim, t_exp_fs, t0_ref_um=None, log_floor_cm3=1e12):
    """rho_e(z, r) and rho_STE(z, r) [log10 cm^-3] AT DELAY `t_exp_fs`.

    Same temporal selection as `channel_phases_2d` (t=0 at collapse, not at
    box entry), but WITHOUT Abel transform: this is the real density in the
    meridional plane (z, r) -- a snapshot, not a complete history -- so that
    this panel reacts to the delay cursor exactly like the phase panel,
    instead of showing all times at once with a simple indicator line.
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
        # Clip to floor instead of NaN: a small value keeps a color
        # (the floor color), not a white hole in the map.
        return np.log10(np.clip(a, float(log_floor_cm3), None))

    z_lab_um = z_sim_um + z_focus_glass_dist_um
    return (z_lab_um, r_um,
            _log(sim.get("rho_rzt")), _log(sim.get("rho_s_rzt")))


# =============================================================================
# == HTML output ===============================================================
# =============================================================================

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Abel phase explorer -- term ablation</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  body { font-family: -apple-system, Segoe UI, sans-serif; margin: 16px; color: #222; }
  .controls { padding: 10px 14px; background: #f4f4f7; border-radius: 6px;
              margin-bottom: 10px; display: flex; flex-wrap: wrap;
              gap: 18px; align-items: center; }
  .controls label { font-size: 14px; }
  .controls input[type=range]  { width: 180px; vertical-align: middle; }
  .controls select { padding: 4px; }
  .channels label { margin-right: 10px; }
  #status { font-size: 13px; color: #555; margin-left: auto; }
  #plot { width: 100%; height: __FIGURE_HEIGHT__px; }
  .heading { font-size: 15px; color: #333; margin: 28px 0 4px 0; }
  #densityPlot { width: 100%; height: 600px; }
  #timeSeriesPlot { width: 100%; height: 400px; margin-top: 12px; }
  #note { font-size: 12px; color: #888; margin-top: 16px; }
</style>
</head>
<body>

<h3 style="margin:6px 0 12px 0">Abel phase explorer — togglable &Delta;n channels, ablation scenarios</h3>

<div class="controls">
  <label>Scenario:
    <select id="scenario"></select></label>
  <label>Time from simulation start:
    <input type="range" id="pulseSlider" min="0" max="__PULSE_MAX__" value="0">
    <span id="pulseLabel" style="display:inline-block; width:220px;">__INIT_LABEL__</span>
  </label>
  <span id="status"></span>
</div>
<div class="controls channels">
  <b>&Delta;n channels:</b>
  <label><input type="checkbox" id="ch_drude" checked> Drude (free electrons, &lt;0)</label>
  <label><input type="checkbox" id="ch_kerr" checked> Kerr (n2&middot;I, &gt;0)</label>
  <label><input type="checkbox" id="ch_ste" checked> STE / Lorentz (trapped excitons)</label>
  <b style="margin-left: 20px;">Densities:</b>
  <label><input type="checkbox" id="cb_rho_e" checked> Electrons</label>
  <label><input type="checkbox" id="cb_rho_s" checked> STE</label>
</div>

<div id="plot"></div>

<h4 class="heading" id="densityHeading" style="display:none;">Electron and STE density snapshots (meridional plane)</h4>
<div id="densityPlot"></div>

<h4 class="heading">Phase vs. Time extraction</h4>
<div class="controls">
  <label>Mode:
    <select id="ts_mode">
      <option value="single">Single Point (z, r)</option>
      <option value="compare_z" selected>Compare Z (at selected r)</option>
      <option value="compare_r">Compare R (at selected z)</option>
    </select>
  </label>
  <label>Z index:
    <input type="range" id="ts_z" min="0" max="100" value="7">
    <span id="ts_z_label" style="display:inline-block; width:120px;"></span>
  </label>
  <label>R index:
    <input type="range" id="ts_r" min="0" max="100" value="50">
    <span id="ts_r_label" style="display:inline-block; width:120px;"></span>
  </label>
  <label>Avg N (pixels along r):
    <input type="range" id="ts_avg" min="1" max="21" value="15" step="2">
    <span id="ts_avg_label" style="display:inline-block; width:40px;">15</span>
  </label>
</div>
<div id="timeSeriesPlot"></div>

<p id="note">&delta;&phi; = Abel-forward transform of &Delta;n, filtered by the probe NA (no FFT recomputation in the browser: each channel is already Abel-transformed and NA-filtered on the Python side; the browser only sums the checked channels).</p>

<script>
const DATA   = __DATA_JSON__;
const LAYOUT = __LAYOUT_JSON__;
const DENSITY_LAYOUT = __DENSITY_LAYOUT_JSON__;
const TS_LAYOUT = __TS_LAYOUT_JSON__;
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

const SUPERSCRIPT = {'-': '⁻', '0':'⁰','1':'¹','2':'²','3':'³','4':'⁴',
                     '5':'⁵','6':'⁶','7':'⁷','8':'⁸','9':'⁹'};
function logColorbar(title, zmin, zmax, extra) {
  // Labels as "10ⁿ" (real density in cm^-3) rather than raw log10
  // (12, 15, 18...) or "1e15" notation -- data stays in log10,
  // only the DISPLAY changes.
  const lo = Math.ceil(zmin), hi = Math.floor(zmax);
  const step = Math.max(1, Math.round((hi - lo) / 5));
  const vals = [];
  for (let v = lo; v <= hi; v += step) vals.push(v);
  const txt = vals.map(v => '10' + String(v).split('').map(c => SUPERSCRIPT[c] || c).join(''));
  return Object.assign({ title, tickvals: vals, ticktext: txt }, extra || {});
}

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

  // Colorbar positions depend on whether transmittance panel is present,
  // so that phase and transmittance colorbars never overlap vertically.
  const hasT = !!pulse.channels.transmittance;
  const phaseCB = hasT
    ? { title: '\u03b4\u03c6 (rad)', len: 0.30, y: 0.84 }
    : { title: '\u03b4\u03c6 (rad)', len: 0.54, y: 0.72 };

  const traces = [
    { type: 'heatmap', x: scen.z_sim, y: scen.x_sim, z: phiT,
      colorscale: 'RdBu', reversescale: false, zmid: 0,
      zmin: -META.clip, zmax: META.clip,
      colorbar: phaseCB,
      hovertemplate: 'z=%{x:.0f} \u00b5m<br>x=%{y:.0f} \u00b5m<br>\u03b4\u03c6=%{z:.3f} rad<extra></extra>',
      xaxis: 'x', yaxis: 'y' },
    { type: 'scatter', x: scen.z_sim, y: lineSim, mode: 'lines',
      line: { color: '#1f77b4', width: 2 }, showlegend: false,
      xaxis: 'x2', yaxis: 'y2',
      hovertemplate: 'z=%{x:.0f} \u00b5m<br>\u03b4\u03c6 on-axis=%{y:.3f} rad<extra></extra>' },
  ];
  if (hasT) {
    const T = pulse.channels.transmittance;
    const nxT = T[0].length;
    const TT = Array.from({length: nxT}, (_, j) => T.map(row => row[j]));
    traces.push({ type: 'heatmap', x: scen.z_sim, y: scen.x_sim, z: TT,
      colorscale: 'Greys', reversescale: true, zmin: META.tmin, zmax: 1.0,
      colorbar: { title: 'T', len: 0.28, y: 0.15 },
      hovertemplate: 'z=%{x:.0f} \u00b5m<br>x=%{y:.0f} \u00b5m<br>T=%{z:.3f}<extra></extra>',
      xaxis: 'x3', yaxis: 'y3' });
  }
  return traces;
}

function transposeZr(arr) {
  // (Nz, Nr) -> (Nr, Nz), for heatmap(x=z, y=r)
  if (!arr) return [];
  const nr = arr[0].length;
  return Array.from({length: nr}, (_, j) => arr.map(row => row[j]));
}

function mirrorR(r_dens, zMap) {
  // Radially symmetric density: mirror r>=0 to r<0, like the x axis
  // (symmetric) of the phase panel -- purely cosmetic, no information
  // lost or invented.
  if (!zMap) return { r: [], z: [] };
  const zt = transposeZr(zMap);                 // (Nr, Nz)
  const rNeg = r_dens.slice(1).map(v => -v).reverse();
  const zNeg = zt.slice(1).reverse();
  return { r: rNeg.concat(r_dens), z: zNeg.concat(zt) };
}

function buildDensityTraces() {
  // Snapshot (z, r) at the CURRENT PULSE delay -- the REAL density in
  // the meridional plane, as computed by the solver -- not what a
  // camera can see. Reacts to the cursor exactly like the phase panel,
  // unlike the old fixed history.
  const scen = DATA.scenarios[scenarioSel.value];
  const pulse = scen.pulses[Math.min(pulseIdx, scen.pulses.length - 1)];

  const show_rho_e = document.getElementById('cb_rho_e').checked;
  const show_rho_s = document.getElementById('cb_rho_s').checked;
  const traces = [];

  if (!scen.r_dens || !pulse.rho_e_map) return [];
  if (show_rho_e) {
    const m = mirrorR(scen.r_dens, pulse.rho_e_map);
    traces.push(
      { type: 'heatmap', x: scen.z_sim, y: m.r, z: m.z,
        colorscale: 'Viridis', reversescale: false, zmin: META.rho_log_min, zmax: META.rho_log_max,
        colorbar: logColorbar('\u03c1e (cm\u207b\u00b3)', META.rho_log_min, META.rho_log_max, {len: 0.42, y: 0.78}),
        hovertemplate: 'z=%{x:.0f} \u00b5m<br>r=%{y:.0f} \u00b5m<br>log10 \u03c1e=%{z:.2f}<extra></extra>',
        xaxis: 'x', yaxis: 'y' }
    );
  }
  if (show_rho_s && pulse.rho_s_map) {
    const m = mirrorR(scen.r_dens, pulse.rho_s_map);
    traces.push(
      { type: 'heatmap', x: scen.z_sim, y: m.r, z: m.z,
        colorscale: 'Plasma', reversescale: false, zmin: META.rho_log_min, zmax: META.rho_log_max,
        colorbar: logColorbar('\u03c1STE (cm\u207b\u00b3)', META.rho_log_min, META.rho_log_max, {len: 0.42, y: 0.22}),
        hovertemplate: 'z=%{x:.0f} \u00b5m<br>r=%{y:.0f} \u00b5m<br>log10 \u03c1STE=%{z:.2f}<extra></extra>',
        xaxis: 'x2', yaxis: 'y2' }
    );
  }
  return traces;
}

function getTimeSeriesTrace(zIdx, xIdx, avgN) {
  const scen = DATA.scenarios[scenarioSel.value];
  const pulses = scen.pulses;
  const t_vals = pulses.map(p => p.t_disp);
  const z_val = scen.z_sim[zIdx];
  const r_val = scen.x_sim[xIdx];
  
  const phi_vals = pulses.map(pulse => {
    const phi2d = sumChannels(pulse);
    if (!phi2d || !phi2d[zIdx]) return null;
    let sum = 0, count = 0;
    const start = Math.max(0, xIdx - Math.floor((avgN - 1) / 2));
    const end = Math.min(phi2d[0].length, xIdx + Math.ceil((avgN - 1) / 2) + 1);
    for (let j = start; j < end; j++) {
      const v = phi2d[zIdx][j];
      if (v !== null && isFinite(v)) { sum += v; count++; }
    }
    return count > 0 ? sum / count : null;
  });

  return {
    x: t_vals,
    y: phi_vals,
    mode: 'lines+markers',
    name: `z=${z_val.toFixed(0)} µm, r=${r_val.toFixed(0)} µm`,
    hovertemplate: 't=%{x:.0f} fs<br>δφ=%{y:.3f} rad<extra></extra>'
  };
}

function buildTimeSeriesTraces() {
  const scen = DATA.scenarios[scenarioSel.value];
  if (!scen) return [];
  const z_arr = scen.z_sim;
  const x_arr = scen.x_sim;
  const traces = [];
  
  const mode = document.getElementById('ts_mode').value;
  const zIdx = parseInt(document.getElementById('ts_z').value);
  const xIdx = parseInt(document.getElementById('ts_r').value);
  const avgN = parseInt(document.getElementById('ts_avg').value);

  if (mode === 'single') {
    traces.push(getTimeSeriesTrace(zIdx, xIdx, avgN));
  } else if (mode === 'compare_z') {
    const zIndices = [
      0, 
      Math.floor(z_arr.length * 0.25), 
      Math.floor(z_arr.length * 0.5), 
      Math.floor(z_arr.length * 0.75), 
      z_arr.length - 1
    ];
    zIndices.forEach(zi => {
      if (zi >= 0 && zi < z_arr.length) {
        traces.push(getTimeSeriesTrace(zi, xIdx, avgN));
      }
    });
  } else if (mode === 'compare_r') {
    const xIndices = [
      0, 
      Math.floor(x_arr.length * 0.25), 
      Math.floor(x_arr.length * 0.5), 
      Math.floor(x_arr.length * 0.75), 
      x_arr.length - 1
    ];
    xIndices.forEach(xi => {
      if (xi >= 0 && xi < x_arr.length) {
        traces.push(getTimeSeriesTrace(zIdx, xi, avgN));
      }
    });
  }
  return traces;
}

function updateTsLabels() {
  const scen = DATA.scenarios[scenarioSel.value];
  const zIdx = parseInt(document.getElementById('ts_z').value);
  const xIdx = parseInt(document.getElementById('ts_r').value);
  const avgN = parseInt(document.getElementById('ts_avg').value);

  document.getElementById('ts_z_label').textContent = `z = ${scen.z_sim[zIdx].toFixed(1)} µm`;
  document.getElementById('ts_r_label').textContent = `r = ${scen.x_sim[xIdx].toFixed(1)} µm`;
  document.getElementById('ts_avg_label').textContent = avgN;
}

function findPulseIdx(t_target) {
    const scen = DATA.scenarios[scenarioSel.value];
    let bestIdx = 0;
    let minDiff = Infinity;
    scen.pulses.forEach((p, idx) => {
        const diff = Math.abs(p.t_disp - t_target);
        if (diff < minDiff) {
            minDiff = diff;
            bestIdx = idx;
        }
    });
    return bestIdx;
}

function render() {
  const scen = DATA.scenarios[scenarioSel.value];
  const pulse = scen.pulses[Math.min(pulseIdx, scen.pulses.length - 1)];
  Plotly.react('plot', buildTraces(), LAYOUT, {responsive: true});

  const show_rho_e = document.getElementById('cb_rho_e').checked;
  const show_rho_s = document.getElementById('cb_rho_s').checked;
  const showDensity = HAS_DENSITY && (show_rho_e || show_rho_s);

  if (showDensity) {
    Plotly.react('densityPlot', buildDensityTraces(), DENSITY_LAYOUT, {responsive: true});
    document.getElementById('densityHeading').style.display = 'block';
    document.getElementById('densityPlot').style.display = 'block';
  } else {
    document.getElementById('densityHeading').style.display = 'none';
    document.getElementById('densityPlot').style.display = 'none';
  }

  // Render Time Series Plot
  Plotly.react('timeSeriesPlot', buildTimeSeriesTraces(), TS_LAYOUT, {responsive: true});

  document.getElementById('pulseLabel').textContent =
    `t = ${pulse.t_disp.toFixed(0)} fs from start`;
  document.getElementById('status').textContent =
    `scenario = ${scenarioSel.value} | probe = ${META.probe_nm[scenarioSel.value].toFixed(0)} nm | ` +
    `gaussian focus = ${META.z_focus[scenarioSel.value].toFixed(0)} \u00b5m | ` +
    `t=0 (sim start) at box entry, z = ${META.t0_ref[scenarioSel.value].toFixed(0)} \u00b5m`;
}

function onScenarioChange() {
  const scen = DATA.scenarios[scenarioSel.value];
  
  // Default time: 3082 fs
  pulseIdx = findPulseIdx(3082);
  document.getElementById('pulseSlider').max = scen.pulses.length - 1;
  document.getElementById('pulseSlider').value = pulseIdx;

  // Update TS sliders limits based on new scenario
  const zMax = scen.z_sim.length - 1;
  const rMax = scen.x_sim.length - 1;
  document.getElementById('ts_z').max = zMax;
  // Keep Z at 7 or max if scenario is smaller
  document.getElementById('ts_z').value = Math.min(7, zMax);
  
  // Force R to 0.0 um (center of symmetric x_sim array)
  document.getElementById('ts_r').max = rMax;
  document.getElementById('ts_r').value = Math.floor(rMax / 2);
  
  updateTsLabels();
  render();
}

scenarioSel.addEventListener('change', onScenarioChange);
document.getElementById('pulseSlider').addEventListener('input', e => { pulseIdx = +e.target.value; render(); });
['ch_drude', 'ch_kerr', 'ch_ste', 'cb_rho_e', 'cb_rho_s'].forEach(id =>
  document.getElementById(id).addEventListener('change', render));

// Time series controls
['ts_mode', 'ts_z', 'ts_r', 'ts_avg'].forEach(id => {
  document.getElementById(id).addEventListener('input', () => {
    updateTsLabels();
    render();
  });
});

// Initialize labels for the first time
onScenarioChange();
</script>

</body>
</html>
"""


def build_layout(xlim, ylim, with_transmittance=True):
    """Three rows when transmittance is available: phase map, on-axis
    lineout, transmittance map. The experiment measures both parts of the
    index, the page must show both. Domains are chosen so that colorbars
    of the phase and transmittance panels never overlap vertically."""
    if not with_transmittance:
        return {
            "template": "plotly_white",
            "margin": {"l": 70, "r": 30, "t": 20, "b": 50},
            "height": 700,
            "xaxis":  {"domain": [0.0, 1.0], "anchor": "y", "range": xlim, "matches": "x2"},
            "yaxis":  {"domain": [0.44, 1.0], "anchor": "x", "range": list(ylim), "title": "x (\u00b5m)"},
            "xaxis2": {"domain": [0.0, 1.0], "anchor": "y2", "range": xlim,
                       "title": "Propagation z (\u00b5m) \u2014 lab frame (0 = interface)"},
            "yaxis2": {"domain": [0.0, 0.38], "anchor": "x2", "title": "\u03b4\u03c6 on-axis (rad)"},
        }
    return {
        "template": "plotly_white",
        "margin": {"l": 70, "r": 30, "t": 20, "b": 50},
        "height": 900,
        "xaxis":  {"domain": [0.0, 1.0], "anchor": "y", "range": xlim, "matches": "x3"},
        "yaxis":  {"domain": [0.68, 1.0], "anchor": "x", "range": list(ylim),
                   "title": "x (\u00b5m) \u2014 \u03b4\u03c6"},
        "xaxis2": {"domain": [0.0, 1.0], "anchor": "y2", "range": xlim, "matches": "x3"},
        "yaxis2": {"domain": [0.38, 0.60], "anchor": "x2", "title": "\u03b4\u03c6 on-axis (rad)"},
        "xaxis3": {"domain": [0.0, 1.0], "anchor": "y3", "range": xlim,
                   "title": "Propagation z (\u00b5m) \u2014 lab frame (0 = interface)"},
        "yaxis3": {"domain": [0.0, 0.30], "anchor": "x3", "range": list(ylim),
                   "title": "x (\u00b5m) \u2014 T"},
    }

def build_density_layout(xlim, rlim):
    """Density panel layout: snapshot (z, r) at the current delay,
    electrons on top / STE on bottom -- not a fixed (z, t) history."""
    return {
        "template": "plotly_white",
        "margin": {"l": 70, "r": 30, "t": 20, "b": 50},
        "height": 600,
        "xaxis":  {"domain": [0.0, 1.0], "anchor": "y", "range": xlim, "matches": "x2",
                   "title": "Propagation z (\u00b5m) \u2014 lab frame"},
        "yaxis":  {"domain": [0.55, 1.0], "anchor": "x", "range": list(rlim),
                   "title": "r (\u00b5m) \u2014 \u03c1e"},
        "xaxis2": {"domain": [0.0, 1.0], "anchor": "y2", "range": xlim, "matches": "x"},
        "yaxis2": {"domain": [0.0, 0.45], "anchor": "x2", "range": list(rlim),
                   "title": "r (\u00b5m) \u2014 \u03c1STE"},
    }

def build_ts_layout():
    """Layout for the time series extraction plot."""
    return {
        "template": "plotly_white",
        "margin": {"l": 70, "r": 150, "t": 20, "b": 50},
        "height": 400,
        "xaxis": {"title": "Time from simulation start (fs)"},
        "yaxis": {"title": "\u03b4\u03c6 (rad)"},
        "legend": {"x": 1.02, "y": 1, "font": {"size": 10}}
    }

def run_slider_scenario(sim_dir, lmd_nm, apply_na_filter,
                         coarsen_z=1, coarsen_r=1, t_step_fs=None):
    """Scans DIRECTLY the simulated temporal grid, without indirection through
    a pulse counter. t=0 (displayed) = the earliest actually simulated instant
    (t_sub_fs.min()), and the phase reference for Eqs. (3)-(4) is the box
    entry (z=0, sim frame) -- not an auto-detected collapse point: this is a
    fixed, unambiguous choice, whereas collapse depends on the run.

    `t_step_fs`: cursor sampling step. None = native cube resolution
    (t_sub_fs), no sub- or super-sampling.
    """
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

    # t=0 = box entry (z_sim = 0), fixed and concrete reference --
    # not the collapse point (which shifts t=0 from run to run).
    t0_ref_um = float(sim["z_sim_um"][0])
    t0_ref_lab_um = t0_ref_um + z_focus

    t_sub_fs = sim["t_sub_fs"]
    t_start_fs = float(t_sub_fs[0])       # displayed as "t = 0"
    t_end_fs = float(t_sub_fs[-1])
    if t_step_fs is None:
        t_list = np.asarray(t_sub_fs, float)
    else:
        t_list = np.arange(t_start_fs, t_end_fs + 0.5 * t_step_fs, t_step_fs)

    r_dens_um = sim["rlist_um"][::coarsen_r]

    frames = []
    z_sim_ref = x_sim_ref = None
    for t_exp in t_list:
        t_exp = float(t_exp)
        z_lab, x_um, phases = channel_phases_2d(
            sim, t_exp,
            apply_na_filter=apply_na_filter, NA_eff=NA_eff, lmd_um=lmd_um,
            probe_lmd_nm=lmd_nm, t0_ref_um=t0_ref_um)
        if z_sim_ref is None:
            z_sim_ref = z_lab; x_sim_ref = x_um
        # Density snapshot (z, r) AT THIS DELAY -- not a complete history
        # as before: reacts to the cursor exactly like the phase panel.
        _, _, rho_e_log, rho_s_log = density_maps_2d(sim, t_exp, t0_ref_um=t0_ref_um)
        frames.append(dict(
            t_exp=t_exp, t_disp=t_exp - t_start_fs,
            channels={k: (_to_json_array(v) if v is not None else None) for k, v in phases.items()},
            rho_e_map=(_to_json_array(rho_e_log[:, ::coarsen_r], 3) if rho_e_log is not None else None),
            rho_s_map=(_to_json_array(rho_s_log[:, ::coarsen_r], 3) if rho_s_log is not None else None),
        ))

    return dict(z_sim=_to_json_array(z_sim_ref, 3), x_sim=_to_json_array(x_sim_ref, 3),
                r_dens=_to_json_array(r_dens_um, 3),
                pulses=frames, z_focus=z_focus, probe_nm=float(lmd_nm),
                t0_ref_lab_um=t0_ref_lab_um)

def build_explorer_html(sim_dirs, save="abel_phase_explorer.html", *,
                         raw_dir=None, energy_uJ=4.0,
                         t_step_fs=None, lmd_nm=None,
                         apply_na_filter=True,
                         phase_clip=0.2, t_min=None, xlim=None, ylim=(-50.0, 50.0),
                         coarsen_z=1, coarsen_r=1, rho_log_min=12.0, rho_log_max=21.0):
    """
    sim_dirs : dict {scenario_name: path_to_dir_containing_result.npz+params.json}
               (exactly what the ablation loop in the notebook produces).
    raw_dir  : optional path to experimental npz shots (unified_filament_slider_v3
               naming convention). Left as None => sim-only page.
    t_step_fs: delay cursor step. None = native cube resolution (t_sub_fs),
               the cursor then scans EVERYTHING that was simulated, from the
               earliest instant (displayed as "t=0") to the latest.
    t_min    : transmittance colorbar floor. None (default) = computed
               automatically from the actual minimum of T across all pulses --
               with a fixed value (0.75 previously), any absorption stronger
               than this floor collapsed to a single color, making the panel
               unreadable whenever transmittance dropped lower (frequent: it
               went down to 0.09 on test data).
    """
    scenarios = {}
    z_focus_by_scenario = {}
    probe_nm_by_scenario = {}
    t0_ref_by_scenario = {}

    for name, sim_dir in sim_dirs.items():
        try:
            scenarios[name] = run_slider_scenario(
                sim_dir, lmd_nm, apply_na_filter,
                coarsen_z=coarsen_z, coarsen_r=coarsen_r, t_step_fs=t_step_fs)
            z_focus_by_scenario[name] = scenarios[name].pop("z_focus")
            probe_nm_by_scenario[name] = scenarios[name].pop("probe_nm")
            t0_ref_by_scenario[name] = scenarios[name].pop("t0_ref_lab_um")
            print(f"[{name}] {len(scenarios[name]['pulses'])} time steps from {sim_dir} "
                  f"(t=0 = simulation start, box entry at z={t0_ref_by_scenario[name]:.0f} um lab)")
        except Exception as e:
            print(f"[{name}] unavailable ({e})")

    if not scenarios:
        raise RuntimeError("No scenario loaded -- check sim_dirs (result.npz + params.json).")

    has_exp = False
    if raw_dir is not None and rotate is not None:
        exp_lmd_nm = float(lmd_nm) if lmd_nm is not None else float(next(iter(probe_nm_by_scenario.values())))
        test_file = Path(raw_dir) / raw_filename(energy_uJ, 0, exp_lmd_nm)
        has_exp = test_file.exists()
        if not has_exp:
            print(f"[exp] no files found in {raw_dir} -- experiment panel disabled")

    xlim_max = max(float(s["z_sim"][-1]) for s in scenarios.values()) + 20
    xlim_eff = list(xlim) if xlim is not None else [-50.0, xlim_max]

    # r-axis for the density panel: snapshot (z, r) at EACH delay (not
    # a fixed (z, t) history) -- same principle as the phase panel.
    r_max_dens = 0.0
    has_density = False
    for sc in scenarios.values():
        if sc.get("r_dens") and any(p.get("rho_e_map") is not None for p in sc["pulses"]):
            r_max_dens = max(r_max_dens, float(sc["r_dens"][-1]))
            has_density = True

    # r-axis mirrored on the JS side (radially symmetric density), like the x
    # of the phase -- the default range must therefore be symmetric too.
    density_layout = build_density_layout(xlim_eff, [-r_max_dens, r_max_dens]) if has_density else {}
    ts_layout = build_ts_layout()

    data_obj = dict(scenarios=scenarios)
    has_T = any(p["channels"].get("transmittance") is not None
                for sc in scenarios.values() for p in sc["pulses"])
    if t_min is None:
        t_vals = [v for sc in scenarios.values() for p in sc["pulses"]
                  for row in (p["channels"].get("transmittance") or [])
                  for v in row if v is not None]
        t_min = min(t_vals) if t_vals else 0.75
    meta = dict(clip=float(phase_clip), z_focus=z_focus_by_scenario,
                probe_nm=probe_nm_by_scenario,
                t0_ref=t0_ref_by_scenario,
                tmin=float(t_min),
                rho_log_min=float(rho_log_min), rho_log_max=float(rho_log_max))
    layout = build_layout(xlim_eff, ylim, with_transmittance=has_T)

    # Dynamic figure height: taller when transmittance row is present.
    fig_height = 900 if has_T else 700

    first = next(iter(scenarios.values()))
    p0 = first["pulses"][0]
    init_label = f"t = {p0['t_disp']:.0f} fs from start"

    html = (HTML_TEMPLATE
        .replace("__FIGURE_HEIGHT__", str(fig_height))
        .replace("__PULSE_MAX__", str(len(first["pulses"]) - 1))
        .replace("__INIT_LABEL__", init_label)
        .replace("__DATA_JSON__", json.dumps(data_obj))
        .replace("__LAYOUT_JSON__", json.dumps(layout))
        .replace("__META_JSON__", json.dumps(meta))
        .replace("__HAS_EXP__", "true" if has_exp else "false")
        .replace("__DENSITY_LAYOUT_JSON__", json.dumps(density_layout))
        .replace("__TS_LAYOUT_JSON__", json.dumps(ts_layout))
        .replace("__HAS_DENSITY__", "true" if has_density else "false"))

    with open(save, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"-> Saved: {save}  ({len(html)/1e6:.1f} MB)")
    return save


if __name__ == "__main__":
    print("Import this module and call build_explorer_html(sim_dirs=...).")