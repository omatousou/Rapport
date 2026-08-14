"""
Filamentation SiO2 - pompe 1030 nm, 4 uJ - simulations + HTML
Script compact dérivé de filament_1030nm_4uJ.ipynb.
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.constants import c as c_SI, epsilon_0, m_e, elementary_charge as q_e

# --- 0. Imports & Configuration ---
SIM_DIR = None
def _resolve_pkg_dirs(explicit=None):
    roots = []
    if explicit:
        p = Path(explicit).expanduser().resolve()
        roots.append(p.parent if p.name == "sim" else p)
    here = Path.cwd().resolve()
    for base in (here, *here.parents):
        roots += [base, base / "deliverable (3)", base / ".github" / "deliverable (3)"]
    for root in roots:
        if (root / "sim" / "filament_sim.py").is_file():
            return {n: root / n for n in ("sim", "web") if (root / n).is_dir()}
    return {}

_dirs = _resolve_pkg_dirs(SIM_DIR)
if not _dirs:
    raise ModuleNotFoundError("dossier sim/ introuvable. Renseignez SIM_DIR.")
for _n, _d in _dirs.items():
    sys.path.insert(0, str(_d))

from keldysh import n_sellmeier
import figures_filament as ff
from abel_phase_explorer import build_explorer_html, probe_optics

OUT_ROOT = Path("runs_z0_probe_sweeps")
HTML_DIR = OUT_ROOT / "html"
OUT_ROOT.mkdir(exist_ok=True)
HTML_DIR.mkdir(exist_ok=True)
print("modules du depot :", _dirs)
print("sorties :", OUT_ROOT.resolve())

# --- 1. Paramètres ---
PUMP_WAVELENGTH_M = 1030e-9
assert np.isclose(PUMP_WAVELENGTH_M * 1e9, 1030.0), "La pompe doit rester a 1030 nm."

SX_UM, SY_UM, DELTA_T_S = 11.5, 11.0, 263e-15
W0_M = np.sqrt(SX_UM * SY_UM) * 1e-6
FRESNEL_T = 1.0 - ((1.45 - 1.0) / (1.45 + 1.0))**2
ENERGY_INCIDENT_UJ = 4.0
ENERGY_IN_GLASS_UJ = ENERGY_INCIDENT_UJ * FRESNEL_T

# Sonde perpendiculaire a la pompe (90 deg, setup Nomarski/Abel), balayee en
# longueur d'onde -- voir pump_probe_0d.USER_SIO2_1030 pour la note sur la
# geometrie de croisement (90 deg ici, pas les 10 deg de Martin et al.).
PROBE_WAVELENGTHS_NM = (490.0, 620.0, 690.0)

N2, UI_EV, MEFF_REL, MEFF_DRUDE_REL = 2.74e-20, 9.0, 0.64, 1.0
TAU_C_S, TAU_R_S, TAU_STE_S, RHO_MAX_CM3 = 1.7e-15, 330e-15, 1e-12, 2.1e22
US_EV, F_R, TAU_D_S, TAU_S_S = 6.0, 0.18, 32e-15, 12e-15
BEGIN_M, END_M = 0.0, 350e-6

GRID = dict(Nz=9000, Nt=4096, Nr=1024, R_factor=8.0, save_stride=20, rho_t_stride=8, rho_r_stride=2)

# Fenetre temporelle comobile : tmax = TMAX_FACTOR * tp (tp = largeur 1/e du
# pulse). Au defaut historique (5.0, Nt=2048) elle ne couvrait que +/-1.1 ps
# -- trop court pour voir la decroissance des STE (~1 ps) : au-dela le cube
# n'a plus de donnees et le curseur du HTML plafonne, meme si on lui demande
# un pas plus fin. Double le facteur ET Nt ensemble (5->10, 2048->4096) :
# meme dt qu'avant (donc meme f_Nyq/f0), fenetre doublee a +/-2.2 ps
# (4.5 ps de large), au prix d'un run ~2x plus long.
TMAX_FACTOR = 10.0

# t=0 = l'instant le plus reculent effectivement simule, a l'entree de la
# boite (voir web/abel_phase_explorer.run_slider_scenario) : le curseur
# balaie donc TOUT ce qui a ete simule, depuis le tout debut.
#
# T_STEP_HTML_FS fixe le pas du curseur -- PAS la resolution native du cube
# (t_step_fs=None dans build_explorer_html balaierait tous les points du
# cube, soit plusieurs centaines d'instants avec rho_t_stride=8 : chaque
# instant porte une carte de phase ET une carte de densite, le HTML
# depasserait rapidement le Go). 67 fs donne une centaine d'instants sur
# toute la duree simulee, largement assez pour voir tau_trap (~330 fs) et la
# decroissance des STE (~1 ps).
T_STEP_HTML_FS, HTML_COARSEN_Z = 67.0, 4
PHASE_CLIP, T_MIN = 0.2, 0.75
Z_LIM_HTML, X_LIM_HTML = (0.0, 350.0), (-50.0, 50.0)
print("parametres definis")

# --- 2. Grandeurs dérivées ---
n0 = n_sellmeier(PUMP_WAVELENGTH_M)
k0 = 2.0 * np.pi * n0 / PUMP_WAVELENGTH_M
zR = k0 * W0_M**2 / 2.0

tp = DELTA_T_S / np.sqrt(2.0 * np.log(2.0))
tmax = TMAX_FACTOR * tp
dt = 2.0 * tmax / GRID["Nt"]
dz = (END_M - BEGIN_M) / GRID["Nz"]
R_MAX = GRID["R_factor"] * W0_M
dr = R_MAX / (GRID["Nr"] - 1)

n_saves = GRID["Nz"] // GRID["save_stride"] + 1
Nt_sub = (GRID["Nt"] - 1) // GRID["rho_t_stride"] + 1
Nr_sub = (GRID["Nr"] - 2) // GRID["rho_r_stride"] + 1
DT_CUBE_FS = 2.0 * tmax * 1e15 / (Nt_sub - 1)

print(f"pompe    = {PUMP_WAVELENGTH_M*1e9:.0f} nm")
print(f"w0       = {W0_M*1e6:.2f} um       z_R = {zR*1e6:.0f} um")
print(f"energie  = {ENERGY_INCIDENT_UJ:g} uJ incidents -> {ENERGY_IN_GLASS_UJ:.3f} uJ dans le verre")
print(f"z        = Nz={GRID['Nz']}  dz={dz*1e9:.1f} nm  |  {n_saves} plans, dz_save={dz*GRID['save_stride']*1e6:.2f} um")
print(f"r        = R_max={R_MAX*1e6:.0f} um  dr={dr*1e9:.0f} nm  ({W0_M/dr:.0f} pts dans w0)")
print(f"t        = Nt={GRID['Nt']}  dt={dt*1e15:.2f} fs  f_Nyq/f0={1.0/(2*dt)/(c_SI/PUMP_WAVELENGTH_M):.2f}")
print(f"cube     = {3*n_saves*Nr_sub*Nt_sub*4/1e6:.0f} Mo")
print(f"delai    = dt_cube={DT_CUBE_FS:.1f} fs")
print("\nsondes HTML :")
for probe_nm in PROBE_WAVELENGTHS_NM:
    n0_probe, nc_probe_cm3 = probe_optics(probe_nm)
    print(f"  {probe_nm:.0f} nm : n0={n0_probe:.4f}, n_c={nc_probe_cm3:.3e} cm-3")

# --- 3. Contrôle avant lancement ---
P_cr = ff.critical_power(N2, PUMP_WAVELENGTH_M, n0)
P_in = ENERGY_IN_GLASS_UJ * 1e-6 / (tp * np.sqrt(np.pi / 2.0))
ratio, L_DF, L_c, _ = ff.marburger_collapse(P_in, P_cr, W0_M, PUMP_WAVELENGTH_M, n0)

print(f"P_cr = {P_cr*1e-6:.2f} MW")
print(f"P/P_cr = {ratio:.1f}   L_c = {L_c*1e6:.0f} um")
ff.check_entrance_intensity(ENERGY_IN_GLASS_UJ, W0_M, DELTA_T_S, BEGIN_M, PUMP_WAVELENGTH_M, n0)

# --- 4. Simulation de propagation ---
RUN_TAG = f"z0_350um_{ENERGY_INCIDENT_UJ:g}uJ_pump{PUMP_WAVELENGTH_M*1e9:.0f}nm"
BASE_OUT_DIR = OUT_ROOT / RUN_TAG

res = ff.load_scenario_npz(BASE_OUT_DIR)
if res is None:
    print(f"\n=== lancement {RUN_TAG} ===")
    from filament_sim import run
    res = run(
        Nz=GRID["Nz"], Nt=GRID["Nt"], Nr=GRID["Nr"], R_factor=GRID["R_factor"],
        begin=BEGIN_M, end=END_M, save_stride=GRID["save_stride"], ckpt_every=200, verbose=True,
        wavelength=PUMP_WAVELENGTH_M, energy_uJ=ENERGY_IN_GLASS_UJ,
        w0=W0_M, delta_t=DELTA_T_S, n2=N2, Ui_eV=UI_EV, meff_rel=MEFF_REL, meff_drude_rel=MEFF_DRUDE_REL,
        tau_c=TAU_C_S, tau_r=TAU_R_S, rho_max=RHO_MAX_CM3, Us_eV=US_EV, tau_ste=TAU_STE_S,
        f_R=F_R, tau_d=TAU_D_S, tau_s=TAU_S_S, enable_ste=True,
        lambda_probe=PROBE_WAVELENGTHS_NM[0] * 1e-9,
        rho_t_stride=GRID["rho_t_stride"], rho_r_stride=GRID["rho_r_stride"],
        out_dir=str(BASE_OUT_DIR), envelope="gaussian_focused",
        tmax_factor=TMAX_FACTOR,
    )
else:
    print(f"cache charge : {BASE_OUT_DIR}")

params_path = BASE_OUT_DIR / "params.json"
if params_path.exists():
    params = json.loads(params_path.read_text())
    pump_nm = float(params.get("wavelength_nm", np.nan))
    assert np.isclose(pump_nm, 1030.0), f"Pompe inattendue dans params.json: {pump_nm} nm"
    print(f"params.json confirme : pompe = {pump_nm:.0f} nm")
else:
    print("Attention : params.json absent, le HTML ne pourra pas etre genere.")

ff.run_health_check(res, out_dir=BASE_OUT_DIR, label=RUN_TAG, rho_max=RHO_MAX_CM3)
SIM_DIRS = {f"{ENERGY_INCIDENT_UJ:g}uJ_pump1030nm": str(BASE_OUT_DIR)}

# --- 5. HTML Abel pour 490, 620 et 690 nm ---
html_files = {}

for probe_nm in PROBE_WAVELENGTHS_NM:
    save = HTML_DIR / f"abel_phase_explorer_1030nm_4uJ_probe_{probe_nm:.0f}nm.html"
    print(f"\n=== HTML sonde {probe_nm:.0f} nm ===")
    build_explorer_html(
        sim_dirs=SIM_DIRS, save=str(save), raw_dir=None, energy_uJ=ENERGY_INCIDENT_UJ,
        lmd_nm=probe_nm, t_step_fs=T_STEP_HTML_FS,
        apply_na_filter=True, phase_clip=PHASE_CLIP,
        t_min=T_MIN, xlim=Z_LIM_HTML, ylim=X_LIM_HTML, coarsen_z=HTML_COARSEN_Z,
    )
    html_files[probe_nm] = save

print("\nHTML generes :")
for probe_nm, path in html_files.items():
    print(f"  {probe_nm:.0f} nm -> {path}")
