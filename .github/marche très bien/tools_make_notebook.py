import json, sys
from pathlib import Path

def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": list(lines)}

def code(*lines):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": list(lines)}

def S(text):
    """Split a text block into ipynb source lines (newline kept on all but last)."""
    lines = text.strip("\n").split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]

cells = []

cells.append(md(*S("""
# Filamentation SiO2 -- 1030 nm, 4 uJ

Standalone version of `run_z0_probe_sweeps.py`, rewritten around the single
entry point `simulate()` of `run_filament.py`.

It runs in two stages, and the first one needs no GPU.

1. **Check the solver on the CPU.** The equivalence test proves that the
   term-registry rewrite of the nonlinear operator changed no physics, by
   comparing it against a frozen copy of the previous implementation over all
   64 combinations of the six field flags. About one second, no CUDA.
2. **Run the propagation on the GPU** and build the interactive HTML pages for
   the 490, 620 and 690 nm probes.

Run the cells in order. Stage 1 tells you the code is sound before you spend
GPU time on stage 2.
""")))

# ---------------------------------------------------------------- 1. packages
cells.append(md("## 1. Packages"))

cells.append(code(*S('''
import importlib, subprocess, sys

def _missing(mod):
    try:
        importlib.import_module(mod)
        return False
    except ImportError:
        return True

# Pure-python side. matplotlib is only used by the health-check figures,
# plotly is not needed at all: the HTML pages load it from a CDN.
need = [pkg for mod, pkg in (("numpy", "numpy"), ("scipy", "scipy"),
                             ("tqdm", "tqdm"), ("matplotlib", "matplotlib"))
        if _missing(mod)]
if need:
    print("installing:", " ".join(need))
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *need], check=True)

# cupy needs the wheel matching the installed CUDA runtime. Already present on
# Colab GPU runtimes, so this usually does nothing.
if _missing("cupy"):
    cuda_major = None
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
        for tok in out.split():
            if tok.startswith("12."):
                cuda_major = 12
                break
            if tok.startswith("11."):
                cuda_major = 11
                break
    except FileNotFoundError:
        pass
    wheel = "cupy-cuda11x" if cuda_major == 11 else "cupy-cuda12x"
    print(f"installing: {wheel}")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", wheel], check=True)

print("packages ready")
''')))

# ---------------------------------------------------------------- 2. sources
cells.append(md(*S("""
## 2. Point the notebook at the solver

`REPO_DIR` must be the folder that contains `sim/` and `web/`, that is
`.github/marche tres bien` in the repository.

Leave it as `None` to search upward from the working directory. Set it by hand
if the notebook lives somewhere else, for example after uploading the folder to
a Colab session.
""")))

cells.append(code(*S('''
from pathlib import Path
import sys

REPO_DIR = None      # e.g. "/content/marche tr\u00e8s bien"

def find_repo(explicit=None):
    roots = []
    if explicit:
        p = Path(explicit).expanduser().resolve()
        roots.append(p.parent if p.name == "sim" else p)
    here = Path.cwd().resolve()
    for base in (here, *here.parents):
        for name in ("marche tres bien", "marche tr\u00e8s bien"):
            roots += [base / name, base / ".github" / name]
    for root in roots:
        if (root / "sim" / "filament_sim.py").is_file():
            return root
    raise FileNotFoundError(
        "sim/filament_sim.py not found. Set REPO_DIR to the folder that "
        "contains sim/ and web/.")

REPO = find_repo(REPO_DIR)
SIM, WEB = REPO / "sim", REPO / "web"
for d in (REPO, SIM, WEB):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

print("repo :", REPO)
print("sim  :", SIM.is_dir(), " web :", WEB.is_dir())
print("registry present :", (SIM / "operators.py").is_file())
print("test present     :", (SIM / "test_operators_equivalence.py").is_file())
''')))

# ------------------------------------------------- 3. CPU check
cells.append(md(*S("""
## 3. Check the new solver, no GPU needed

`split()` only uses functions numpy also has, so the test replaces cupy with a
numpy shim and runs on the CPU.

It runs in a **subprocess on purpose**. Importing it here would leave the numpy
shim registered as `cupy` in this kernel, and the real GPU run below would then
silently use numpy instead of CUDA.

Expected: `alpha` bit-identical in all 128 cases, and the right-hand side within
a few units in the last place, which is floating point reassociation and nothing
else.
""")))

cells.append(code(*S('''
import subprocess, sys

r = subprocess.run([sys.executable, str(SIM / "test_operators_equivalence.py")],
                   capture_output=True, text=True)
print(r.stdout or r.stderr)
assert r.returncode == 0, "equivalence test FAILED -- do not trust the run below"
''')))

cells.append(md(*S("""
### The equation the solver will actually integrate

Read straight off `FIELD_TERMS`, the registry `split()` loops over. Also in a
subprocess, for the same reason.
""")))

cells.append(code(*S('''
SHOW_TERMS = r"""
import sys, types, numpy as np
shim = types.ModuleType("cupy"); shim.__dict__.update(np.__dict__)
shim.RawKernel = lambda *a, **k: None
sys.modules["cupy"] = shim
sys.path.insert(0, SIMDIR)
from operators import FIELD_TERMS
print(f"{'flag':32s} {'kind':6s} {'T^':3s}  equation line")
print("-" * 92)
for t in FIELD_TERMS:
    print(f"{t.flag:32s} {t.kind:6s} {t.T_power:<3d}  {t.equation}")
"""
r = subprocess.run([sys.executable, "-c",
                    f"SIMDIR = {str(SIM)!r}\\n" + SHOW_TERMS],
                   capture_output=True, text=True)
print(r.stdout or r.stderr)
''')))

# ------------------------------------------------- 4. GPU
cells.append(md("## 4. GPU check"))

cells.append(code(*S('''
import cupy as cp

n = cp.cuda.runtime.getDeviceCount()
assert n > 0, "no CUDA device: the propagation below cannot run"
props = cp.cuda.runtime.getDeviceProperties(0)
free_b, total_b = cp.cuda.runtime.memGetInfo()
print(f"device : {props['name'].decode()}")
print(f"memory : {free_b/1e9:.1f} GB free / {total_b/1e9:.1f} GB")
print(f"cupy   : {cp.__version__}")
''')))

# ------------------------------------------------- 5. parameters
cells.append(md(*S("""
## 5. Parameters

Every physical parameter is a named argument of `simulate()`, so the whole
configuration of a run is one call and a misspelled name raises instead of
being silently ignored.

Two notes carried over from the original script.

`ENERGY_INCIDENT_UJ` is the energy **before** the sample. `apply_fresnel=True`
multiplies it by the transmission of the air-glass interface, so the solver
receives the energy actually inside the glass.

`TMAX_FACTOR` sets the comoving window, `tmax = TMAX_FACTOR * tp`. At the
historical 5.0 it covers only about +/-1.1 ps, too short to see the STE decay
of roughly 1 ps: past `tmax` the cube simply has no data and the HTML cursor
plateaus. Doubling it together with `Nt` (5 to 10, 2048 to 4096) keeps the same
`dt` and doubles the window, at roughly twice the run time.
""")))

cells.append(code(*S('''
import numpy as np

# ---- pump ------------------------------------------------------------------
PUMP_WAVELENGTH_M   = 1030e-9
ENERGY_INCIDENT_UJ  = 4.0
SX_UM, SY_UM        = 11.5, 11.0
DELTA_T_S           = 263e-15

# ---- probe, 90 deg from the pump (Nomarski / Abel) -------------------------
PROBE_WAVELENGTHS_NM = (490.0, 620.0, 690.0)

# ---- material, SiO2 --------------------------------------------------------
N2_M2W        = 2.74e-20     # Kerr
UI_EV, US_EV  = 9.0, 6.0     # band gap, STE gap
E_TR_EV       = 4.2          # STE resonance seen by the pump
MEFF_REL      = 0.64         # reduced mass in the Keldysh rate
MEFF_DRUDE_REL= 1.0          # effective mass in sigma_w
TAU_C_S       = 1.7e-15      # electron collision time
TAU_R_S       = 330e-15      # trapping, N -> N_STE
TAU_STE_S     = 1e-12        # STE decay to the ground state
RHO_MAX_CM3   = 2.1e22       # N_at
F_R           = 0.18         # Raman fraction
TAU_D_S       = 32e-15
TAU_S_S       = 12e-15

# ---- box and grid ----------------------------------------------------------
BEGIN_M, END_M = 0.0, 350e-6
NZ, NT, NR     = 9000, 4096, 1024
R_FACTOR       = 8.0
TMAX_FACTOR    = 10.0
SAVE_STRIDE    = 20
RHO_T_STRIDE   = 8
RHO_R_STRIDE   = 2

# ---- HTML ------------------------------------------------------------------
# T_STEP_HTML_FS is the cursor step, not the cube resolution. Leaving it at
# None would put every cube instant in the page, several hundred of them, each
# carrying a phase map AND a density map, and the file would run into gigabytes.
T_STEP_HTML_FS = 67.0
HTML_COARSEN_Z = 4
PHASE_CLIP     = 0.2
Z_LIM_UM       = (0.0, 350.0)
X_LIM_UM       = (-50.0, 50.0)

OUT_ROOT = "runs_z0_probe_sweeps"

print(f"w0 = {np.sqrt(SX_UM*SY_UM):.2f} um   from sx={SX_UM}, sy={SY_UM} um")
''')))

# ------------------------------------------------- 6. run
cells.append(md(*S("""
## 6. Run

`simulate()` prints the grid summary, then the ON/OFF listing of every term of
both equations, then propagates and writes one HTML page per probe wavelength.

The listing is read from the registry checked in section 3, so it is what the
solver assembles and not a separate description of it.

To switch a term off, pass its flag here, for example
`enable_kerr_raman=False`. A run is cached under `OUT_ROOT/run_tag`, and
`reuse_cached=True` picks it up instead of recomputing, so re-running this cell
after changing only the HTML options is cheap. Changing any solver source file
invalidates the cache through `code_fingerprint()`.
""")))

cells.append(code(*S('''
from run_filament import simulate

out = simulate(
    # pump
    wavelength_m=PUMP_WAVELENGTH_M,
    energy_incident_uJ=ENERGY_INCIDENT_UJ,
    apply_fresnel=True,
    spot_sx_um=SX_UM, spot_sy_um=SY_UM,
    delta_t_s=DELTA_T_S,

    # material
    n2_m2W=N2_M2W, Ui_eV=UI_EV, Us_eV=US_EV, E_tr_eV=E_TR_EV,
    meff_rel=MEFF_REL, meff_drude_rel=MEFF_DRUDE_REL,
    tau_c_s=TAU_C_S, tau_r_s=TAU_R_S, tau_ste_s=TAU_STE_S,
    rho_max_cm3=RHO_MAX_CM3,
    f_R=F_R, tau_d_s=TAU_D_S, tau_s_s=TAU_S_S,

    # box and grid
    begin_m=BEGIN_M, end_m=END_M,
    Nz=NZ, Nt=NT, Nr=NR, R_factor=R_FACTOR, tmax_factor=TMAX_FACTOR,
    save_stride=SAVE_STRIDE,
    rho_t_stride=RHO_T_STRIDE, rho_r_stride=RHO_R_STRIDE,

    # every term of the field equation, on
    enable_kerr_instantaneous=True,
    enable_kerr_raman=True,
    enable_self_steepening=True,
    enable_photoionization_loss=True,
    enable_plasma_absorption=True,
    enable_plasma_defocusing=True,
    enable_ste_index=True,
    enable_space_time_focusing=True,
    enable_spectral_filter=True,

    # every term of the carrier equations, on
    enable_avalanche=True,
    enable_recombination=True,
    enable_ste=True,

    # probes and output
    probe_wavelengths_nm=PROBE_WAVELENGTHS_NM,
    out_root=OUT_ROOT,
    make_html=True,
    html_t_step_fs=T_STEP_HTML_FS,
    html_coarsen_z=HTML_COARSEN_Z,
    html_phase_clip=PHASE_CLIP,
    html_t_min=None,          # None = colorbar floor from the actual minimum
    html_z_lim_um=Z_LIM_UM,
    html_x_lim_um=X_LIM_UM,

    reuse_cached=True,
    verbose=True,
    sim_dir=str(SIM),
)

print("\\noutput directory:", out["out_dir"])
''')))

# ------------------------------------------------- 7. results
cells.append(md("## 7. The HTML pages"))

cells.append(code(*S('''
from pathlib import Path

for probe_nm, path in out["html"].items():
    p = Path(path)
    size = p.stat().st_size / 1e6 if p.exists() else 0.0
    print(f"{probe_nm:>5.0f} nm  {size:7.1f} MB  {p}")
''')))

cells.append(md(*S("""
Open one of them below. The pages are large, so a browser tab is usually more
comfortable than an inline frame. On Colab, use the file browser on the left to
download them.
""")))

cells.append(code(*S('''
from IPython.display import IFrame

first = list(out["html"].values())[0]
IFrame(src=str(first), width="100%", height=900)
''')))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

dest = Path(sys.argv[1])
dest.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", dest, len(cells), "cells")
