"""Insere une cellule d'installation des dependances en tete de chaque notebook.

Idempotent : la cellule porte un marqueur, une seconde execution la remplace au
lieu d'en empiler une deuxieme.
"""
import json
import sys
from pathlib import Path

MARKER = "# [deps-install-cell]"

CODE = '''{marker}
# ============================================================================
#  Installation des dependances
# ============================================================================
# A executer une fois, en premier. Si cupy est installe par cette cellule,
# REDEMARRER LE NOYAU avant de continuer.
#
# Deux niveaux :
#   - CPU  : numpy / scipy / matplotlib (+ {extra_txt}). Suffisent pour tout
#            le POST-TRAITEMENT, c'est-a-dire relire un result.npz deja calcule
#            et regenerer les figures.
#   - GPU  : cupy. Necessaire uniquement pour LANCER un calcul
#            (filament_sim.run()), qui est un solveur CUDA.
import importlib
import importlib.util
import re
import shutil
import subprocess
import sys

REQUIRED = ["numpy", "scipy", "matplotlib", "pillow"]{extra_line}
_IMPORT_NAME = {{"pillow": "PIL", "ipywidgets": "ipywidgets"}}


def _pip(*args):
    print("  pip install", *args)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])


print("Dependances CPU :")
for pkg in REQUIRED:
    mod = _IMPORT_NAME.get(pkg, pkg)
    if importlib.util.find_spec(mod) is None:
        _pip(pkg)
    else:
        print(f"  {{pkg:12s}} deja present")

print("\\nDependance GPU (cupy) :")
if importlib.util.find_spec("cupy") is not None:
    import cupy
    print(f"  cupy {{cupy.__version__}} deja present")
    try:
        print(f"  GPU visible : {{cupy.cuda.runtime.getDeviceCount()}} device(s)")
    except Exception as exc:
        print(f"  /!\\\\ cupy importe mais aucun GPU utilisable ({{type(exc).__name__}})")
else:
    # La roue cupy depend de la version de CUDA du pilote : il n'existe pas de
    # paquet "cupy" generique qui marche partout, d'ou la detection.
    cuda_major = None
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(["nvidia-smi"], text=True)
            m = re.search(r"CUDA Version:\\s*(\\d+)\\.", out)
            cuda_major = int(m.group(1)) if m else None
        except Exception:
            pass
    if cuda_major is None:
        print("  aucun GPU NVIDIA detecte -> cupy N'EST PAS installe.")
        print("  Le post-traitement fonctionne quand meme sur un result.npz")
        print("  deja calcule ; seul filament_sim.run() a besoin du GPU.")
    else:
        _pip(f"cupy-cuda{{12 if cuda_major >= 12 else 11}}x")
        print("  cupy installe -> REDEMARRER LE NOYAU avant de continuer.")

print("\\nVersions :")
for _m in ("numpy", "scipy", "matplotlib"):
    try:
        print(f"  {{_m:12s}} {{importlib.import_module(_m).__version__}}")
    except ImportError:
        print(f"  {{_m:12s}} ABSENT")
'''

MD = ("## Installation des dépendances\n\n"
      "À exécuter en premier. Le post-traitement ne demande que numpy / scipy / "
      "matplotlib ; `cupy` n'est nécessaire que pour lancer un nouveau calcul "
      "sur GPU.\n")

TARGETS = {
    ".github/deliverable (3)/notebooks/filament_1030nm_5uJ.ipynb": [],
    ".github/deliverable (3)/notebooks/filament_visualization_1030nm.ipynb": [],
    ".github/deliverable (3)/notebooks/term_ablation_study.ipynb": ["tqdm", "ipywidgets"],
    ".github/latex_project_text/Simulation_07juillet/notebooksimu3juillet.ipynb": ["tqdm"],
}


def build_cells(extra):
    extra_line = ("\nREQUIRED += " + repr(extra)) if extra else ""
    extra_txt = " / ".join(extra) if extra else "pillow"
    code = CODE.format(marker=MARKER, extra_line=extra_line, extra_txt=extra_txt)
    return (
        dict(cell_type="markdown", metadata={}, source=MD.splitlines(keepends=True)),
        dict(cell_type="code", metadata={}, execution_count=None, outputs=[],
             source=code.splitlines(keepends=True)),
    )


def process(path, extra):
    p = Path(path)
    nb = json.loads(p.read_text(encoding="utf-8"))
    cells = nb["cells"]

    # retire une insertion precedente (cellule code marquee + son markdown)
    keep = []
    for c in cells:
        src = "".join(c["source"])
        if c["cell_type"] == "code" and MARKER in src:
            if keep and keep[-1]["cell_type"] == "markdown" \
                    and "Installation des dépendances" in "".join(keep[-1]["source"]):
                keep.pop()
            continue
        keep.append(c)

    at = 1 if (keep and keep[0]["cell_type"] == "markdown") else 0
    md, code = build_cells(extra)
    nb["cells"] = keep[:at] + [md, code] + keep[at:]
    p.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{p.name:42s} cellule inseree en position {at+1}  "
          f"({len(nb['cells'])} cellules)")


if __name__ == "__main__":
    for path, extra in TARGETS.items():
        process(path, extra)
