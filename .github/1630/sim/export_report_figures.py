#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_report_figures.py -- render every report figure to figures/ in one call.

Purpose: produce a stable, named set of PNGs plus a MANIFEST.json describing
what each one demonstrates, so the figures can be dropped into the LaTeX report
without re-deciding each time what they show or where they belong.

Two tiers:

  * `standalone` figures need no simulation at all (they evaluate formulas or
    integrate a 1D model). They always run, including on a machine with no GPU.
  * `run` figures need `results`, the dict of scenario -> result.npz produced by
    notebooks/term_ablation_study.ipynb. Missing scenarios are skipped and
    reported rather than crashing the export.

Usage from the notebook, after the ablation loop has filled `results`:

    import sys; sys.path.insert(0, "../sim")
    from export_report_figures import export_all
    export_all(results, outdir="figures")

Usage from the shell (standalone figures only):

    python export_report_figures.py

Every entry carries `effect` (the physical claim the figure supports) and
`section` (where it belongs in the report). MANIFEST.json records which figures
were actually written, so a partial export is self-documenting.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

__all__ = ["export_all", "FIGURES"]


# ================================================================================
#  Figure registry
# ================================================================================
# Each entry: name -> dict(tier, effect, section, needs, fn)
#   tier    "standalone" | "run"
#   effect  the physical claim the figure is evidence for
#   section target section of the report
#   needs   scenario keys required in `results` (empty for standalone)
#   fn      callable(results, path) -> matplotlib Figure
# ================================================================================

def _f_ionization_rate(results, path):
    from figures_article import fig2_ionization_rate, Params
    prm = Params(wavelength=800e-9, Ui_eV=9.0)
    out = fig2_ionization_rate(prm, save=path, show=False)
    return out[0] if isinstance(out, tuple) else out


def _f_steepening_maps(results, path):
    from self_steepening import fig_steepening_maps
    fig, _ = fig_steepening_maps(save=path, show=False)
    return fig


def _f_steepening_scaling(results, path):
    from self_steepening import fig_shock_scaling
    fig, _ = fig_shock_scaling(save=path, show=False)
    return fig


def _f_populations(results, path):
    from figures_article import fig2_populations
    return fig2_populations(results["full"], _article_params(), save=path, show=False)


def _f_dephasage(results, path):
    from figures_article import fig10_dephasage
    return fig10_dephasage(results["full"], _article_params(), band_um=10.0,
                           save=path, show=False)


def _f_density_vs_z(results, path):
    from figures_article import fig13_electron_density_vs_z
    return fig13_electron_density_vs_z(results["full"], _article_params(),
                                       save=path, show=False)


def _f_avalanche_vs_time(results, path):
    from figures_article import fig10_avalanche_vs_time
    return fig10_avalanche_vs_time(results["full"], _article_params(),
                                   save=path, show=False)


def _f_bulgakova11(results, path):
    from figures_article import fig11_bulgakova
    return fig11_bulgakova(results["bulgakova_1uJ"], save=path, show=False)


def _f_bulgakova12(results, path):
    from figures_article import fig12_bulgakova
    return fig12_bulgakova(results["bulgakova_1uJ"], save=path, show=False)


def _f_steepening_from_run(results, path):
    from self_steepening import fig_steepening_from_run
    return fig_steepening_from_run(results["full"],
                                   results.get("no_self_steepening"),
                                   save=path, show=False)


def _article_params():
    """Params mirroring BASE_PARAMS. Kept in one place so the tau_c that the
    solver actually used is never silently replaced by the dataclass default
    (a ~5.5x error on beta_g/beta_s, see term_ablation_study.ipynb)."""
    from figures_article import Params
    return Params(wavelength=800e-9, Ui_eV=9.0, Us_eV=6.0, meff_rel=0.64,
                  tau_c=1e-14, tau_r=150e-15, rho_max=2.1e22,
                  n2=3.54e-20, lambda_probe=490e-9)


FIGURES = {
    # ---------------- standalone: no simulation required ----------------
    "ionization_rate": dict(
        tier="standalone", fn=_f_ionization_rate, needs=(),
        section="Photoionization",
        effect="The general Keldysh formula used by the solver merges into the "
               "multiphoton limit at low intensity and the tunnel limit at high "
               "intensity. Both limits are asymptotics OF that formula, so the "
               "merging is a parameter-free check that it is assembled correctly."),
    "steepening_maps": dict(
        tier="standalone", fn=_f_steepening_maps, needs=(),
        section="Kerr effect and self-steepening",
        effect="Shock term ON vs OFF. Pure SPM leaves the temporal profile "
               "rigorously unchanged and the spectrum symmetric; the shock term "
               "steepens the trailing edge and makes the blue spectral edge run "
               "several times further than the red one."),
    "steepening_scaling": dict(
        tier="standalone", fn=_f_steepening_scaling, needs=(),
        section="Kerr effect and self-steepening",
        effect="Quantitative test of the Burgers description: the slope-growth "
               "law 1/(1-z/z_shock) is reproduced across input pulse durations, "
               "validating both the factor 3 in the Burgers equation and the "
               "0.39 coefficient in the shock distance."),

    # ---------------- need a solver run ----------------
    "populations": dict(
        tier="run", fn=_f_populations, needs=("full",),
        section="Carrier density rate equations",
        effect="Electron population at the nonlinear focus, separating MPI, "
               "avalanche and STE trapping, with the CUDA kernel output "
               "superposed as validation of the 0D rate equations."),
    "dephasage": dict(
        tier="run", fn=_f_dephasage, needs=("full",),
        section="Interferometry",
        effect="Probe phase shift decomposed into cross-Kerr, Drude plasma and "
               "bound STE contributions, reproducing the positive-negative-"
               "positive sequence measured by pump-probe interferometry."),
    "density_vs_z": dict(
        tier="run", fn=_f_density_vs_z, needs=("full",),
        section="Carrier density rate equations",
        effect="On-axis electron density along the propagation axis for three "
               "ionization models, with the CUDA output superposed."),
    "avalanche_vs_time": dict(
        tier="run", fn=_f_avalanche_vs_time, needs=("full",),
        section="Avalanche ionization",
        effect="Electron density with and without the avalanche channel at fixed "
               "z, showing how much of the final density avalanche actually "
               "contributes for a femtosecond pulse."),
    "steepening_solver": dict(
        tier="run", fn=_f_steepening_from_run, needs=("full",),
        section="Kerr effect and self-steepening",
        effect="Self-steepening as it appears in the full 3D solver, comparing "
               "the complete model with the ablation run that disables the "
               "shock term."),
    "bulgakova_fig11": dict(
        tier="run", fn=_f_bulgakova11, needs=("bulgakova_1uJ",),
        section="Filamentation",
        effect="Fluence, absorbed energy, peak intensity and electron density "
               "maps, cross-checked against published contour levels."),
    "bulgakova_fig12": dict(
        tier="run", fn=_f_bulgakova12, needs=("bulgakova_1uJ",),
        section="Filamentation",
        effect="Sequential energy absorption in four time slices, showing the "
               "absorption front moving toward the laser as the plasma builds."),
}


# ================================================================================
def export_all(results=None, outdir="figures", dpi=170, only=None, verbose=True):
    """Render every runnable figure into `outdir` and write MANIFEST.json.

    results : dict scenario -> result.npz contents. None runs standalone only.
    only    : optional iterable of figure names to restrict the export to.
    Returns the manifest dict.
    """
    results = results or {}
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    manifest, skipped, failed = {}, {}, {}
    names = list(only) if only else list(FIGURES)

    for name in names:
        spec = FIGURES[name]
        missing = [k for k in spec["needs"] if k not in results]
        if missing:
            skipped[name] = f"missing scenario(s): {', '.join(missing)}"
            if verbose:
                print(f"[skip] {name}: {skipped[name]}")
            continue
        path = out / f"{name}.png"
        try:
            fig = spec["fn"](results, str(path))
            plt.close(fig if fig is not None else "all")
            manifest[name] = dict(file=path.name, section=spec["section"],
                                  effect=spec["effect"], tier=spec["tier"])
            if verbose:
                print(f"[ok]   {name} -> {path}")
        except Exception as exc:                       # keep going, report at end
            failed[name] = f"{type(exc).__name__}: {exc}"
            if verbose:
                print(f"[FAIL] {name}: {failed[name]}")
                traceback.print_exc(limit=2)

    doc = dict(figures=manifest, skipped=skipped, failed=failed)
    (out / "MANIFEST.json").write_text(json.dumps(doc, indent=2))
    if verbose:
        print(f"\n{len(manifest)} written, {len(skipped)} skipped, "
              f"{len(failed)} failed -> {out/'MANIFEST.json'}")
    return doc


def main():
    print("Standalone figures only (no `results` supplied).\n")
    export_all(None, outdir="figures")


if __name__ == "__main__":
    main()
