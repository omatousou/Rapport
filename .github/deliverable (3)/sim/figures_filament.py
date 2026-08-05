"""
Helpers de tracé partagés entre notebooks (contours de fluence, intensité
crête, densités électroniques, pertes d'énergie) + les longueurs
caractéristiques de filamentation (P_cr, L_c de Marburger).

Ces fonctions vivaient jusqu'ici dans les cellules 4-5 de
`term_ablation_study.ipynb`, donc elles n'étaient pas réutilisables par un
second notebook. Elles sont ici à l'identique (mêmes corrections et mêmes
commentaires : première traversée pour la FWHM, préférence pour les traces
on-axis pleine résolution, etc.) pour pouvoir être importées :

    from figures_filament import plot_fig7_fluence_contours, marburger_collapse

Le module ne dépend que de numpy/scipy/matplotlib -- il est importable sans
GPU, contrairement à `filament_sim` qui a besoin de cupy.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c

__all__ = [
    "load_scenario_npz", "z_um_of", "r_um_of", "onaxis_index", "peak_z_um",
    "plot_fig7_fluence_contours", "plot_fig8_peak_intensity",
    "plot_fig9_electron_density", "plot_fig12_energy_losses",
    "plot_fig13_free_vs_trapped", "plot_free_vs_trapped_vs_z",
    "plot_scenario_summary", "fluence_level_extent", "run_health_check",
    "count_refocusing_cycles",
    "critical_power", "entrance_radius", "marburger_collapse",
    "build_abel_matrix", "decompose_probe_phase",
]


# ================================================================================
#  Accès npz / axes
# ================================================================================
def load_scenario_npz(out_dir, ignore_stale=False):
    """Relit un result.npz déjà calculé, SAUF s'il a été produit par une
    version antérieure du solveur (empreinte du code dans params.json).
    Renvoie None si absent, périmé ou illisible -> déclenche un recalcul."""
    npz_path = Path(out_dir) / "result.npz"
    if not npz_path.exists():
        return None
    pj = Path(out_dir) / "params.json"
    if not ignore_stale and pj.exists():
        try:
            old_fp = json.loads(pj.read_text()).get("code_fingerprint")
        except Exception:
            old_fp = None
        try:
            from filament_sim import code_fingerprint
            new_fp = code_fingerprint()
        except Exception:
            new_fp = old_fp  # pas de cupy dispo : on ne peut pas vérifier
        if old_fp != new_fp:
            print(f"[{Path(out_dir).name}] cache PERIME "
                  f"(empreinte {old_fp} != {new_fp}) -> recalcul")
            return None
    try:
        return dict(np.load(npz_path, allow_pickle=True))
    except Exception as exc:
        # Un run tué en cours d'écriture laissait un npz tronqué ; le relire
        # levait un zlib "invalid block type". On le traite comme absent.
        print(f"[{Path(out_dir).name}] result.npz ILLISIBLE "
              f"({type(exc).__name__}: {exc}) -> ignoré, recalcul")
        return None


def z_um_of(res, z_shift_um=0.0):
    """z_shift_um replace l'origine sur la face d'entrée (comme les articles,
    z=0 = entrée) au lieu de l'origine du solveur (z=0 = foyer linéaire).
    N'affecte que l'affichage."""
    return np.asarray(res["z"]) * 1e6 + z_shift_um


def r_um_of(res):
    r = np.asarray(res["r"])
    return r * 1e6 if np.max(np.abs(r)) < 1e-3 else r


def onaxis_index(res):
    return int(np.argmin(np.abs(np.asarray(res["r"]))))


def peak_z_um(res):
    """z (µm) où l'intensité crête on-axis est maximale."""
    z_um = z_um_of(res)
    return float(z_um[int(np.argmax(res["Imax_z"]))])


# ================================================================================
#  Longueurs caractéristiques (Marburger / Dawes)
# ================================================================================
def critical_power(n2, wavelength, n0, factor=3.77):
    """P_cr = factor*lambda^2/(8 pi n0 n2). factor=3.77 (gaussienne),
    3.72 (profil de Townes)."""
    return factor * wavelength**2 / (8 * np.pi * n0 * n2)


def entrance_radius(w0, begin, wavelength, n0):
    """Rayon RÉEL du faisceau au plan d'entrée z=begin, pour une enveloppe
    `gaussian_focused` de waist w0 au foyer (z=0).

    C'est w0*|1 + 2i*begin/b| avec b = k*w0^2, exactement le facteur `curv`
    de envelope_gaussian_focused(). Point important : pour un faisceau
    fortement focalisé, ce rayon est BEAUCOUP plus grand que w0 (typiquement
    x10), et c'est LUI qu'il faut mettre dans L_DF = k w^2/2 pour Marburger,
    pas le waist focal -- sinon L_c est sous-estimée d'un facteur ~100."""
    k = 2 * np.pi * n0 / wavelength
    z_R = k * w0**2 / 2
    return w0 * np.sqrt(1.0 + (begin / z_R)**2)


def marburger_collapse(P_in, P_cr, w_input, wavelength, n0, f_ext=None):
    """Longueur de collapse de Dawes-Marburger.

        L_c = 0.367 L_DF / sqrt([ (P_in/P_cr)^1/2 - 0.852 ]^2 - 0.0219)

    avec L_DF = k w_input^2/2 la longueur de Rayleigh AU PLAN D'ENTRÉE.
    Si f_ext est fourni (focalisation externe), renvoie aussi
    1/L_cf = 1/L_c + 1/f_ext.

    Renvoie (ratio, L_DF, L_c, L_cf). L_c/L_cf valent nan sous le seuil.
    """
    ratio = P_in / P_cr
    k = 2 * np.pi * n0 / wavelength
    L_DF = k * w_input**2 / 2
    if ratio <= 0.852**2:
        return ratio, L_DF, np.nan, np.nan
    inner = (np.sqrt(ratio) - 0.852)**2 - 0.0219
    if inner <= 0:
        return ratio, L_DF, np.nan, np.nan
    L_c = 0.367 * L_DF / np.sqrt(inner)
    L_cf = 1.0 / (1.0 / L_c + 1.0 / f_ext) if f_ext else np.nan
    return ratio, L_DF, L_c, L_cf


def count_refocusing_cycles(res, I_clamp=5e13, z_shift_um=0.0, verbose=True):
    """Maxima locaux de l'intensité crête au-dessus de I_clamp : le premier est
    le collapse initial, les suivants sont les cycles de refocalisation."""
    from scipy.signal import find_peaks
    z_um = z_um_of(res, z_shift_um)
    Imax = np.asarray(res["Imax_z"])
    idx, _ = find_peaks(Imax, height=I_clamp)
    if verbose:
        print(f"{len(idx)} maximum(aux) local(aux) au-dessus de I_clamp={I_clamp:.0e} :")
        for i, ip in enumerate(idx):
            print(f"  cycle {i+1}: z = {z_um[ip]:+9.2f} µm   I = {Imax[ip]:.3e} W/cm²")
        if len(idx) >= 2:
            print("  espacements (µm):", np.round(np.diff(z_um[idx]), 2))
    return idx, z_um[idx] if len(idx) else np.array([])


# ================================================================================
#  Panneaux élémentaires
# ================================================================================
def _plot_peak_intensity_ax(ax, res, label, I_clamp=5e13, show_clamp=True,
                            z_shift_um=0.0, xlim=None, ylim=None):
    ax.plot(z_um_of(res, z_shift_um), res["Imax_z"], lw=1.6, label=label)
    if show_clamp:
        ax.axhline(I_clamp, ls="--", color="crimson", lw=1, label=f"I_clamp≈{I_clamp:.0e}")
    ax.set_yscale("log")
    if xlim is not None: ax.set_xlim(*xlim)
    if ylim is not None: ax.set_ylim(*ylim)
    ax.set_xlabel("z (µm)"); ax.set_ylabel("Peak intensity (W/cm²)")
    ax.set_title("Peak intensity vs z")


def _plot_electron_density_ax(ax, res, label, rho_lines=(1e20, 2e20),
                              nc_probe_cm3=None, show_lines=True, z_shift_um=0.0,
                              xlim=None, ylim=(1e15, 1e21)):
    i_axis = onaxis_index(res)
    rho_onaxis = res["rho_rz"][:, i_axis]
    ax.plot(z_um_of(res, z_shift_um), np.clip(rho_onaxis, 1e-3, None), lw=1.6, label=label)
    if show_lines:
        for rl in rho_lines:
            ax.axhline(rl, ls="--", color="orange" if rl == max(rho_lines) else "crimson", lw=1)
        if nc_probe_cm3 is not None:
            ax.axhline(nc_probe_cm3, ls=":", color="purple", lw=1,
                       label=f"n_c(probe)={nc_probe_cm3:.2e}")
    ax.set_yscale("log")
    if xlim is not None: ax.set_xlim(*xlim)
    if ylim is not None: ax.set_ylim(*ylim)
    ax.set_xlabel("z (µm)"); ax.set_ylabel("On-axis ρ_e (cm⁻³)")
    ax.set_title("On-axis electron density vs z")


def _plot_free_vs_trapped_ax(ax, res, z_target_um, label="", z_shift_um=0.0,
                             xlim=None, ylim=(1e16, 1e22), ylim_I=None):
    # Préfère la trace on-axis PLEINE résolution (rho_onaxis_t/I_onaxis_t) au
    # cube rho_rzt sous-échantillonné : le taux multiphotonique est très
    # sensible à l'intensité (~I^K), donc un pic étroit manqué entre deux
    # échantillons du cube fait paraître le résultat faux alors que le noyau
    # CUDA (grille pleine) l'a correctement calculé.
    z_um = z_um_of(res)
    iz = int(np.argmin(np.abs(z_um - z_target_um)))
    ir = onaxis_index(res)

    if res.get("rho_onaxis_t") is not None and res.get("t_full_fs") is not None:
        t_fs = np.asarray(res["t_full_fs"])
        rho_e, rho_s, I_t = res["rho_onaxis_t"][iz, :], res["rho_s_onaxis_t"][iz, :], res["I_onaxis_t"][iz, :]
    elif res.get("rho_rzt") is not None and np.asarray(res["rho_rzt"]).shape != ():
        t_fs = np.asarray(res["t_sub_fs"])
        rho_e, rho_s, I_t = res["rho_rzt"][iz, ir, :], res["rho_s_rzt"][iz, ir, :], res["I_rzt"][iz, ir, :]
    else:
        ax.text(0.5, 0.5, "rho_rzt/rho_onaxis_t indisponible\n(relance avec rho_t_stride > 0)",
                ha="center", va="center", fontsize=9)
        ax.set_axis_off()
        return

    # Plancher à ~0 (pas à ylim[0]) pour que les points réellement sous la
    # fenêtre tombent HORS du cadre au lieu d'être remontés sur un faux
    # plateau à l'intérieur.
    ax.plot(t_fs, np.clip(rho_e, 1e-30, None), color="black", lw=1.4, label="ρ_e libre")
    ax.plot(t_fs, np.clip(rho_s, 1e-30, None), color="tab:blue", lw=1.4, label="ρ_s piégé")
    ax.set_yscale("log")
    if xlim is not None: ax.set_xlim(*xlim)
    if ylim is not None: ax.set_ylim(*ylim)
    ax.set_xlabel("Time (fs)"); ax.set_ylabel("ρ (cm⁻³)")
    ax.set_title(f"Free vs trapped @ z={z_um[iz] + z_shift_um:.0f}µm  [{label}]")

    ax2 = ax.twinx()
    ax2.plot(t_fs, I_t, "--", color="crimson", lw=1.0, label="Pulse intensity")
    ax2.set_yscale("log")
    if ylim_I is not None: ax2.set_ylim(*ylim_I)
    ax2.set_ylabel("Intensity (W/cm²)", color="crimson")
    ax2.tick_params(axis="y", colors="crimson")

    l1, lab1 = ax.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lab1 + lab2, loc="upper left", fontsize=7)


def _plot_fluence_contours_ax(ax, res, levels=(0.1, 1.0, 5.0, 10.0), label="",
                              z_shift_um=0.0, xlim=None, rlim=None):
    z_um = z_um_of(res, z_shift_um)
    r_um = r_um_of(res)
    fluence = res["fluence_rz"]  # déjà en J/cm² (invE2 inclut m^2->cm^2)

    cs = ax.contour(z_um, r_um, fluence.T, levels=levels, colors="black", linewidths=0.8)
    ax.clabel(cs, inline=True, fontsize=6, fmt="%.1f J/cm²")

    # FWHM/2 : demi-largeur à mi-hauteur radiale, PREMIÈRE traversée depuis
    # l'axe (avec interpolation), pas la dernière : pendant la défocalisation
    # plasma le profil développe des anneaux, et un anneau externe repassant
    # au-dessus de la mi-hauteur faisait sauter l'ancienne version très loin
    # sur le bord, déformant la courbe exactement dans la zone intéressante.
    half = 0.5 * fluence.max(axis=1)
    fwhm_half_um = np.full(len(z_um), np.nan)
    Nr_pos = len(r_um) // 2
    r_pos = r_um[Nr_pos:]
    for iz in range(len(z_um)):
        prof = fluence[iz, Nr_pos:]
        below = np.where(prof < half[iz])[0]
        if below.size and below[0] > 0:
            j = below[0]
            f0, f1 = prof[j - 1], prof[j]
            w = 0.0 if f0 == f1 else (f0 - half[iz]) / (f0 - f1)
            fwhm_half_um[iz] = r_pos[j - 1] + w * (r_pos[j] - r_pos[j - 1])
    ax.plot(z_um, fwhm_half_um, "--", color="tab:blue", lw=1, label="Beam radius (FWHM/2)")
    ax.plot(z_um, -fwhm_half_um, "--", color="tab:blue", lw=1)

    if xlim is not None: ax.set_xlim(*xlim)
    if rlim is not None: ax.set_ylim(*rlim)
    ax.set_xlabel("z (µm)"); ax.set_ylabel("r (µm)")
    ax.set_title(f"Fluence contours and beam FWHM vs z  [{label}]")


def _plot_energy_losses_ax(ax, res, label="", z_shift_um=0.0, xlim=(0, 150), ylim=(1e-3, 1e0)):
    """Pertes d'énergie cumulées (fraction de U0) vs z, styles de la Fig. 12
    de Couairon 2005."""
    z_um = z_um_of(res, z_shift_um)
    # NewSim3juillet.py alloue E_plasma_z/E_MPI_z mais ne les remplit jamais :
    # le tracé serait un aplat à zéro qu'on pourrait prendre pour "pas de
    # pertes". Le dire plutôt que de tracer du vide.
    if float(np.max(np.abs(res["E_total_z"]))) == 0.0:
        ax.text(0.5, 0.5, "E_total_z identiquement nul\n(npz produit par un solveur\n"
                          "qui ne calcule pas les pertes)",
                ha="center", va="center", fontsize=10, transform=ax.transAxes)
        ax.set_axis_off()
        print("/!\\ plot_fig12_energy_losses : E_total_z est nul partout -- ce npz ne "
              "contient pas de pertes d'énergie (cf. NewSim3juillet.py).")
        return
    tag = f"{label} -- " if label else ""
    ax.semilogy(z_um, np.clip(res["E_total_z"],  1e-6, None), "-",  color="black", lw=1.8, label=f"{tag}Plasma+Photo")
    ax.semilogy(z_um, np.clip(res["E_plasma_z"], 1e-6, None), "--", color="black", lw=1.4, label=f"{tag}Plasma")
    ax.semilogy(z_um, np.clip(res["E_MPI_z"],    1e-6, None), "-.", color="black", lw=1.4, label=f"{tag}Photo")
    if xlim is not None: ax.set_xlim(*xlim)
    if ylim is not None: ax.set_ylim(*ylim)
    ax.set_xlabel("z (µm)"); ax.set_ylabel("Energy losses (fraction of U0)")
    ax.set_title("Energy losses vs z")


# ================================================================================
#  Figures complètes
# ================================================================================
def plot_fig8_peak_intensity(results, scenario_names=None, I_clamp=5e13, save=None,
                             z_shift_um=0.0, xlim=None, ylim=None):
    """Intensité crête vs z, superposée sur plusieurs scénarios."""
    scenario_names = scenario_names or list(results.keys())
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, name in enumerate(scenario_names):
        _plot_peak_intensity_ax(ax, results[name], name, I_clamp=I_clamp,
                                show_clamp=(i == 0), z_shift_um=z_shift_um,
                                xlim=xlim, ylim=ylim)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    if save: fig.savefig(save, dpi=150)
    return fig


def plot_fig9_electron_density(results, scenario_names=None, rho_lines=(1e20, 2e20),
                               nc_probe_cm3=None, save=None, z_shift_um=0.0,
                               xlim=None, ylim=(1e15, 1e21)):
    """Densité électronique on-axis vs z, superposée sur plusieurs scénarios."""
    scenario_names = scenario_names or list(results.keys())
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, name in enumerate(scenario_names):
        _plot_electron_density_ax(ax, results[name], name, rho_lines=rho_lines,
                                  nc_probe_cm3=nc_probe_cm3, show_lines=(i == 0),
                                  z_shift_um=z_shift_um, xlim=xlim, ylim=ylim)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    if save: fig.savefig(save, dpi=150)
    return fig


def plot_fig13_free_vs_trapped(res, z_target_um=None, label="", save=None,
                               z_shift_um=0.0, xlim=None, ylim=(1e16, 1e22), ylim_I=None):
    """Électrons libres vs piégés + intensité, à z fixé.
    z_target_um=None -> z du pic d'intensité on-axis."""
    if z_target_um is None:
        z_target_um = peak_z_um(res)
    fig, ax = plt.subplots(figsize=(9, 6))
    _plot_free_vs_trapped_ax(ax, res, z_target_um, label=label, z_shift_um=z_shift_um,
                             xlim=xlim, ylim=ylim, ylim_I=ylim_I)
    fig.tight_layout()
    if save: fig.savefig(save, dpi=150)
    return fig


def plot_free_vs_trapped_vs_z(res, save=None, z_shift_um=0.0, rho_max_cm3=None,
                              nc_probe_cm3=None, xlim=None, ylim=None,
                              vlines=None):
    """ρ_e (libres) ET ρ_s (STE) on-axis en fonction de z, sur le même axe.

    Complémentaire de plot_fig13_free_vs_trapped, qui montre les deux
    populations en fonction du TEMPS à un z fixé : ici on suit leur maximum
    temporel le long de la propagation (colonnes rho_rz / rho_s_rz du npz).

    vlines : liste de (z_um, label, couleur) -- p.ex. L_c,f prédite, foyer
    géométrique, face d'entrée.
    """
    z_um = z_um_of(res, z_shift_um)
    i_axis = onaxis_index(res)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(z_um, np.clip(res["rho_rz"][:, i_axis], 1e-3, None),
            lw=1.6, color="black", label="ρ_e (libres)")
    ax.plot(z_um, np.clip(res["rho_s_rz"][:, i_axis], 1e-3, None),
            lw=1.6, color="tab:blue", label="ρ_s (STE)")
    if rho_max_cm3:
        ax.axhline(rho_max_cm3, ls=":", color="gray", lw=1, label=f"ρ_max={rho_max_cm3:.1e}")
    if nc_probe_cm3:
        ax.axhline(nc_probe_cm3, ls="-.", color="purple", lw=1,
                   label=f"n_c(sonde)={nc_probe_cm3:.2e}")
    for zv, lab, col in (vlines or []):
        ax.axvline(zv, ls="--", lw=1.1, color=col, label=lab)
    ax.set_yscale("log")
    if xlim is not None: ax.set_xlim(*xlim)
    if ylim is not None: ax.set_ylim(*ylim)
    ax.set_xlabel("z (µm)"); ax.set_ylabel("On-axis ρ (cm⁻³)")
    ax.set_title("Électrons libres vs excitons auto-piégés le long de z")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    if save: fig.savefig(save, dpi=150)
    return fig


def plot_fig7_fluence_contours(res, levels=(0.1, 1.0, 5.0, 10.0), label="", save=None,
                               z_shift_um=0.0, xlim=None, rlim=None, vlines=None):
    """Contours de fluence + enveloppe FWHM/2 vs z."""
    fig, ax = plt.subplots(figsize=(9, 3.5))
    _plot_fluence_contours_ax(ax, res, levels=levels, label=label,
                              z_shift_um=z_shift_um, xlim=xlim, rlim=rlim)
    for zv, lab, col in (vlines or []):
        ax.axvline(zv, ls="--", lw=1.1, color=col, label=lab)
    ax.legend(fontsize=7)
    fig.tight_layout()
    if save: fig.savefig(save, dpi=150)
    return fig


def plot_fig12_energy_losses(res, label="", save=None, z_shift_um=0.0,
                             xlim=(0, 150), ylim=(1e-3, 1e0)):
    """Pertes d'énergie cumulées (Plasma/Photo/combiné) vs z."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    _plot_energy_losses_ax(ax, res, label=label, z_shift_um=z_shift_um, xlim=xlim, ylim=ylim)
    if ax.get_legend_handles_labels()[0]:   # rien à légender si le npz n'a pas de pertes
        ax.legend(fontsize=8)
    fig.tight_layout()
    if save: fig.savefig(save, dpi=150)
    return fig


def plot_scenario_summary(res, name, z_target_um=None, I_clamp=5e13,
                          rho_lines=(1e20, 2e20), nc_probe_cm3=None,
                          fluence_levels=(0.1, 1.0, 5.0, 10.0), save=None,
                          z_shift_um=0.0, z_xlim=None, I_ylim=None,
                          rho_ylim=(1e15, 1e21), trapped_ylim=(1e16, 1e22),
                          trapped_ylim_I=None, fluence_xlim=None, fluence_rlim=None,
                          show=True):
    """Figure compacte 2x2 pour UN scénario (débug en direct après un run)."""
    if z_target_um is None:
        z_target_um = peak_z_um(res)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(f"Scenario: {name}  (z_target = {z_target_um + z_shift_um:+.0f} µm)",
                 fontsize=13, fontweight="bold")
    _plot_peak_intensity_ax(axes[0, 0], res, name, I_clamp=I_clamp,
                            z_shift_um=z_shift_um, xlim=z_xlim, ylim=I_ylim)
    axes[0, 0].legend(fontsize=7)
    _plot_electron_density_ax(axes[0, 1], res, name, rho_lines=rho_lines,
                              nc_probe_cm3=nc_probe_cm3, z_shift_um=z_shift_um,
                              xlim=z_xlim, ylim=rho_ylim)
    axes[0, 1].legend(fontsize=7)
    _plot_free_vs_trapped_ax(axes[1, 0], res, z_target_um, label=name,
                             z_shift_um=z_shift_um, ylim=trapped_ylim,
                             ylim_I=trapped_ylim_I)
    _plot_fluence_contours_ax(axes[1, 1], res, levels=fluence_levels, label=name,
                              z_shift_um=z_shift_um, xlim=fluence_xlim, rlim=fluence_rlim)
    axes[1, 1].legend(fontsize=7)
    fig.tight_layout()
    if save: fig.savefig(save, dpi=150)
    if show: plt.show()
    return fig


# ================================================================================
#  Diagnostics chiffrés
# ================================================================================
def fluence_level_extent(res, levels=(1.0, 2.0, 3.0), z_shift_um=0.0, label=""):
    """Étendue en z du domaine où la fluence ON-AXIS dépasse chaque niveau
    -> compare des nombres au lieu de comparer des formes à l'oeil."""
    z_um = z_um_of(res, z_shift_um)
    ia = onaxis_index(res)
    f_axis = np.asarray(res["fluence_rz"])[:, ia]
    print(f"[{label}] fluence on-axis max = {f_axis.max():.2f} J/cm² "
          f"@ z = {z_um[int(np.argmax(f_axis))]:.0f} µm")
    out = {}
    for lv in levels:
        m = f_axis >= lv
        if m.any():
            out[lv] = (float(z_um[m][0]), float(z_um[m][-1]))
            print(f"    >= {lv:g} J/cm² : z de {out[lv][0]:6.1f} à {out[lv][1]:6.1f} µm"
                  f"   (longueur {out[lv][1]-out[lv][0]:.0f} µm)")
        else:
            out[lv] = None
            print(f"    >= {lv:g} J/cm² : JAMAIS atteint")
    return out


def run_health_check(res, out_dir=None, label="", I_band=(4.5e13, 5.5e13),
                     rho_band=(2e20, 4e20), rho_max=2.1e22):
    """Confronte un run aux valeurs chiffrées de Couairon 2005 et rappelle
    quels interrupteurs ont RÉELLEMENT servi (lus dans params.json)."""
    print(f"=== {label} ===")
    I_pk = float(np.max(res["Imax_z"]))
    z_pk = z_um_of(res)[int(np.argmax(res["Imax_z"]))]
    flag = "OK" if I_band[0] <= I_pk <= I_band[1] else "HORS BANDE"
    print(f"  I_max            = {I_pk:.3e} W/cm2 @ z_sim={z_pk:+.0f} um   [attendu {I_band[0]:.1e}-{I_band[1]:.1e} -> {flag}]")

    rho_pk = float(np.max(res["rho_rz"]))
    flag = "OK" if rho_band[0] <= rho_pk <= rho_band[1] else "HORS BANDE"
    print(f"  rho_e max        = {rho_pk:.3e} cm-3   [attendu {rho_band[0]:.0e}-{rho_band[1]:.0e} -> {flag}]")
    if rho_pk > 0.5 * rho_max:
        print(f"    /!\\ rho_e s'approche de rho_max={rho_max:.1e} : emballement, pas un clampage physique")

    if "rho_s_rz" in res and np.max(res["rho_s_rz"]) > 0:
        print(f"  rho_s max (STE)  = {float(np.max(res['rho_s_rz'])):.3e} cm-3")

    if "E_total_z" in res and np.max(res["E_total_z"]) > 0:
        loss = float(np.max(res["E_total_z"]))
        print(f"  pertes totales   = {loss*100:.1f} %  ->  transmission {100*(1-loss):.1f} %")

    if out_dir is not None:
        pj = Path(out_dir) / "params.json"
        if pj.exists():
            prm = json.loads(pj.read_text())
            tg = prm.get("toggles", {})
            for key in ("enable_spectral_filter", "enable_space_time_focusing", "enable_ste"):
                val = tg.get(key, prm.get(key))
                print(f"  {key:32s} = {val}")
        else:
            print(f"  (params.json introuvable dans {out_dir})")


# ================================================================================
#  Déphasage sonde : décomposition en canaux
# ================================================================================
def build_abel_matrix(r_um, x_um):
    """Matrice Abel forward, coquilles sur grille r non uniforme :
        phi(x_i) = sum_j A[i,j] f_j  ~=  2 int_|x_i|^inf f(r) r/sqrt(r^2-x^2) dr.
    Identique à celle de unified_filament_slider_v3.py."""
    r_um = np.asarray(r_um, float); x_um = np.asarray(x_um, float)
    r_mid = 0.5 * (r_um[:-1] + r_um[1:])
    r_lo = np.concatenate(([0.0], r_mid))
    r_hi = np.concatenate((r_mid, [r_um[-1] + (r_um[-1] - r_mid[-1])]))
    A = np.zeros((len(x_um), len(r_um)))
    ax = np.abs(x_um)[:, None]
    valid = r_hi[None, :] >= ax
    upper = r_hi[None, :]**2 - ax**2
    lower = np.maximum(r_lo[None, :], ax)**2 - ax**2
    A[valid] = 2.0 * (np.sqrt(np.maximum(upper[valid], 0.0))
                      - np.sqrt(np.maximum(lower[valid], 0.0)))
    return A


def decompose_probe_phase(res, lambda_probe_m=490e-9, E_tr_eV=4.2, n2=3.54e-20,
                          rho_max_cm3=2.1e22, x_half_um=50.0, dx_um=0.1,
                          verbose=True):
    """Borne SUPÉRIEURE du déphasage sonde, canal par canal.

    Utilise les cartes 2D du npz (`rho_rz`, `rho_s_rz`, `Ipeak_rz`), qui sont
    des MAXIMA sur le temps : le résultat majore donc ce que verrait une sonde
    à un délai donné, et les canaux ne culminent pas tous au même instant
    (le Kerr n'existe que pendant l'impulsion, le STE monte après). C'est
    voulu -- le but est de répondre à « quel canal peut produire plusieurs
    radians », pas de reproduire un délai précis.

    Renvoie un dict {canal: (dn_max, phi_max_rad)} plus 'x_um'/'phi_xz'.
    """
    from scipy.constants import epsilon_0, m_e, c as c_, elementary_charge as qe_
    from keldysh import n_sellmeier as _ns

    n0p = _ns(lambda_probe_m)
    nc = epsilon_0 * m_e * (2 * np.pi * c_ / lambda_probe_m)**2 / qe_**2 * 1e-6
    E_probe = 1239.84193 / (lambda_probe_m * 1e9)
    f_ste = E_probe**2 / (E_tr_eV**2 - E_probe**2)
    lam_um = lambda_probe_m * 1e6

    r_full = r_um_of(res)
    half = len(r_full) // 2
    r_pos = r_full[half:]                     # moitié r >= 0
    den = 2.0 * n0p * nc

    chans = {}
    missing = []
    chans["Drude (rho_e)"] = -np.asarray(res["rho_rz"])[:, half:] / den

    if "rho_s_rz" in res and np.asarray(res["rho_s_rz"]).shape != ():
        chans[f"STE (rho_s, E_tr={E_tr_eV} eV)"] = f_ste * np.asarray(res["rho_s_rz"])[:, half:] / den
    else:
        missing.append("STE : 'rho_s_rz' absent du npz")

    # Kerr : Ipeak_rz (max sur t) si présent, sinon reconstruit depuis le cube
    # I_rzt. Ce canal ne doit JAMAIS être sauté en silence : à l'intensité de
    # clampage il vaut à lui seul ~1.4 rad sur 6 µm, donc l'omettre ferait
    # conclure à tort que le STE est seul responsable d'un déphasage trop fort.
    Ipk = None
    if "Ipeak_rz" in res and np.asarray(res["Ipeak_rz"]).shape != ():
        Ipk = np.asarray(res["Ipeak_rz"])[:, half:]
    elif "I_rzt" in res and np.asarray(res["I_rzt"]).shape != ():
        cube = np.asarray(res["I_rzt"])
        if cube.shape[1] == len(r_full):
            Ipk = cube.max(axis=2)[:, half:]
            missing.append("Kerr : 'Ipeak_rz' absent -> reconstruit par max(I_rzt) sur t "
                           "(sous-échantillonné en t, donc légèrement sous-estimé)")
        else:
            missing.append(f"Kerr : 'Ipeak_rz' absent et I_rzt a un axe radial "
                           f"({cube.shape[1]}) != r ({len(r_full)}) -> CANAL KERR NON CALCULE")
    else:
        missing.append("Kerr : ni 'Ipeak_rz' ni 'I_rzt' -> CANAL KERR NON CALCULE")
    if Ipk is not None:
        chans["Kerr (n2*I)"] = n2 * Ipk * 1e4

    x_max = float(min(x_half_um, r_pos[-1]))
    x_um = np.linspace(-x_max, x_max, int(2 * x_max / dx_um) + 1)
    A = build_abel_matrix(r_pos, x_um)

    out = {}
    for name, dn in chans.items():
        phi = (2.0 * np.pi / lam_um) * (dn @ A.T)
        out[name] = (float(np.abs(dn).max()), float(np.abs(phi).max()))
    total = sum(chans.values())
    phi_tot = (2.0 * np.pi / lam_um) * (total @ A.T)
    out["TOTAL (somme des canaux)"] = (float(np.abs(total).max()), float(np.abs(phi_tot).max()))

    if verbose:
        print(f"Sonde {lambda_probe_m*1e9:.0f} nm : n0'={n0p:.4f}, n_c={nc:.3e} cm-3, "
              f"E_probe={E_probe:.3f} eV, f_STE={f_ste:.4f}")
        for m in missing:
            print(f"  /!\\ {m}")
        print(f"{'canal':34s} {'|dn|max':>11s} {'|phi|max (rad)':>15s}")
        for name, (dn_m, phi_m) in out.items():
            print(f"  {name:32s} {dn_m:11.3e} {phi_m:15.2f}")
        rho_e_max = float(np.max(res["rho_rz"]))
        frac = rho_e_max / rho_max_cm3
        print(f"\nrho_e max = {rho_e_max:.3e} cm-3 = {frac*100:.1f} % de rho_max={rho_max_cm3:.1e}")
        if frac > 0.5:
            print("  /!\\ EMBALLEMENT : rho_e approche la densite d'atomes. Ce n'est plus")
            print("      du clampage physique -- tout dephasage calcule la-dessus est faux.")
        elif frac > 0.1:
            print("  (!) rho_e depasse 10 % de rho_max : a surveiller.")

    out["x_um"] = x_um
    out["phi_xz"] = phi_tot
    return out
