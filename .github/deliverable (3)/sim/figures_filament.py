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


def count_refocusing_cycles(res, I_clamp=5e13, z_shift_um=0.0,
                            min_prominence_frac=0.3, verbose=True):
    """Maxima locaux de l'intensité crête au-dessus de I_clamp : le premier est
    le collapse initial, les suivants sont les cycles de refocalisation.

    `min_prominence_frac` est indispensable et vaut 0.3 par défaut. Sans lui,
    find_peaks(height=I_clamp) compte toutes les rides d'une rampe montante :
    sur un run où l'intensité grimpe régulièrement de 5.0 à 6.1e13, il
    rapportait 12 "cycles" espacés de 6 µm et d'intensité strictement
    croissante -- alors qu'un cycle de refocalisation ALTERNE (collapse, arrêt,
    re-collapse), donc son intensité monte PUIS redescend, avec une
    proéminence de l'ordre de 100 % du niveau et non de 2 %.
    """
    from scipy.signal import find_peaks
    z_um = z_um_of(res, z_shift_um)
    Imax = np.asarray(res["Imax_z"])
    idx, _ = find_peaks(Imax, height=I_clamp,
                        prominence=min_prominence_frac * I_clamp)
    if verbose:
        print(f"{len(idx)} maximum(aux) local(aux) au-dessus de I_clamp={I_clamp:.0e} "
              f"et de proeminence > {min_prominence_frac:.0%} :")
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


def clamping_density(n2, I_clamp_Wcm2, wavelength_m, peak_factor=6.0):
    """Densité électronique attendue au clampage : rho = 2 rho_c n2 I_clamp.

    C'est l'égalité des deux termes d'indice, n2*I = rho/(2 rho_c), qui EST la
    définition du clampage. `peak_factor` traduit que le pic dépasse
    transitoirement l'équilibre : Couairon 2005 rapporte 2-4e20 là où ce bilan
    donne 6.2e19, soit un facteur ~6.

    Important : rho_c décroît quand lambda augmente, donc la densité attendue à
    1030 nm est PLUS BASSE qu'à 800 nm. Comparer un run à 1030 nm à la bande
    2-4e20 de Couairon, mesurée à 800 nm et à ses w0/énergie, n'a pas de sens.
    """
    from scipy.constants import epsilon_0, m_e, c as c_, elementary_charge as qe_
    rho_c = epsilon_0 * m_e * (2 * np.pi * c_ / wavelength_m)**2 / qe_**2 * 1e-6
    return 2 * rho_c * n2 * I_clamp_Wcm2 * 1e4 * peak_factor


def run_health_check(res, out_dir=None, label="", I_band=(4.5e13, 5.5e13),
                     rho_band=(2e20, 4e20), rho_max=2.1e22,
                     rho_band_source="Couairon 2005 @800nm, 1.1uJ, w0=1um"):
    """Confronte un run à des valeurs de référence et rappelle quels
    interrupteurs ont RÉELLEMENT servi (lus dans params.json).

    `rho_band` par défaut est celle de Couairon 2005 à SES paramètres. Elle
    n'est pas universelle : la densité de clampage vaut 2 rho_c n2 I, donc elle
    dépend de lambda (via rho_c), de n2 et de I_clamp. Pour un run à d'autres
    paramètres, passer `rho_band` calculée avec `clamping_density()`, sinon le
    verdict HORS BANDE compare des choses différentes.
    """
    print(f"=== {label} ===")
    I_pk = float(np.max(res["Imax_z"]))
    z_pk = z_um_of(res)[int(np.argmax(res["Imax_z"]))]
    flag = "OK" if I_band[0] <= I_pk <= I_band[1] else "HORS BANDE"
    print(f"  I_max            = {I_pk:.3e} W/cm2 @ z_sim={z_pk:+.0f} um   [attendu {I_band[0]:.1e}-{I_band[1]:.1e} -> {flag}]")

    rho_pk = float(np.max(res["rho_rz"]))
    flag = "OK" if rho_band[0] <= rho_pk <= rho_band[1] else "HORS BANDE"
    print(f"  rho_e max        = {rho_pk:.3e} cm-3   [ref {rho_band[0]:.0e}-{rho_band[1]:.0e} "
          f"({rho_band_source}) -> {flag}]")
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


# ================================================================================
#  Cartes de déphasage sonde, style expérience pompe-sonde
# ================================================================================
def _populations_at(res, t_local_fs, tau_r_s=330e-15, tau_ste_s=None):
    """rho_e, rho_s, I au temps local demandé pour chaque plan z.

    Dans la fenêtre du solveur : interpolation dans le cube.
    AU-DELÀ : intégration analytique. C'est légitime parce qu'à t = 5 t_p le
    champ vaut exp(-25) ~ 1e-11 de son maximum, donc les équations de
    population se réduisent à

        drho_e/dt = -rho_e/tau_r
        drho_s/dt = +rho_e/tau_r - rho_s/tau_ste

    dont la solution est exacte :
        rho_e(t) = rho_e0 exp(-t/tau_r)
        rho_s(t) = rho_s0 exp(-t/tau_s) + B [exp(-t/tau_r) - exp(-t/tau_s)],
        B = (rho_e0/tau_r) / (1/tau_s - 1/tau_r)
    (et rho_s(t) = rho_s0 + rho_e0[1-exp(-t/tau_r)] si tau_ste est None,
    c'est-à-dire sans canal de décroissance des STE.)

    Ça permet d'atteindre 2 ps sans agrandir la fenêtre temporelle, donc sans
    payer le Nt correspondant. Aucune physique thermique ou acoustique n'est
    incluse : au-delà de quelques ps l'expérience en voit, pas ce modèle.
    """
    t_sub = np.asarray(res["t_sub_fs"], float)
    rho_e_c = np.asarray(res["rho_rzt"], float)
    rho_s_c = np.asarray(res["rho_s_rzt"], float)
    I_c = np.asarray(res["I_rzt"], float)
    nz = rho_e_c.shape[0]
    t_local = np.atleast_1d(t_local_fs) * np.ones(nz)

    inside = t_local <= t_sub[-1]
    rho_e = np.empty(rho_e_c.shape[:2])
    rho_s = np.empty_like(rho_e)
    I = np.zeros_like(rho_e)

    # --- dans la fenêtre : plus proche voisin en t (le cube est sous-échantillonné) ---
    if inside.any():
        k = np.clip(np.searchsorted(t_sub, t_local[inside]), 1, len(t_sub) - 1)
        k = np.where(np.abs(t_sub[k - 1] - t_local[inside]) < np.abs(t_sub[k] - t_local[inside]),
                     k - 1, k)
        iz = np.arange(nz)[inside]
        rho_e[inside] = rho_e_c[iz, :, k]
        rho_s[inside] = rho_s_c[iz, :, k]
        I[inside] = I_c[iz, :, k]

    # --- au-delà : évolution analytique depuis le dernier plan temporel ---
    if (~inside).any():
        iz = np.arange(nz)[~inside]
        dt = (t_local[~inside] - t_sub[-1])[:, None] * 1e-15      # s
        e0, s0 = rho_e_c[iz, :, -1], rho_s_c[iz, :, -1]
        decay_r = np.exp(-dt / tau_r_s)
        rho_e[~inside] = e0 * decay_r
        if tau_ste_s is None:
            rho_s[~inside] = s0 + e0 * (1.0 - decay_r)
        else:
            decay_s = np.exp(-dt / tau_ste_s)
            B = (e0 / tau_r_s) / (1.0 / tau_ste_s - 1.0 / tau_r_s)
            rho_s[~inside] = s0 * decay_s + B * (decay_r - decay_s)
        # I reste nul : plus de champ
    return rho_e, rho_s, I


def probe_phase_map(res, delay_fs, lambda_probe_m=490e-9, E_tr_eV=4.2,
                    n2=3.54e-20, tau_r_s=330e-15, tau_ste_s=None, n_g=1.4627,
                    include=("drude", "ste", "kerr"), x_half_um=20.0, dx_um=0.1):
    """Déphasage sonde phi(x, z) [rad] au délai pompe-sonde `delay_fs`.

    Sonde TRANSVERSE : la pompe atteint le plan z au temps z/v_g, donc le temps
    local vu en z vaut t_local(z) = delay - z/v_g. Même convention que
    unified_filament_slider_v3.py.
    """
    from scipy.constants import epsilon_0, m_e, c as c_, elementary_charge as qe_
    from keldysh import n_sellmeier as _ns

    n0p = _ns(lambda_probe_m)
    nc = epsilon_0 * m_e * (2 * np.pi * c_ / lambda_probe_m)**2 / qe_**2 * 1e-6
    E_probe = 1239.84193 / (lambda_probe_m * 1e9)
    f_ste = E_probe**2 / (E_tr_eV**2 - E_probe**2)
    lam_um = lambda_probe_m * 1e6

    z_um = z_um_of(res)
    v_g = 299.792458 / n_g                       # µm/ps
    t_local = delay_fs - z_um / (v_g * 1e-3)     # fs

    rho_e, rho_s, I = _populations_at(res, t_local, tau_r_s, tau_ste_s)

    # axe radial du cube (sous-échantillonné si rho_r_stride > 1)
    if res.get("r_sub") is not None and np.asarray(res["r_sub"]).shape != ():
        r_pos = np.asarray(res["r_sub"], float) * 1e6
    else:
        r_full = r_um_of(res)
        r_pos = r_full[len(r_full) // 2:]
    r_pos = r_pos[:rho_e.shape[1]]

    den = 2.0 * n0p * nc
    dn = np.zeros_like(rho_e)
    if "drude" in include:
        dn -= rho_e / den
    if "ste" in include:
        dn += f_ste * rho_s / den
    if "kerr" in include:
        dn += n2 * I * 1e4

    x_max = float(min(x_half_um, r_pos[-1]))
    x_um = np.linspace(-x_max, x_max, int(2 * x_max / dx_um) + 1)
    A = build_abel_matrix(r_pos, x_um)
    phi = (2.0 * np.pi / lam_um) * (dn @ A.T)
    return x_um, z_um, phi


def plot_delay_series(res, delays_fs, save=None, clip_rad=None, z_face_um=None,
                      x_half_um=15.0, z_lim=None, **kw):
    """Planche façon expérience pompe-sonde : une colonne par délai.

    Ligne du haut  : vue de face phi(x, y) au plan z_face_um (axisymétrique).
    Ligne du bas   : vue de côté phi(x, z), axe de propagation vertical,
                     comme sur les figures expérimentales.
    """
    import matplotlib.pyplot as plt
    n = len(delays_fs)
    fig, axes = plt.subplots(2, n, figsize=(1.9 * n, 7.5),
                             gridspec_kw=dict(height_ratios=[1, 2.6]))
    axes = np.atleast_2d(axes)
    maps = [probe_phase_map(res, d, x_half_um=x_half_um, **kw) for d in delays_fs]
    if clip_rad is None:
        # Percentile 99.5 et non le maximum : un run ou le milieu s'ionise
        # totalement produit quelques pixels a des centaines de radians, qui
        # ecrasent toute l'echelle et rendent la planche uniformement blanche.
        allv = np.concatenate([np.abs(p).ravel() for _, _, p in maps])
        clip_rad = float(np.percentile(allv, 99.5)) or 1.0

    for j, (d, (x_um, z_um, phi)) in enumerate(zip(delays_fs, maps)):
        # Par defaut le plan ou le signal est maximum. Imposer z_face_um a une
        # valeur theorique (L_c par exemple) donne une vignette vide des que le
        # run ne se comporte pas comme prevu -- exactement le cas a debugger.
        iz = (int(np.argmin(np.abs(z_um - z_face_um))) if z_face_um is not None
              else int(np.argmax(np.abs(phi).max(axis=1))))
        prof = phi[iz]
        X, Y = np.meshgrid(x_um, x_um)
        R = np.hypot(X, Y)
        face = np.interp(R, np.abs(x_um[x_um >= 0]), prof[x_um >= 0], left=0, right=0)
        axes[0, j].imshow(face, cmap="bwr", vmin=-clip_rad, vmax=clip_rad,
                          extent=[x_um[0], x_um[-1], x_um[0], x_um[-1]], origin="lower")
        axes[0, j].set_title(f"{d/1000:.2f} ps", fontsize=11, fontweight="bold")
        axes[0, j].tick_params(labelsize=6)
        if j == 0:
            axes[0, j].set_ylabel("y (µm)", fontsize=8)

        im = axes[1, j].imshow(phi, cmap="bwr", vmin=-clip_rad, vmax=clip_rad, aspect="auto",
                               extent=[x_um[0], x_um[-1], z_um[-1], z_um[0]])
        # z_lim : cadrer sur la zone d'interaction pour comparer aux figures
        # experimentales, dont l'axe long ne couvre que quelques dizaines de um.
        if z_lim is not None:
            axes[1, j].set_ylim(max(z_lim), min(z_lim))
        axes[1, j].axhline(z_um[iz], color="k", lw=0.5, ls=":")
        axes[1, j].tick_params(labelsize=6)
        axes[1, j].set_xlabel("x (µm)", fontsize=8)
        if j == 0:
            axes[1, j].set_ylabel("z (µm), propagation", fontsize=8)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.015, pad=0.01,
                 label="phase delay (rad)")
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig


def check_entrance_intensity(energy_uJ, w0_m, delta_t_s, begin_m, wavelength_m, n0,
                             I_clamp_Wcm2=5e13, verbose=True):
    """Intensité crête au plan d'ENTRÉE, comparée au clampage.

    À écrire avant tout `run()`. Le solveur pose I0 = 2P/(pi w0^2) au waist ;
    si le plan de départ est proche du waist et que l'énergie est grande, on
    démarre au-dessus du seuil d'ionisation, le milieu s'ionise dès le premier
    micron (rho_e -> rho_max), l'énergie est absorbée et il ne reste rien à
    propager. Le run se termine normalement, sans erreur : c'est pour ça que
    ce contrôle doit être explicite.

    Renvoie (I_entree, w_entree, z_min_safe) où z_min_safe est la distance
    minimale au waist pour rester sous I_clamp.
    """
    tp = delta_t_s / np.sqrt(2 * np.log(2))
    P = energy_uJ * 1e-6 / (tp * np.sqrt(np.pi / 2))
    I0 = 2 * P / (np.pi * w0_m**2) * 1e-4                 # W/cm^2, au waist
    z_R = 2 * np.pi * n0 / wavelength_m * w0_m**2 / 2
    w_in = w0_m * np.sqrt(1 + (begin_m / z_R)**2)
    I_in = I0 * (w0_m / w_in)**2
    z_safe = z_R * np.sqrt(max(I0 / I_clamp_Wcm2 - 1, 0.0))
    if verbose:
        print(f"  P_crete   = {P*1e-6:.1f} MW")
        print(f"  I au waist= {I0:.2e} W/cm2   ({I0/I_clamp_Wcm2:.1f} x I_clamp)")
        print(f"  w(entree) = {w_in*1e6:.1f} um  ->  I(entree) = {I_in:.2e} W/cm2")
        if I_in > I_clamp_Wcm2:
            print(f"  /!\\ ENTREE AU-DESSUS DU CLAMPAGE : le milieu va s'ioniser des le")
            print(f"       premier micron et absorber l'impulsion. Demarrer a |z| > "
                  f"{z_safe*1e6:.0f} um du waist, ou baisser l'energie a "
                  f"{energy_uJ*I_clamp_Wcm2/I0:.2f} uJ.")
        else:
            print(f"  OK : {I_clamp_Wcm2/I_in:.1f} x sous le clampage a l'entree.")
    return I_in, w_in, z_safe


def w0_from_channel_length(L_um, wavelength_m=1030e-9, n0=1.4500):
    """Waist déduit de la longueur du canal observé.

    La longueur sur laquelle un faisceau focalisé reste assez intense pour
    ioniser est fixée par sa longueur de Rayleigh, z_R = pi w0^2 n0/lambda.
    Un canal de longueur L implique donc w0 ~ sqrt(L lambda/(pi n0)).

    C'est une mesure de w0 gratuite, et surtout indépendante de toute
    calibration de caméra : si la simulation étale l'ionisation sur bien plus
    de longueur que l'expérience, c'est que le waist utilisé est trop grand.
    """
    return float(np.sqrt(L_um * wavelength_m * 1e6 / (np.pi * n0)))


# ================================================================================
#  OPL (nm) et transmittance : le format des figures experimentales
# ================================================================================
def probe_sigma(lambda_probe_m, tau_c_s, meff_drude_rel=1.0):
    """Section efficace d'absorption par porteurs libres (Bremsstrahlung
    inverse) À LA LONGUEUR D'ONDE SONDE, en cm².

    Même expression que le `sigmaomega` du solveur, mais évaluée à
    omega_sonde et non omega_pompe. C'est elle qui donne la partie IMAGINAIRE
    de l'indice, donc la transmittance -- la partie réelle (défocalisation)
    seule ne suffit pas à reproduire les cartes expérimentales.
    """
    from scipy.constants import epsilon_0, m_e, c as c_, elementary_charge as qe_
    from keldysh import n_sellmeier as _ns
    n0p = _ns(lambda_probe_m)
    w = 2 * np.pi * c_ / lambda_probe_m
    k = w * n0p / c_
    m = meff_drude_rel * m_e
    return (k * qe_**2 * tau_c_s) / (n0p**2 * m * epsilon_0 * w * (1 + (w * tau_c_s)**2)) * 1e4


def probe_opl_transmittance(res, delay_fs, lambda_probe_m=515e-9, E_tr_eV=4.2,
                            n2=3.54e-20, tau_c_s=1.7e-15, meff_drude_rel=1.0,
                            tau_r_s=330e-15, tau_ste_s=None, n_g=1.4627,
                            include=("drude", "ste", "kerr"),
                            x_half_um=70.0, dx_um=0.25,
                            material=None, linearize=False, xpm_factor=None):
    """OPL [nm] et transmittance le long de la ligne de visée, au délai donné.

    OPL = integrale de Delta_n sur la corde, en nanomètres -- c'est la
    grandeur que trace l'expérience, reliée à la phase par
    phi = 2 pi OPL / lambda (soit 1 rad = 82 nm à 515 nm).

    Deux modeles de reponse coexistent :

    - `material=None` (defaut, comportement historique) : Delta n est la somme
      de trois termes ecrits a la main -- Drude -rho/(2 n0 rho_c) avec la masse
      NUE et sans correction de collision, une bande STE unique de force
      d'oscillateur 1 a E_tr_eV, et n2*I sans facteur de phase croisee. La
      transmittance vient d'un sigma calcule separement, donc rien ne garantit
      qu'elle soit coherente avec la phase.
    - `material=<MaterialResponse>` (voir sim/permittivity.py) : phase ET
      absorption sortent de la MEME permittivite complexe, suivant Martin et
      al., PRB 55, 5799 (1997), Eq. (2) -- deux bandes STE avec leurs forces
      d'oscillateur et leurs largeurs, masse effective 0.5 m_e, deplation de la
      bande de valence, et n = sqrt(eps) au lieu du developpement au premier
      ordre. A 515 nm cela divise le canal STE par 2.8 et multiplie le canal
      Drude par 2 par rapport au chemin historique.

    `xpm_factor` n'a d'effet que sur le chemin `material` : 2 pour une sonde
    faible a une autre frequence avec un n2 d'auto-modulation, 1 si n2 est deja
    un coefficient ajuste sur une mesure sonde. Defaut : 2.
    """
    from scipy.constants import epsilon_0, m_e, c as c_, elementary_charge as qe_
    from keldysh import n_sellmeier as _ns

    n0p = _ns(lambda_probe_m)
    nc = epsilon_0 * m_e * (2 * np.pi * c_ / lambda_probe_m)**2 / qe_**2 * 1e-6
    E_probe = 1239.84193 / (lambda_probe_m * 1e9)
    f_ste = E_probe**2 / (E_tr_eV**2 - E_probe**2)

    z_um = z_um_of(res)
    v_g = 299.792458 / n_g
    t_local = delay_fs - z_um / (v_g * 1e-3)
    rho_e, rho_s, I = _populations_at(res, t_local, tau_r_s, tau_ste_s)

    if res.get("r_sub") is not None and np.asarray(res["r_sub"]).shape != ():
        r_pos = np.asarray(res["r_sub"], float) * 1e6
    else:
        r_full = r_um_of(res)
        r_pos = r_full[len(r_full) // 2:]
    r_pos = r_pos[:rho_e.shape[1]]

    x_max = float(min(x_half_um, r_pos[-1]))
    x_um = np.linspace(-x_max, x_max, int(2 * x_max / dx_um) + 1)
    A = build_abel_matrix(r_pos, x_um)

    if material is None:
        # ---- chemin historique : trois Delta n ecrits a la main -------------
        den = 2.0 * n0p * nc
        dn = np.zeros_like(rho_e)
        if "drude" in include:
            dn -= rho_e / den
        if "ste" in include:
            dn += f_ste * rho_s / den
        if "kerr" in include:
            dn += n2 * I * 1e4

        opl_nm = (dn @ A.T) * 1e3                   # µm -> nm
        sigma = probe_sigma(lambda_probe_m, tau_c_s, meff_drude_rel)
        tau_opt = sigma * (rho_e @ A.T) * 1e-4      # cm^-3 * µm * cm^2 -> sans dim
        alpha_cm = sigma * rho_e
        f_ste_eff = f_ste
    else:
        # ---- chemin permittivite : phase et absorption depuis le meme eps ---
        from permittivity import XPM
        mat = material
        if getattr(mat, "n2_m2W", 0.0) == 0.0 and n2:
            import copy
            mat = copy.copy(material)
            mat.n2_m2W = n2
        inc = tuple(include)
        if "depletion" not in inc and mat.enable_valence_depletion:
            inc = inc + ("depletion",)
        resp = mat.response(lambda_probe_m, n0p,
                            rho_e_cm3=rho_e, rho_s_cm3=rho_s, I_Wcm2=I,
                            xpm_factor=(XPM if xpm_factor is None else xpm_factor),
                            linearize=linearize, include=inc)
        dn = np.asarray(resp["dn"], float)
        alpha_cm = np.asarray(resp["alpha_cm"], float)
        opl_nm = (dn @ A.T) * 1e3
        # alpha en cm^-1 integre sur une corde en µm -> 1e-4 pour passer en cm
        tau_opt = (alpha_cm @ A.T) * 1e-4
        sigma = mat.sigma_fca_cm2(lambda_probe_m, n0p)
        f_ste_eff = mat.f_ste_effective(lambda_probe_m)

    return dict(x_um=x_um, z_um=z_um, opl_nm=opl_nm,
                transmittance=np.exp(-np.clip(tau_opt, 0, None)),
                sigma_cm2=sigma, f_ste=f_ste_eff,
                # bruts (r, z), pour la vue de dessus qui integre le long de z
                dn_rz=dn, rho_e_rz=rho_e, r_pos_um=r_pos,
                alpha_cm_rz=alpha_cm)


def plot_opl_panel(res, delay_fs, z_face_um=None, z_shift_um=0.0, z_lim=None,
                   r_lim=None, opl_clip_nm=15.0, t_lim=(0.75, 1.15), x_half_um=75.0,
                   rho_max_cm3=None, validity_frac=0.1, title=None, save=None, **kw):
    """Planche 4 panneaux au format des figures expérimentales :
    OPL vue de dessus / de côté, transmittance vue de dessus / de côté."""
    import matplotlib.pyplot as plt
    d = probe_opl_transmittance(res, delay_fs, x_half_um=x_half_um, **kw)
    x, z = d["x_um"], d["z_um"] + z_shift_um
    opl, T = d["opl_nm"], d["transmittance"]

    # --- vue de dessus : la sonde regarde LE LONG de l'axe, donc elle traverse
    # toute la colonne. Il faut integrer Delta_n et rho_e sur TOUT z, pas
    # echantillonner un seul plan. Comme le milieu est axisymetrique autour de
    # z, chaque rayon a la distance r voit Delta_n(r, z) sur tout son trajet :
    # pas de transformee d'Abel ici, une simple integrale en z.
    dn_rz, rho_e_rz = d["dn_rz"], d["rho_e_rz"]
    r_pos = d["r_pos_um"]
    z_m = z * 1e-6
    # np.trapz a disparu dans numpy 2, np.trapezoid n'existe pas avant
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    opl_top_nm = _trapz(dn_rz, z_m, axis=0) * 1e9                  # m -> nm
    # alpha_cm_rz est deja une absorption locale en cm^-1 (chemin permittivite) ;
    # sur le chemin historique elle vaut sigma*rho_e, donc la meme expression
    # marche pour les deux et on n'a plus a re-multiplier par sigma ici.
    tau_top = _trapz(d["alpha_cm_rz"], z_m, axis=0) * 1e2           # cm^-1 * m -> cm
    T_top = np.exp(-np.clip(tau_top, 0, None))

    X, Y = np.meshgrid(x, x)
    R = np.hypot(X, Y)
    # left = valeur a r_pos[0] et non 0 : la grille de Hankel commence a
    # r ~ 1e-2 um, donc le pixel central (R < r_pos[0]) tombait hors domaine
    # et etait mis a zero -- exactement la ou le signal est maximal.
    face_opl = np.interp(R, r_pos, opl_top_nm, left=opl_top_nm[0], right=0.0)
    face_T = np.interp(R, r_pos, T_top, left=T_top[0], right=1.0)

    # le plan repere sur les vues de cote (trait pointille) reste le plus intense
    iz = (int(np.argmin(np.abs(z - z_face_um))) if z_face_um is not None
          else int(np.argmax(np.abs(opl).max(axis=1))))

    fig, ax = plt.subplots(2, 2, figsize=(16, 9),
                           gridspec_kw=dict(width_ratios=[1, 2.9]))
    ext_face = [x[0], x[-1], x[0], x[-1]]
    ext_side = [z[0], z[-1], x[0], x[-1]]

    im0 = ax[0, 0].imshow(face_opl, cmap="bwr", vmin=-opl_clip_nm, vmax=opl_clip_nm,
                          extent=ext_face, origin="lower")
    ax[0, 0].set_title("top OPL (integre sur toute la colonne)"); ax[0, 0].set_xlabel("x [um]"); ax[0, 0].set_ylabel("y [um]")

    im1 = ax[0, 1].imshow(opl.T, cmap="bwr", vmin=-opl_clip_nm, vmax=opl_clip_nm,
                          extent=ext_side, origin="lower", aspect="auto")
    ax[0, 1].set_title("side OPL")
    ax[0, 1].set_xlabel("z from interface [um]"); ax[0, 1].set_ylabel("r from axis [um]")
    fig.colorbar(im1, ax=ax[0, 1], label="OPL [nm]", fraction=0.02, pad=0.01)

    im2 = ax[1, 0].imshow(face_T, cmap="gray", vmin=t_lim[0], vmax=t_lim[1],
                          extent=ext_face, origin="lower")
    ax[1, 0].set_title("top transmittance (integre sur toute la colonne)"); ax[1, 0].set_xlabel("x [um]"); ax[1, 0].set_ylabel("y [um]")

    im3 = ax[1, 1].imshow(T.T, cmap="gray", vmin=t_lim[0], vmax=t_lim[1],
                          extent=ext_side, origin="lower", aspect="auto")
    ax[1, 1].set_title("side transmittance")
    ax[1, 1].set_xlabel("z from interface [um]"); ax[1, 1].set_ylabel("r from axis [um]")
    fig.colorbar(im3, ax=ax[1, 1], label="transmittance", fraction=0.02, pad=0.01)

    # z_lim / r_lim : caler exactement sur les bornes des figures
    # experimentales, sinon la comparaison visuelle est trompeuse (une zone
    # d'interaction parait large ou etroite selon le cadrage, pas selon la
    # physique -- c'est ce qui avait masque le probleme de waist trop grand).
    for a in (ax[0, 1], ax[1, 1]):
        a.axhline(0, color="k", lw=0.5, ls=":")
        a.axvline(z[iz], color="k", lw=0.5, ls=":")
        if z_lim is not None:
            a.set_xlim(*z_lim)
        if r_lim is not None:
            a.set_ylim(*r_lim)
    for a in (ax[0, 0], ax[1, 0]):
        if r_lim is not None:
            a.set_xlim(*r_lim); a.set_ylim(*r_lim)
    # Domaine de validite : la ou rho_e depasse une fraction de rho_max, le
    # modele sort de son domaine (pas d'enlevement de matiere, Drude a tau_c
    # fixe, pas de renormalisation du gap). On le marque au lieu de laisser
    # croire que la carte y est quantitative.
    if rho_max_cm3 is not None and "rho_rz" in res:
        rho_side = np.asarray(res["rho_rz"])
        half_r = rho_side.shape[1] // 2
        rho_max_z = rho_side[:, half_r:].max(axis=1)
        bad = rho_max_z > validity_frac * rho_max_cm3
        if bad.any():
            for a in (ax[0, 1], ax[1, 1]):
                a.fill_between(z, x[0], x[-1], where=bad, color="lime",
                               alpha=0.13, step="mid", zorder=5)
            zb = z[bad]
            print(f"/!\\ hors domaine de validite (rho_e > {validity_frac:.0%} de rho_max) "
                  f"sur z = {zb.min():.0f} a {zb.max():.0f} um  -- zones hachurees")

    fig.suptitle(title or f"simulation, delay {delay_fs/1000:+.3f} ps"
                          f"   (sigma_probe = {d['sigma_cm2']:.2e} cm2)", fontsize=12)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=140, bbox_inches="tight")
    return fig, d


def max_energy_for_clamping(w0_m, delta_t_s, I_clamp_Wcm2=5e13, fraction=1.0):
    """Énergie maximale (J, dans le milieu) pour que l'intensité AU WAIST reste
    sous `fraction * I_clamp`.

    Utile quand le waist est au plan d'entrée : là I(entrée) = I(waist), donc
    l'énergie est bornée par le clampage, et la borne croît comme w0².
    C'est aussi pour ça que P/P_cr accessible sans dépasser le clampage croît
    en w0² : un waist plus large autorise beaucoup plus de puissance.
    """
    tp = delta_t_s / np.sqrt(2 * np.log(2))
    P = fraction * I_clamp_Wcm2 * 1e4 * np.pi * w0_m**2 / 2
    return P * tp * np.sqrt(np.pi / 2)


def compare_probe_models(res, delay_fs=0.0, lambda_probe_m=515e-9,
                         material=None, x_half_um=70.0, **kw):
    """Confronte le post-traitement historique au modele de permittivite.

    Imprime, au delai donne, le max de |OPL|, la phase correspondante et la
    transmittance minimale pour les deux chemins, canal par canal quand c'est
    possible. Sert a chiffrer ce que change le passage a Martin et al. (1997)
    avant de regenerer quoi que ce soit.
    """
    from permittivity import SIO2_MARTIN1997
    mat = SIO2_MARTIN1997 if material is None else material
    lam_nm = lambda_probe_m * 1e9

    old = probe_opl_transmittance(res, delay_fs, lambda_probe_m=lambda_probe_m,
                                  x_half_um=x_half_um, **kw)
    new = probe_opl_transmittance(res, delay_fs, lambda_probe_m=lambda_probe_m,
                                  x_half_um=x_half_um, material=mat, **kw)
    lin = probe_opl_transmittance(res, delay_fs, lambda_probe_m=lambda_probe_m,
                                  x_half_um=x_half_um, material=mat,
                                  linearize=True, **kw)

    def _row(name, d):
        opl = float(np.abs(d["opl_nm"]).max())
        return (f"  {name:26s} |OPL|max = {opl:9.2f} nm = {opl/lam_nm*2*np.pi:7.3f} rad"
                f"   T min = {d['transmittance'].min():.4f}")

    print(f"Sonde {lam_nm:.0f} nm, delai {delay_fs:+.0f} fs")
    print(_row("historique", old))
    print(_row("Martin 1997 (sqrt eps)", new))
    print(_row("Martin 1997 (linearise)", lin))
    print(f"  f_STE   : historique {old['f_ste']:.4f}  ->  Martin {new['f_ste']:.4f}"
          f"   (x{new['f_ste']/old['f_ste']:.2f})")
    print(f"  sigma_FCA : historique {old['sigma_cm2']:.3e} cm2  ->  "
          f"Martin {new['sigma_cm2']:.3e} cm2   (x{new['sigma_cm2']/old['sigma_cm2']:.2f})")
    ovd = np.asarray(mat.response(lambda_probe_m, 1.4615,
                                  rho_e_cm3=np.asarray(new["rho_e_rz"]))["overdense"])
    if ovd.any():
        print(f"  /!\\ {100*ovd.mean():.1f} % des mailles sont SURDENSES "
              f"(Re(eps) <= 0) : le milieu y reflechit, aucun des deux modeles "
              f"de dephasage n'y a de sens.")
    return dict(legacy=old, martin=new, martin_linear=lin)
