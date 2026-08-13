"""Reduire un result.npz, et extraire la courbe delta_phi(tau) mesuree.

Deux problemes pratiques, deux outils.

1. result.npz fait plusieurs Go et ne passe nulle part. Or le post-traitement
   sonde n'a besoin que d'une petite partie du fichier : les cubes (z, r, t)
   des populations et de l'intensite. `shrink_result_npz` ne garde que ceux-la,
   sous-echantillonnes et en float32, ce qui suffit largement pour une courbe
   de dephasage et divise typiquement la taille par cinquante.
   `export_probe_curve` va plus loin : il ne garde QUE la courbe finale,
   quelques kilo-octets.

2. "ou recuperer les points mesures delta_phi(tau)" : nulle part, ils sont
   deja la. Une courbe temporelle n'est pas une acquisition separee, c'est UNE
   COUPE le long de l'axe des delais dans la pile d'images deja enregistree --
   un pixel, ou une petite fenetre, de chacune des N frames du balayage.
   `measured_phase_curve` fait cette coupe en rejouant le pretraitement
   interferometrique deja present dans web/abel_phase_explorer.py.
"""

import re
from pathlib import Path

import numpy as np

__all__ = [
    "shrink_result_npz", "export_probe_curve",
    "measured_phase_curve", "measured_curve_from_maps",
]


# ================================================================================
#  1. Reduire le npz
# ================================================================================
#  Ce dont le post-traitement sonde a strictement besoin. Tout le reste
#  (fluence_rz, Ipeak_rz, absorbed_rz, les traces on-axis pleine resolution...)
#  sert aux diagnostics de propagation, pas aux cartes de sonde.
_PROBE_KEYS = ("z", "r", "rlist", "r_sub", "t_sub_fs",
               "rho_rzt", "rho_s_rzt", "I_rzt")
_SMALL_KEYS = ("Imax_z", "E_MPI_z", "E_plasma_z", "E_total_z", "E_STE_z")


def shrink_result_npz(in_dir, out_path=None, z_stride=2, t_stride=1, r_stride=1,
                      dtype=np.float32, keep_diagnostics=True, verbose=True):
    """Reecrit un result.npz en ne gardant que de quoi refaire les cartes sonde.

    `z_stride`, `t_stride`, `r_stride` sous-echantillonnent les cubes. Attention
    a `t_stride` : c'est lui qui fixe la resolution en delai, donc le pas de
    balayage temporel qui aura encore un sens. z_stride est le moins couteux,
    les cartes de sonde etant lisses le long de z.

    Renvoie le chemin de sortie.
    """
    src = Path(in_dir)
    npz = src / "result.npz" if src.is_dir() else src
    out = Path(out_path) if out_path else npz.with_name("result_small.npz")
    d = dict(np.load(npz, allow_pickle=True))
    size_in = npz.stat().st_size

    keep = {}
    for k in _PROBE_KEYS + (_SMALL_KEYS if keep_diagnostics else ()):
        v = d.get(k)
        if v is None or np.asarray(v).shape == ():
            continue
        a = np.asarray(v)
        if a.ndim == 3:                     # (z, r, t)
            a = a[::z_stride, ::r_stride, ::t_stride]
        elif k in ("z",) + _SMALL_KEYS:
            a = a[::z_stride]
        elif k in ("r_sub", "rlist"):
            a = a[::r_stride]
        elif k == "t_sub_fs":
            a = a[::t_stride]
        if np.issubdtype(a.dtype, np.floating) and k not in ("z", "r", "rlist",
                                                             "r_sub", "t_sub_fs"):
            a = a.astype(dtype)
        keep[k] = a

    np.savez_compressed(out, **keep)
    size_out = out.stat().st_size
    if verbose:
        print(f"{npz.name} : {size_in/1e6:8.1f} Mo")
        print(f"{out.name} : {size_out/1e6:8.1f} Mo   "
              f"(facteur {size_in/max(size_out,1):.0f})")
        t = keep.get("t_sub_fs")
        if t is not None and len(t) > 1:
            print(f"  resolution en delai conservee : {np.mean(np.diff(t)):.1f} fs")
        print(f"  cubes conserves : "
              f"{[k for k in keep if np.asarray(keep[k]).ndim == 3]}")
        print(f"  /!\\ ce fichier ne sert QU'AU post-traitement sonde ; les "
              f"diagnostics de propagation (fluence, Ipeak, pertes resolues) "
              f"n'y sont pas.")
    return out


def export_probe_curve(res, out_path, delays_fs=None, inst=None, **kw):
    """N'exporte que la courbe delta_phi(tau) et la transmittance : quelques ko.

    C'est le format a envoyer quand meme un npz reduit est de trop.
    """
    from virtual_experiment import sample_as_experiment, NOMARSKI_515
    pts = sample_as_experiment(res, delays_fs, inst or NOMARSKI_515, **kw)
    out = Path(out_path)
    np.savez_compressed(out,
                        delays_fs=pts["delays_fs"],
                        phase_rad=pts["phase_rad"],
                        opl_nm=pts["opl_nm"],
                        transmittance=pts["transmittance"],
                        z_um=pts["z_um"], x_um=pts["x_um"])
    # miroir texte, pour pouvoir le lire sans numpy
    csv = out.with_suffix(".csv")
    np.savetxt(csv, np.column_stack([pts["delays_fs"], pts["phase_rad"],
                                     pts["opl_nm"], pts["transmittance"]]),
               header="delay_fs,phase_rad,opl_nm,transmittance",
               delimiter=",", comments="")
    print(f"-> {out.name} ({out.stat().st_size/1e3:.1f} ko) et {csv.name}")
    return out


# ================================================================================
#  2. Extraire la courbe MESUREE de la pile d'images
# ================================================================================
def _default_delay_from_name(name):
    """Recupere le delai depuis le nom de fichier.

    Reconnait les deux conventions rencontrees dans le depot :
      '4.0uJ_+12pulse_515nm.npz'   -> 12 impulsions  (convention historique)
      '..._+1.901ps...'            -> 1901 fs
      '..._frame078...'            -> indice de frame, converti par fs_per_step
    Renvoie (valeur, unite) avec unite dans {'pulse', 'fs', 'frame'}.
    """
    m = re.search(r"([+-]?\d+(?:\.\d+)?)\s*ps", name)
    if m:
        return float(m.group(1)) * 1000.0, "fs"
    m = re.search(r"([+-]?\d+(?:\.\d+)?)\s*fs", name)
    if m:
        return float(m.group(1)), "fs"
    m = re.search(r"([+-]?\d+(?:\.\d+)?)\s*pulse", name)
    if m:
        return float(m.group(1)), "pulse"
    m = re.search(r"(?:frame|_)(\d{2,4})(?:\.|_|$)", name)
    if m:
        return float(m.group(1)), "frame"
    return None, None


def measured_phase_curve(frames_dir, pattern="*.npz", x_um=0.0, z_um=None,
                         average_x_um=None, average_z_um=None,
                         fs_per_step=67.0, delay_from_name=None,
                         lmd_nm=515.0, z_origin_px=None, verbose=True):
    """delta_phi(tau) mesure, extrait d'une pile de frames deja enregistree.

    Une courbe temporelle n'est PAS une acquisition separee : c'est une coupe
    le long de l'axe des delais dans les frames qui ont deja servi a faire les
    planches. Cette fonction rejoue, pour chaque frame, le pretraitement
    interferometrique de web/abel_phase_explorer.py (Takeda + rotation +
    soustraction du plan de fond), puis lit la phase en un point ou sur une
    petite fenetre.

    `frames_dir` doit contenir les npz bruts (cles `side_sig` / `side_bg`).
    `delay_from_name` : fonction nom -> delai en fs, si la convention de
    nommage n'est pas reconnue automatiquement.

    Renvoie dict(delays_fs, phase_rad, files) pret pour
    virtual_experiment.plot_experiment_overlay(..., measured=...).
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "web"))
    from abel_phase_explorer import preprocess_side_full

    files = sorted(Path(frames_dir).glob(pattern))
    if not files:
        raise FileNotFoundError(f"aucun fichier {pattern} dans {frames_dir}")

    delays, phases, used = [], [], []
    for f in files:
        if delay_from_name is not None:
            tau = float(delay_from_name(f.name))
        else:
            val, unit = _default_delay_from_name(f.name)
            if val is None:
                if verbose:
                    print(f"  delai illisible dans {f.name}, ignore")
                continue
            tau = val * fs_per_step if unit in ("pulse", "frame") else val
        try:
            phase, s_um_per_px, _ = preprocess_side_full(str(f), lmd_nm)
        except Exception as exc:
            if verbose:
                print(f"  {f.name} : {type(exc).__name__} {exc}")
            continue

        ny, nx = phase.shape
        cy = ny // 2 if z_origin_px is None else int(z_origin_px)
        # axe 0 = r (transverse), axe 1 = z, comme les planches de cote
        r_ax = (np.arange(ny) - cy) * s_um_per_px
        z_ax = np.arange(nx) * s_um_per_px
        iz = (int(np.argmax(np.abs(phase).max(axis=0))) if z_um is None
              else int(np.argmin(np.abs(z_ax - z_um))))
        mr = (np.abs(r_ax - x_um) <= average_x_um) if average_x_um else \
             (np.arange(ny) == int(np.argmin(np.abs(r_ax - x_um))))
        mz = (np.abs(z_ax - z_ax[iz]) <= average_z_um) if average_z_um else \
             (np.arange(nx) == iz)
        delays.append(tau)
        phases.append(float(np.nanmean(phase[np.ix_(mr, mz)])))
        used.append(f.name)

    o = np.argsort(delays)
    out = dict(delays_fs=np.array(delays)[o], phase_rad=np.array(phases)[o],
               files=[used[i] for i in o])
    if verbose:
        print(f"{len(out['delays_fs'])} frames exploitees, delais de "
              f"{out['delays_fs'][0]:.0f} a {out['delays_fs'][-1]:.0f} fs")
        print(f"  phase mesuree : min {out['phase_rad'].min():+.3f}, "
              f"max {out['phase_rad'].max():+.3f} rad")
    return out


def measured_curve_from_maps(maps, delays_fs, x_index, z_index,
                             average=0, verbose=True):
    """Meme coupe, mais depuis des cartes de phase DEJA calculees.

    `maps` : tableau (n_delais, n_r, n_z) ou liste de cartes 2D. A utiliser si
    le pretraitement a deja tourne ailleurs et que seules les cartes ont ete
    conservees.
    """
    arr = np.asarray(maps, float)
    a = max(int(average), 0)
    vals = []
    for m in arr:
        sl = m[max(0, x_index - a):x_index + a + 1,
                max(0, z_index - a):z_index + a + 1]
        vals.append(float(np.nanmean(sl)))
    out = dict(delays_fs=np.asarray(delays_fs, float), phase_rad=np.array(vals))
    if verbose:
        print(f"{len(vals)} points, phase de {out['phase_rad'].min():+.3f} a "
              f"{out['phase_rad'].max():+.3f} rad")
    return out
