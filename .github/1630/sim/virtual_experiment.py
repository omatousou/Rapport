"""Echantillonner la simulation COMME LA MANIP, pour superposer les deux.

Une carte de simulation et une carte mesuree ne sont pas comparables telles
quelles : la mesure passe par une chaine d'instrument qui degrade et discretise
tout. Ce module applique cette chaine a la simulation, de sorte que ce qui en
sort ait le meme statut qu'un point de mesure -- et puisse etre superpose en
nuage de points sur la courbe modele, comme la Fig. 6 de Martin et al. 1997
superpose ses points a son ajustement.

Ce que la chaine applique, dans l'ordre :

1. PROJECTION. La sonde ne voit pas Delta n(r) mais son integrale sur la corde
   (transformee d'Abel directe). Deja fait par probe_opl_transmittance.
2. REPONSE TEMPORELLE. La sonde a une duree finie : le delai nominal tau ne
   donne pas un instantane mais une moyenne de l'etat du milieu ponderee par
   l'enveloppe de la sonde. C'est exactement ce que l'article signale comme
   invalide entre 0 et 200 fs, la ou le signal bascule le plus vite.
3. RESOLUTION OPTIQUE. Filtre passe-bas a la frequence de coupure NA/lambda :
   toute structure plus fine que lambda/NA est perdue.
4. ECHANTILLONNAGE. Les delais reellement balayes par la ligne a retard, pas
   une grille continue.
5. BRUIT. Au niveau du plancher de la mesure.

Sans les etapes 2 a 5, comparer simulation et mesure revient a reprocher au
modele de ne pas ressembler a un instrument.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

__all__ = [
    "Instrument", "NOMARSKI_515",
    "experimental_delay_grid", "sample_as_experiment",
    "plot_experiment_overlay", "compare_to_measurement",
]


@dataclass
class Instrument:
    """Reponse de l'instrument de mesure."""
    name: str = "Nomarski, sonde 515 nm"
    lambda_probe_m: float = 515e-9
    NA: float = 0.23
    probe_fwhm_fs: float = 263.0        # duree de l'impulsion sonde
    pixel_um: float = 0.345             # taille de pixel ramenee a l'objet
    noise_opl_nm: float = 0.5           # plancher de bruit, en nm d'OPL
    noise_transmittance: float = 0.02

    @property
    def resolution_um(self):
        """Critere d'Abbe : la plus petite structure transmise par l'optique."""
        return self.lambda_probe_m * 1e6 / self.NA


NOMARSKI_515 = Instrument()


def experimental_delay_grid(fine=(-1000.0, 2000.0, 33.0),
                            coarse_ps=(3, 5, 10, 20, 50, 100, 200, 500,
                                       1000, 2000, 6000)):
    """Grille de delais de la manip : balayage fin puis points espaces.

    `fine` = (debut_fs, fin_fs, pas_fs). Les points grossiers sont donnes en ps.
    """
    t0, t1, dt = fine
    g = list(np.arange(t0, t1 + 0.5 * dt, dt))
    g += [p * 1000.0 for p in coarse_ps if p * 1000.0 > t1]
    return np.array(sorted(set(g)), float)


# ================================================================================
#  Filtre optique
# ================================================================================
def lowpass_NA(field, d0_um, d1_um, NA, lambda_um):
    """Passe-bas isotrope a k_max = 2 pi NA / lambda, sur une carte 2D."""
    f = np.asarray(field, float)
    bad = ~np.isfinite(f)
    g = np.where(bad, 0.0, f)
    k0 = 2 * np.pi * np.fft.fftfreq(g.shape[0], d=d0_um)
    k1 = 2 * np.pi * np.fft.fftfreq(g.shape[1], d=d1_um)
    K0, K1 = np.meshgrid(k0, k1, indexing="ij")
    mask = (K0**2 + K1**2) <= (2 * np.pi * NA / lambda_um) ** 2
    out = np.real(np.fft.ifft2(np.fft.fft2(g) * mask))
    out[bad] = np.nan
    return out


# ================================================================================
#  Echantillonnage
# ================================================================================
def sample_as_experiment(res, delays_fs=None, inst: Instrument = NOMARSKI_515,
                         z_um=None, x_um=0.0, average_x_um=None,
                         average_z_um=None, apply_na=True, add_noise=True,
                         seed=0, convolve_probe=True, x_half_um=70.0,
                         oversample=None, **probe_kw):
    """Renvoie un jeu de points (delai, dephasage, transmittance) simule.

    `z_um`   plan de lecture ; None = le plan ou |OPL| est maximal.
    `x_um`   position transverse ; 0 = sur l'axe.
    `average_x_um` / `average_z_um` : demi-largeurs de la fenetre sur laquelle
             moyenner, pour imiter une lecture integree plutot que ponctuelle.
    `convolve_probe` : convoluer par l'enveloppe temporelle de la sonde.
             Le pas de la grille interne est choisi automatiquement (tp/8),
             il n'y a plus de nombre de sous-echantillons a regler.
             `oversample` est conserve pour compatibilite : toute valeur <= 1
             desactive la convolution.

    Les grandeurs renvoyees ont le statut d'une MESURE : projetees, lissees par
    l'optique, moyennees par la duree de sonde, echantillonnees et bruitees.
    """
    from figures_filament import probe_opl_transmittance

    if delays_fs is None:
        delays_fs = experimental_delay_grid()
    delays_fs = np.atleast_1d(np.asarray(delays_fs, float))

    # PROBE_KW du notebook contient deja lambda_probe_m ; l'instrument aussi.
    # On garde celui de l'instrument et on signale un desaccord plutot que de
    # lever "multiple values for keyword argument".
    lam_kw = probe_kw.pop("lambda_probe_m", None)
    if lam_kw is not None and abs(lam_kw - inst.lambda_probe_m) > 1e-12:
        print(f"  (!) lambda_probe_m={lam_kw*1e9:.0f} nm passe en argument, mais "
              f"l'instrument est a {inst.lambda_probe_m*1e9:.0f} nm : "
              f"c'est celle de l'instrument qui est utilisee.")

    # --- convolution par l'enveloppe de la sonde ------------------------
    # ANCIENNE VERSION, FAUSSE : pour chaque delai demande on evaluait
    # `oversample` sous-delais repartis sur +/-1.5 tp et on en faisait la
    # moyenne ponderee. Avec oversample=5 et tp=223 fs, les sous-delais sont
    # espaces de 168 fs : ce n'est pas un noyau gaussien, c'est un PEIGNE de
    # cinq dents. Chaque dent balaie le pic de Kerr et en produit une replique,
    # d'ou les bosses parasites espacees de ~170 fs sur le front montant.
    #
    # VERSION CORRECTE : on evalue la reponse BRUTE une seule fois sur une
    # grille interne fine et REGULIERE, on convolue proprement, puis on
    # interpole aux delais demandes. C'est aussi moins cher : le nombre
    # d'evaluations ne depend plus du nombre de points demandes.
    tp = inst.probe_fwhm_fs / np.sqrt(2.0 * np.log(2.0))
    convolve = bool(convolve_probe) and not (oversample is not None
                                             and oversample <= 1)

    t_sub = np.asarray(res["t_sub_fs"], float)
    dt_cube = float(np.mean(np.diff(t_sub))) if len(t_sub) > 1 else np.inf
    dt_scan = float(np.min(np.diff(delays_fs))) if len(delays_fs) > 1 else np.inf
    if dt_scan < dt_cube:
        print(f"  (!) pas de balayage {dt_scan:.0f} fs plus fin que la resolution "
              f"temporelle du cube ({dt_cube:.0f} fs, fixee par rho_t_stride) : "
              f"les points intermediaires n'apportent rien. Relancer avec un "
              f"rho_t_stride plus petit pour un vrai balayage fin.")

    if convolve:
        # pas interne : assez fin pour echantillonner le noyau ET le cube
        dt_fine = min(tp / 8.0, max(dt_cube, 1.0), dt_scan)
        pad = 3.0 * tp
        t_eval = np.arange(delays_fs[0] - pad, delays_fs[-1] + pad + dt_fine,
                           dt_fine)
    else:
        dt_fine = dt_scan
        t_eval = delays_fs

    rng = np.random.default_rng(seed)
    raw_opl, raw_tr, z_used = [], [], None

    for tau in t_eval:
        d = probe_opl_transmittance(res, float(tau),
                                    lambda_probe_m=inst.lambda_probe_m,
                                    x_half_um=x_half_um, **probe_kw)
        acc_opl = np.asarray(d["opl_nm"], float)
        acc_tr = np.asarray(d["transmittance"], float)
        x, z = d["x_um"], d["z_um"]

        if apply_na:
            dz = float(np.mean(np.diff(z))) if len(z) > 1 else 1.0
            dx = float(np.mean(np.diff(x)))
            lam_um = inst.lambda_probe_m * 1e6
            acc_opl = lowpass_NA(acc_opl, dz, dx, inst.NA, lam_um)
            acc_tr = lowpass_NA(acc_tr, dz, dx, inst.NA, lam_um)

        if z_used is None:
            z_used = (float(z[int(np.argmax(np.abs(acc_opl).max(axis=1)))])
                      if z_um is None else float(z_um))
        mz = (np.abs(z - z_used) <= average_z_um) if average_z_um else \
             (np.arange(len(z)) == int(np.argmin(np.abs(z - z_used))))
        mx = (np.abs(x - x_um) <= average_x_um) if average_x_um else \
             (np.arange(len(x)) == int(np.argmin(np.abs(x - x_um))))
        raw_opl.append(float(np.nanmean(acc_opl[np.ix_(mz, mx)])))
        raw_tr.append(float(np.nanmean(acc_tr[np.ix_(mz, mx)])))

    raw_opl = np.array(raw_opl)
    raw_tr = np.array(raw_tr)

    if convolve:
        half = np.arange(0.0, 3.0 * tp + dt_fine, dt_fine)
        k_t = np.concatenate((-half[:0:-1], half))
        kern = np.exp(-(k_t / tp) ** 2)
        kern /= kern.sum()
        # bords : on prolonge par la valeur extreme, le padding de 3 tp garantit
        # que la zone utile n'est pas touchee
        m = len(kern) // 2
        def _conv(y):
            yp = np.concatenate((np.full(m, y[0]), y, np.full(m, y[-1])))
            return np.convolve(yp, kern, mode="same")[m:m + len(y)]
        raw_opl, raw_tr = _conv(raw_opl), _conv(raw_tr)
        opl_pts = np.interp(delays_fs, t_eval, raw_opl)
        tr_pts = np.interp(delays_fs, t_eval, raw_tr)
    else:
        opl_pts, tr_pts = raw_opl, raw_tr

    opl = np.array(opl_pts)
    tr = np.array(tr_pts)
    if add_noise:
        opl = opl + rng.normal(0.0, inst.noise_opl_nm, opl.shape)
        tr = tr + rng.normal(0.0, inst.noise_transmittance, tr.shape)

    lam_nm = inst.lambda_probe_m * 1e9
    return dict(delays_fs=delays_fs, opl_nm=opl,
                phase_rad=2.0 * np.pi * opl / lam_nm,
                transmittance=tr, z_um=z_used, x_um=x_um,
                instrument=inst,
                opl_clean=np.array(opl_pts), tr_clean=np.array(tr))


# ================================================================================
#  Traces
# ================================================================================
def plot_experiment_overlay(sampled, model=None, measured=None, xlim_ps=(-0.5, 2.0),
                            title=None, save=None, show_transmittance=True):
    """Nuage de points simule + courbe modele + mesure, facon Fig. 6.

    `sampled`  sortie de sample_as_experiment (points, avec bruit)
    `model`    dict(delays_fs=..., phase_rad=...) trace en ligne continue
    `measured` dict(delays_fs=..., phase_rad=..., [transmittance=...]) : les
               VRAIS points de la manip, si tu les as.
    """
    import matplotlib.pyplot as plt
    n = 2 if show_transmittance else 1
    fig, ax = plt.subplots(n, 1, figsize=(9, 4.2 * n), squeeze=False, sharex=True)
    a = ax[0][0]
    a.axhline(0.0, color="0.6", lw=1)
    if model is not None:
        a.plot(np.asarray(model["delays_fs"]) * 1e-3, model["phase_rad"],
               color="k", lw=1.8, zorder=1, label="modele (continu)")
    a.plot(sampled["delays_fs"] * 1e-3, sampled["phase_rad"], "o",
           ms=3.5, color="crimson", alpha=0.8, zorder=2,
           label="simulation echantillonnee comme la mesure")
    if measured is not None:
        a.plot(np.asarray(measured["delays_fs"]) * 1e-3, measured["phase_rad"],
               "s", ms=4, mfc="none", color="royalblue", zorder=3,
               label="mesure")
    a.set_ylabel(r"$\delta\varphi$ (rad)")
    a.legend(fontsize=9)
    a.grid(alpha=0.3)
    a.set_title(title or
                f"z = {sampled['z_um']:.0f} um, x = {sampled['x_um']:.0f} um, "
                f"{sampled['instrument'].name}")

    if show_transmittance:
        b = ax[1][0]
        b.axhline(1.0, color="0.6", lw=1)
        b.plot(sampled["delays_fs"] * 1e-3, sampled["transmittance"], "o",
               ms=3.5, color="darkred", alpha=0.8, label="simulation")
        if measured is not None and "transmittance" in measured:
            b.plot(np.asarray(measured["delays_fs"]) * 1e-3,
                   measured["transmittance"], "s", ms=4, mfc="none",
                   color="royalblue", label="mesure")
        b.set_ylabel("transmittance")
        b.legend(fontsize=9)
        b.grid(alpha=0.3)
    ax[-1][0].set_xlabel("optical delay (ps)")
    for row in ax:
        row[0].set_xlim(*xlim_ps)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=160, bbox_inches="tight")
    return fig


def compare_to_measurement(sampled, measured, verbose=True):
    """Chiffre l'ecart entre la simulation echantillonnee et la mesure.

    Compare ce qui est comparable : les trois reperes de la courbe (pic, creux,
    plateau) et, si les grilles de delai se recouvrent, l'ecart quadratique
    moyen apres interpolation sur les delais mesures.
    """
    def _feat(t, y):
        i = int(np.argmax(y))
        after = np.arange(len(y)) > i
        j = int(np.argmin(np.where(after, y, np.inf))) if after.any() else i
        tail = t >= t.max() - 300.0
        return dict(peak=float(y[i]), t_peak=float(t[i]),
                    dip=float(y[j]), t_dip=float(t[j]),
                    plateau=float(np.mean(y[tail])))

    fs, fm = (_feat(sampled["delays_fs"], sampled["phase_rad"]),
              _feat(np.asarray(measured["delays_fs"], float),
                    np.asarray(measured["phase_rad"], float)))
    y_i = np.interp(measured["delays_fs"], sampled["delays_fs"],
                    sampled["phase_rad"])
    resid = np.asarray(measured["phase_rad"], float) - y_i
    rms = float(np.sqrt(np.mean(resid**2)))
    span = float(np.ptp(measured["phase_rad"]))
    out = dict(sim=fs, meas=fm, rms_rad=rms, rms_over_span=rms / max(span, 1e-12))

    if verbose:
        print(f"{'':10s} {'simulation':>14s} {'mesure':>14s} {'rapport':>10s}")
        for k in ("peak", "dip", "plateau"):
            s, m = fs[k], fm[k]
            r = s / m if abs(m) > 1e-12 else float("nan")
            print(f"{k:10s} {s:+14.4f} {m:+14.4f} {r:10.2f}")
        print(f"{'t_pic (fs)':10s} {fs['t_peak']:14.0f} {fm['t_peak']:14.0f}")
        print(f"{'t_creux':10s} {fs['t_dip']:14.0f} {fm['t_dip']:14.0f}")
        print(f"\nRMS residuel = {rms:.4f} rad, soit "
              f"{100*out['rms_over_span']:.1f} % de l'amplitude mesuree")
    return out
