"""
Ajustement gaussien d'une caustique mesurée (`.github/beam profile to fit/`)
pour vérifier le `w0` utilisé dans la simulation.

Les fichiers sont `beamprofile_<z>.npy`, images 200x200 float32 d'une caméra
10 bits (0-1023, offset ~150-280 selon la frame).

CE QUE CE MODULE PEUT ET NE PEUT PAS DONNER
-------------------------------------------
Il mesure, sans aucune calibration :
    w0 [px], z0 [unités des noms de fichier], zR [mêmes unités]
Le rapport zR/w0[px] est l'invariant qui contraint tout le reste.

Il ne peut PAS donner w0 en µm sans deux informations extérieures :
    p = taille de pixel effective [µm/px] = pixel caméra / grandissement
    s = échelle de l'axe z          [µm par unité de nom de fichier]
liées par, pour un faisceau gaussien de facteur M² :
    p / sqrt(s) = sqrt(zR_fichier * lambda / pi) / w0_px * sqrt(M²)
Avec la contrainte physique M² >= 1, toute combinaison donnant M² < 1 est
exclue. Donner p OU s (et M²=1) suffit à fixer l'autre.
"""

import glob
import os
import re

import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter

__all__ = ["load_profiles", "fit_gaussian_2d", "measure_caustic",
           "fit_caustic", "caustic_asymmetry", "calibration_table", "plot_caustic",
           "peak_intensity_ratio", "power_consistency", "w0_from_pixel_size"]

SAT_LEVEL = 1023.0          # caméra 10 bits


def load_profiles(folder):
    """[(z, image), ...] trié par z croissant, z lu dans le nom de fichier."""
    out = []
    for f in glob.glob(os.path.join(folder, "*.npy")):
        m = re.search(r"(-?\d+)", os.path.basename(f))
        if m:
            out.append((int(m.group(1)), np.load(f).astype(float), os.path.basename(f)))
    return sorted(out, key=lambda t: t[0])


def _g2(xy, A, x0, y0, sx, sy, off):
    x, y = xy
    return (A * np.exp(-2 * ((x - x0)**2 / sx**2 + (y - y0)**2 / sy**2)) + off).ravel()


def fit_gaussian_2d(a, half=45, bg_corner=40, smooth=2.0):
    """Gaussienne 2D + offset sur une fenêtre centrée sur le pic lissé.

    La fenêtre locale est importante : ces images ont une bande horizontale
    plus sombre sur le quart supérieur et un striage vertical, qui font
    diverger un ajustement plein champ. Le fond est pris dans le coin
    bas-gauche pour éviter cette bande.

    Renvoie dict(wx, wy, w, x0, y0, A, r2, sat).
    """
    sat = int((a >= SAT_LEVEL).sum())
    sm = gaussian_filter(a, smooth)
    iy, ix = np.unravel_index(np.argmax(sm), sm.shape)
    bg = np.median(a[-bg_corner:, :bg_corner])

    i0, i1 = max(0, iy - half), min(a.shape[0], iy + half)
    j0, j1 = max(0, ix - half), min(a.shape[1], ix + half)
    sub = a[i0:i1, j0:j1] - bg
    ys, xs = np.indices(sub.shape)

    # amorce par les moments pondérés en intensité^2 : écrase le fond résiduel
    wgt = np.clip(sub, 0, None)**2
    tot = wgt.sum()
    mx, my = (wgt * xs).sum() / tot, (wgt * ys).sum() / tot
    sxg = float(np.clip(2 * np.sqrt((wgt * (xs - mx)**2).sum() / tot), 2, 40))
    syg = float(np.clip(2 * np.sqrt((wgt * (ys - my)**2).sum() / tot), 2, 40))

    p, _ = curve_fit(_g2, (xs, ys), sub.ravel(),
                     p0=(sub.max(), mx, my, sxg, syg, 0.0),
                     bounds=([0, 0, 0, 1, 1, -500],
                             [3000, sub.shape[1], sub.shape[0], 80, 80, 500]),
                     maxfev=40000)
    A, x0, y0, sx, sy, off = p
    r2 = 1 - ((sub.ravel() - _g2((xs, ys), *p))**2).sum() / ((sub.ravel() - sub.mean())**2).sum()
    return dict(wx=abs(sx), wy=abs(sy), w=float(np.sqrt(abs(sx) * abs(sy))),
                x0=x0 + j0, y0=y0 + i0, A=A, r2=r2, sat=sat)


def measure_caustic(folder, verbose=True):
    """Ajuste chaque image. Renvoie un tableau structuré (z, wx, wy, w, r2, sat)."""
    rows = []
    if verbose:
        print(f"{'z':>8s} {'wx(px)':>7s} {'wy(px)':>7s} {'w(px)':>7s} {'R2':>6s} {'sat':>5s}  note")
    for z, a, name in load_profiles(folder):
        try:
            f = fit_gaussian_2d(a)
        except Exception as exc:
            if verbose:
                print(f"{z:8d}  ECHEC {type(exc).__name__}: {exc}")
            continue
        note = "SATUREE -> a exclure" if f["sat"] else ("anneaux (R2 bas)" if f["r2"] < 0.85 else "")
        rows.append((z, f["wx"], f["wy"], f["w"], f["A"], f["r2"], f["sat"]))
        if verbose:
            print(f"{z:8d} {f['wx']:7.2f} {f['wy']:7.2f} {f['w']:7.2f} "
                  f"{f['r2']:6.3f} {f['sat']:5d}  {note}")
    return np.array(rows, dtype=[("z", "f8"), ("wx", "f8"), ("wy", "f8"),
                                 ("w", "f8"), ("A", "f8"), ("r2", "f8"), ("sat", "f8")])


def _caustic(z, w0, z0, zR):
    return w0 * np.sqrt(1 + ((z - z0) / zR)**2)


def fit_caustic(caus, exclude_saturated=True, z_window=None, mask=None, verbose=True):
    """Ajuste w(z) = w0 sqrt(1+((z-z0)/zR)^2), en unités brutes (px et unités
    de nom de fichier). z_window : demi-largeur autour du minimum, pour se
    restreindre à la zone où la parabole est valide."""
    sel = np.ones(len(caus), bool)
    if exclude_saturated:
        sel &= caus["sat"] == 0
    if mask is not None:
        sel &= mask
    if z_window is not None:
        zmin = caus["z"][np.argmin(np.where(caus["sat"] > 0, np.inf, caus["w"]))]
        sel &= np.abs(caus["z"] - zmin) <= z_window
    z, w = caus["z"][sel], caus["w"][sel]
    if len(z) < 4:
        raise ValueError(f"seulement {len(z)} points retenus")
    p, c = curve_fit(_caustic, z, w, p0=(w.min(), z[np.argmin(w)], (z.max() - z.min()) / 2),
                     maxfev=40000)
    e = np.sqrt(np.diag(c))
    out = dict(w0_px=p[0], z0=p[1], zR=abs(p[2]),
               w0_err=e[0], z0_err=e[1], zR_err=e[2], n=len(z),
               rms=float(np.sqrt(np.mean((w - _caustic(z, *p))**2))))
    if verbose:
        print(f"n={out['n']}  w0={out['w0_px']:.2f}±{out['w0_err']:.2f} px  "
              f"z0={out['z0']:.1f}±{out['z0_err']:.1f}  zR={out['zR']:.1f}±{out['zR_err']:.1f}  "
              f"rms={out['rms']:.2f} px")
        print(f"  invariant zR/w0 = {out['zR']/out['w0_px']:.1f}")
    return out


def caustic_asymmetry(caus, offsets=(300, 400, 500, 600, 800), valid=None, verbose=True):
    """Compare w²-w0² à ±dz du minimum. Une caustique gaussienne donne 1.00.
    C'est le test qui dit si un modèle gaussien unique est légitime."""
    ok = caus["sat"] == 0 if valid is None else valid
    z, w = caus["z"][ok], caus["w"][ok]
    i = int(np.argmin(w))
    out = {}
    if verbose:
        print(f"minimum a z={z[i]:.0f}, w={w[i]:.2f} px  (1.00 = gaussienne parfaite)")
    for dz in offsets:
        L = np.isclose(z, z[i] - dz)
        R = np.isclose(z, z[i] + dz)
        if L.any() and R.any():
            l = w[L][0]**2 - w[i]**2
            r = w[R][0]**2 - w[i]**2
            out[dz] = r / l if l > 0 else np.nan
            if verbose:
                print(f"  dz=±{dz:4d} : avant {l:8.1f}  apres {r:8.1f}  rapport {out[dz]:6.2f}")
    return out


def calibration_table(fit, wavelength_um=1.030, pixel_candidates=(0.345, 1.0, 3.45),
                      s_candidates=(1.0, 0.5, 0.1, 0.05, 0.01)):
    """Lève la dégénérescence (p, s) en tabulant M² = pi (w0_px p)²/(lambda zR s).
    M² < 1 est non physique -> exclut la combinaison."""
    w0px, zRf = fit["w0_px"], fit["zR"]
    print(f"w0 = {w0px:.2f} px, zR = {zRf:.1f} (unites de nom de fichier), "
          f"lambda = {wavelength_um*1000:.0f} nm")
    print(f"Relation gaussienne : p/sqrt(s) = {np.sqrt(zRf*wavelength_um/np.pi)/w0px:.4f} * sqrt(M2)  [um/px]\n")
    hdr = f"{'s (um/unite)':>13s} {'zR reel(um)':>12s} {'w0 si M2=1':>11s} {'p si M2=1':>10s} | "
    hdr += " ".join(f"{'M2(p=' + str(p) + ')':>13s}" for p in pixel_candidates)
    print(hdr)
    for s in s_candidates:
        zR = zRf * s
        w0_M1 = np.sqrt(zR * wavelength_um / np.pi)
        row = f"{s:13.4f} {zR:12.1f} {w0_M1:11.2f} {w0_M1/w0px:10.3f} | "
        row += " ".join(f"{np.pi*(w0px*p)**2/(wavelength_um*zR):13.3f}" for p in pixel_candidates)
        print(row)
    print("\nM2 < 1 impossible -> ces cases excluent la combinaison (p, s).")


def plot_caustic(caus, fit=None, valid=None, pixel_um=None, s_um_per_unit=None,
                 fit_window=None, save=None):
    """w(z) mesurée + ajustement, en distinguant les trois catégories d'images.

    `valid` (sortie de power_consistency) sépare les images du plateau de
    celles écartées pour cause de changement de densité optique : les tracer
    du même symbole donnerait à croire que l'ajustement ignore des points
    valables, alors qu'ils sont rejetés pour une raison mesurée.
    """
    import matplotlib.pyplot as plt
    p = pixel_um or 1.0
    s = s_um_per_unit or 1.0
    sat = caus["sat"] > 0
    good = (~sat) if valid is None else valid
    rejected = (~sat) & (~good)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.plot(caus["z"][good] * s, caus["w"][good] * p, "o", color="tab:blue", ms=7,
            label=f"retenues ({good.sum()}) -- puissance coherente")
    if rejected.any():
        ax.plot(caus["z"][rejected] * s, caus["w"][rejected] * p, "s", mfc="none",
                mec="darkorange", ms=8, mew=1.6,
                label=f"ecartees ({rejected.sum()}) -- densite optique differente")
    if sat.any():
        ax.plot(caus["z"][sat] * s, caus["w"][sat] * p, "x", color="crimson", ms=10, mew=2,
                label="saturee")
    if fit:
        zz = np.linspace(caus["z"].min(), caus["z"].max(), 400)
        ax.plot(zz * s, _caustic(zz, fit["w0_px"], fit["z0"], fit["zR"]) * p, "-",
                color="black", lw=1.5,
                label=(f"gaussienne : w0={fit['w0_px']*p:.2f}"
                       f"{' µm' if pixel_um else ' px'}, zR={fit['zR']*s:.0f}"))
        ax.axvline(fit["z0"] * s, ls=":", color="gray", lw=1)
        if fit_window:
            ax.axvspan((fit["z0"] - fit_window) * s, (fit["z0"] + fit_window) * s,
                       color="tab:blue", alpha=0.07, label="zone d'ajustement")
    unit = "µm" if (pixel_um and s_um_per_unit) else "unites brutes"
    ax.set_xlabel(f"z ({unit})")
    ax.set_ylabel(f"w, rayon 1/e2 ({'µm' if pixel_um else 'px'})")
    ax.set_title("Caustique mesuree vs modele gaussien")
    ax.legend(fontsize=8, loc="upper center")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
    return fig


def peak_intensity_ratio(folder, z_values, radii=(30, 40, 50, 60, 70),
                         bg_radius=60, verbose=True):
    """Rapport A_eff/A_gauss = (P/I_pic) / (pi w^2/2) près du foyer.

    SANS DIMENSION : ne demande ni taille de pixel ni échelle en z. C'est le
    facteur par lequel le solveur SURESTIME l'intensité crête, puisqu'il pose
    I0 = 2P/(pi w0^2), c'est-à-dire A_eff = pi w0^2/2 exactement.

    Le fond est un PLAN ajusté hors du faisceau (r > bg_radius), pas une
    médiane locale : ces images ont un gradient et un banding qui font diverger
    l'intégrale avec le rayon d'ouverture si on les ignore (le rapport passait
    de 1.29 à 2.27 entre r<30 et r<50 ; avec le plan, 1.58 à 2.07).

    La dépendance résiduelle au rayon est réelle (ailes/anneaux) et doit être
    rapportée comme une plage, pas comme un nombre unique.
    """
    out = {R: [] for R in radii}
    for z in z_values:
        a = np.load(os.path.join(folder, f"beamprofile_{z}.npy")).astype(float)
        sm = gaussian_filter(a, 2.0)
        iy, ix = np.unravel_index(np.argmax(sm), sm.shape)
        yy, xx = np.indices(a.shape)
        rr = np.hypot(xx - ix, yy - iy)
        m = rr > bg_radius
        M = np.c_[xx[m].ravel(), yy[m].ravel(), np.ones(m.sum())]
        coef, *_ = np.linalg.lstsq(M, a[m].ravel(), rcond=None)
        b = a - (coef[0] * xx + coef[1] * yy + coef[2])

        h = 45
        sub = b[iy - h:iy + h, ix - h:ix + h]
        ys, xs = np.indices(sub.shape)
        p, _ = curve_fit(_g2, (xs, ys), sub.ravel(),
                         p0=(sub.max(), h, h, 8, 8, 0),
                         bounds=([0, 0, 0, 1, 1, -500], [3000, 2 * h, 2 * h, 80, 80, 500]),
                         maxfev=40000)
        w2 = abs(p[3]) * abs(p[4])
        Ipk = gaussian_filter(b, 1.0).max()
        for R in radii:
            out[R].append((b[rr < R].sum() / Ipk) / (np.pi * w2 / 2))

    means = {R: float(np.mean(v)) for R, v in out.items()}
    lo, hi = min(means.values()), max(means.values())
    if verbose:
        print(f"{'rayon':>8s} {'A_eff/A_gauss':>14s} {'I_pic/I_gauss':>14s}")
        for R in radii:
            print(f"  r<{R:3d} {means[R]:12.2f} +/-{np.std(out[R]):.2f} {1/means[R]:14.2f}")
        print(f"\n-> le solveur surestime l'intensite crete d'un facteur {lo:.1f} a {hi:.1f}")
        print(f"   w0 effectif qui reproduirait la vraie intensite crete :")
        print(f"     w0_eff = w0_nominal x {np.sqrt(lo):.2f} a {np.sqrt(hi):.2f}")
        print("   ATTENTION : gonfler w0 corrige l'intensite mais degrade la caustique")
        print("   (zR ~ w0^2). Une seule gaussienne ne peut pas faire les deux.")
    return means


def power_consistency(caus, folder, plateau=None, tol=0.20, verbose=True):
    """Contrôle qualité : A*w² (puissance totale de la gaussienne ajustée) doit
    être constant d'une image à l'autre SI la densité optique et le temps
    d'exposition n'ont pas changé.

    Indispensable ici : ces caustiques ont été prises avec un filtre de densité
    ajusté « plus ou moins » pour éviter la saturation, donc l'atténuation n'est
    pas la même partout. Une frame sous-atténuée ou bruitée gonfle à la fois sa
    largeur ajustée et sa puissance apparente -- et fausse alors le test de
    symétrie, qui compare justement des largeurs de part et d'autre du foyer.

    Une atténuation uniforme ne change NI la largeur ajustée NI le rapport
    A_eff/A_gauss (les deux sont invariants d'échelle) : seul un CHANGEMENT
    d'atténuation entre images pose problème.

    Renvoie un masque booléen des images cohérentes avec le plateau.
    """
    P = caus["A"] * caus["w"]**2 if "A" in caus.dtype.names else None
    if P is None:
        raise ValueError("caus doit contenir la colonne 'A' (voir measure_caustic)")
    if plateau is None:
        ref = float(np.median(P[caus["sat"] == 0]))
    else:
        m = (caus["z"] >= plateau[0]) & (caus["z"] <= plateau[1])
        ref = float(np.median(P[m]))
    rel = P / ref
    good = (np.abs(rel - 1) <= tol) & (caus["sat"] == 0)
    if verbose:
        print(f"reference A*w^2 = {ref:.3e}")
        print(f"{'z':>8s} {'A*w^2/ref':>10s}  verdict")
        for z, r, g, s in zip(caus["z"], rel, good, caus["sat"]):
            v = "SATUREE" if s else ("ok" if g else f"ECARTEE ({(r-1)*100:+.0f} %)")
            print(f"{z:8.0f} {r:10.2f}  {v}")
        print(f"\n{good.sum()}/{len(good)} images cohérentes a +/-{tol*100:.0f} %")
    return good


def w0_from_pixel_size(fit, pixel_um, NA=None, wavelength_um=1.030,
                       s_um_per_unit=None, w0_sim_um=None, verbose=True):
    """w0 physique à partir du seul pixel effectif.

        w0 [µm] = w0_px * p

    C'est tout : ni l'échelle de l'axe z ni la longueur d'onde n'entrent ici.
    Elles ne servent qu'à en déduire M² (si s est fourni).

    Si NA est donné, corrige la PSF de l'imageur par déconvolution quadratique
    w_vrai² = w_mesuré² − w_psf², avec w_psf ≈ 0.42 λ/NA : à NA=0.28 et
    1030 nm la PSF fait 1.5 µm, ce qui gonfle de 13 % une mesure de 2.9 µm et
    de 3 % seulement une mesure de 5.8 µm — négligeable dans un cas, pas dans
    l'autre.
    """
    w0_meas = fit["w0_px"] * pixel_um
    out = dict(pixel_um=pixel_um, w0_measured_um=w0_meas, w0_um=w0_meas)
    if NA:
        w_psf = 0.42 * wavelength_um / NA
        out["w_psf_um"] = w_psf
        out["w0_um"] = float(np.sqrt(max(w0_meas**2 - w_psf**2, 0.0)))
    if s_um_per_unit:
        zR = fit["zR"] * s_um_per_unit
        out["zR_um"] = zR
        out["M2"] = float(np.pi * out["w0_um"]**2 / (wavelength_um * zR))
    if verbose:
        print(f"p = {pixel_um:.4f} µm/px  ->  w0 mesure = {w0_meas:.2f} µm")
        if NA:
            print(f"  PSF (NA={NA}, {wavelength_um*1000:.0f} nm) = {out['w_psf_um']:.2f} µm"
                  f"  ->  w0 deconvolue = {out['w0_um']:.2f} µm")
        if s_um_per_unit:
            print(f"  zR = {out['zR_um']:.1f} µm  ->  M2 = {out['M2']:.2f}"
                  + ("   /!\\ M2 < 1 : NON PHYSIQUE, une hypothese est fausse"
                     if out["M2"] < 1 else ""))
        if w0_sim_um:
            f_w = (out["w0_um"] / w0_sim_um)**2
            print(f"\n  simulation : w0 = {w0_sim_um:.2f} µm  ->  facteur {out['w0_um']/w0_sim_um:.2f} sur w0")
            print(f"  intensite crete surestimee de {f_w:.2f}x par ce seul effet")
            print(f"  a combiner avec le facteur 1.6-2.4 du profil non gaussien")
            print(f"  -> surestimation totale : {f_w*1.6:.1f} a {f_w*2.4:.1f}x")
    return out
