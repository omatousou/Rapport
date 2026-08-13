"""Dephasage et absorption sonde en fonction du delai pompe-sonde, modele 0D.

Reproduit le type de resultat de Martin, Guizard, Daguzan, Petite et al.,
Phys. Rev. B 55, 5799 (1997), Fig. 6 (dephasage) et Fig. 7 (absorption) : la
signature en trois temps d'une experience pompe-sonde dans un dielectrique a
grand gap.

    1. pic POSITIF synchrone de la pompe        -> Kerr, proportionnel a I
    2. bascule NEGATIVE juste apres             -> electrons de conduction
    3. remontee vers un plateau POSITIF         -> excitons auto-piegees

C'est un calcul 0D : on fixe l'intensite de pompe au lieu de la propager. Pas
de GPU, pas de result.npz. L'interet est double :

- valider la chaine dielectrique (permittivity.py) contre une courbe publiee,
  independamment du solveur de propagation ;
- predire la meme courbe pour la configuration de la manip (pompe 1030 nm,
  sonde 515 nm) avant de la mesurer.

Ce que le modele contient, et que la lecture naive "Delta n = -rho/2 rho_c"
n'avait pas : les deux bandes STE avec leurs forces d'oscillateur et leurs
largeurs, la masse effective des porteurs libres, la deplation de la bande de
valence, et n = sqrt(eps) plutot que le developpement au premier ordre. Voir
permittivity.py.
"""

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.constants import c as c_SI, elementary_charge as q_e, hbar

from permittivity import MaterialResponse, SPM, XPM

__all__ = [
    "PumpProbe0D", "MARTIN_SIO2", "USER_SIO2_1030",
    "integrate_populations", "phase_and_absorption",
    "plot_delay_scan", "reproduce_martin_fig6", "predict_user_config",
]


# ================================================================================
#  Jeux de parametres
# ================================================================================
@dataclass
class PumpProbe0D:
    """Configuration d'une experience pompe-sonde 0D."""
    name: str
    lambda_pump_m: float
    lambda_probe_m: float
    pulse_fwhm_s: float
    probe_response_s: float          # largeur temporelle effective de la sonde
    overlap_length_m: float          # L : longueur de recouvrement pompe-sonde
    material: MaterialResponse
    n2_m2W: float
    xpm_factor: float = XPM

    # --- source d'ionisation -------------------------------------------------
    # 'multiphoton' : N0 sigma_K F^K, la forme de l'article (leur Eq. 8).
    # 'keldysh'     : le taux du depot, pour la configuration de la manip.
    ionization: str = "multiphoton"
    K: int = 4
    sigma_K: float = 2.3e-114        # cm^(2K) s^(K-1), Table II de l'article
    Ui_eV: float = 9.0
    meff_keldysh_rel: float = 0.64

    tau_trap_s: float = 150e-15      # piegeage : CB -> STE
    tau_ste_s: Optional[float] = None   # None = pas de decroissance des STE
    enable_avalanche: bool = False
    tau_c_avalanche_s: Optional[float] = None

    def photon_flux(self, I_Wcm2):
        """Flux de photons pompe [cm^-2 s^-1] pour une intensite en W/cm^2."""
        E_ph = 2.0 * np.pi * hbar * c_SI / self.lambda_pump_m      # J
        return np.asarray(I_Wcm2, float) / E_ph

    def ionization_rate(self, I_Wcm2):
        """Taux de creation de paires [cm^-3 s^-1], hors deplation."""
        if self.ionization == "multiphoton":
            F = self.photon_flux(I_Wcm2)
            return self.material.N0_cm3 * self.sigma_K * F**self.K
        if self.ionization == "keldysh":
            from keldysh import KeldyshSiO2, n_sellmeier
            k = KeldyshSiO2(self.lambda_pump_m, self.Ui_eV,
                            self.meff_keldysh_rel * 9.1093837015e-31,
                            float(n_sellmeier(self.lambda_pump_m)))
            r = np.nan_to_num(k.rate(np.clip(np.asarray(I_Wcm2, float), 1e6, None)))
            return np.maximum(r, 0.0) * 1e6        # m^-3 s^-1 -> cm^-3 s^-1
        raise ValueError(f"ionization inconnue : {self.ionization}")


# --- SiO2 de l'article : Table II, pompe 395 nm, sonde 618 nm ----------------
MARTIN_SIO2 = PumpProbe0D(
    name="Martin et al. 1997, SiO2, pompe 395 nm / sonde 618 nm",
    lambda_pump_m=395e-9,
    lambda_probe_m=618e-9,
    pulse_fwhm_s=120e-15,
    probe_response_s=150e-15,
    overlap_length_m=300e-6,
    material=MaterialResponse(
        name="SiO2, Table II de Martin et al. 1997",
        N0_cm3=2.2e22, meff_rel=0.5, f_CB=1.0, tau_ep_s=1.0 / 1.5e15,
        ste_bands=((5.2, 0.40, 1.5), (4.2, 0.15, 1.0)),
    ),
    n2_m2W=2.0e-20,                  # 2e-16 cm^2/W, Table II
    xpm_factor=SPM,                  # leur n2 est DEJA le coefficient vu par la sonde
    ionization="multiphoton", K=4, sigma_K=2.3e-114,
    tau_trap_s=150e-15, tau_ste_s=None,
)

# --- configuration de la manip : pompe 1030 nm, sonde 515 nm -----------------
USER_SIO2_1030 = PumpProbe0D(
    name="manip Nomarski, pompe 1030 nm / sonde 515 nm",
    lambda_pump_m=1030e-9,
    lambda_probe_m=515e-9,
    pulse_fwhm_s=263e-15,
    probe_response_s=263e-15,
    overlap_length_m=60e-6,          # corde traversee, a ajuster sur la manip
    material=MaterialResponse(),     # defauts = Table II
    n2_m2W=2.74e-20,                 # n2 d'AUTO-modulation du solveur
    xpm_factor=XPM,                  # donc facteur de phase croisee explicite
    ionization="keldysh", Ui_eV=9.0, meff_keldysh_rel=0.64,
    tau_trap_s=330e-15, tau_ste_s=1e-12,
)


# ================================================================================
#  Equations de population
# ================================================================================
def integrate_populations(cfg: PumpProbe0D, I_peak_Wcm2, t_fs=None, n_t=4001):
    """Integre les equations de population sous une pompe gaussienne.

        dN_CB/dt = W(I) (1 - (N_CB+N_tr)/N0)  [+ beta I N_CB]  - N_CB/tau_trap
        dN_tr/dt = N_CB/tau_trap  [- N_tr/tau_ste]

    Le facteur de deplation borne la densite a N0 ; l'avalanche est optionnelle
    et absente du modele de l'article (leur Eq. 8 n'a qu'un terme
    multiphotonique et une decroissance exponentielle).
    """
    tp = cfg.pulse_fwhm_s / np.sqrt(2.0 * np.log(2.0))
    if t_fs is None:
        t_fs = np.linspace(-4.0 * tp * 1e15, 2500.0, n_t)
    t_s = np.asarray(t_fs, float) * 1e-15
    I = float(I_peak_Wcm2) * np.exp(-(t_s / tp) ** 2)

    beta = 0.0
    if cfg.enable_avalanche and cfg.tau_c_avalanche_s:
        from figures_report import avalanche_beta
        from keldysh import n_sellmeier
        beta, _ = avalanche_beta(cfg.lambda_pump_m, cfg.tau_c_avalanche_s,
                                 cfg.Ui_eV, 1.0,
                                 float(n_sellmeier(cfg.lambda_pump_m)))

    W = cfg.ionization_rate(I)
    N0 = cfg.material.N0_cm3
    inv_trap = 1.0 / cfg.tau_trap_s
    inv_ste = (1.0 / cfg.tau_ste_s) if cfg.tau_ste_s else 0.0

    n_cb = np.zeros_like(t_s)
    n_tr = np.zeros_like(t_s)
    for i in range(1, len(t_s)):
        dt = t_s[i] - t_s[i - 1]
        # RK2 (point milieu) : le pas est fin devant tous les temps du probleme
        def deriv(cb, tr, k):
            depl = max(0.0, 1.0 - (cb + tr) / N0)
            gen = W[k] * depl + beta * I[k] * cb * depl
            return gen - cb * inv_trap, cb * inv_trap - tr * inv_ste
        d1 = deriv(n_cb[i - 1], n_tr[i - 1], i - 1)
        mid_cb = n_cb[i - 1] + 0.5 * dt * d1[0]
        mid_tr = n_tr[i - 1] + 0.5 * dt * d1[1]
        d2 = deriv(mid_cb, mid_tr, i - 1)
        n_cb[i] = max(n_cb[i - 1] + dt * d2[0], 0.0)
        n_tr[i] = max(n_tr[i - 1] + dt * d2[1], 0.0)

    return dict(t_fs=np.asarray(t_fs, float), I_Wcm2=I, N_CB=n_cb, N_tr=n_tr)


# ================================================================================
#  Dephasage et absorption
# ================================================================================
def phase_and_absorption(cfg: PumpProbe0D, I_peak_Wcm2, t_fs=None,
                         convolve_probe=True, linearize=False,
                         spatial_average=None, n_r=24):
    """Voir la version mono-intensite plus bas ; `spatial_average` = rayon
    (en unites de w_pompe) sur lequel moyenner le signal, comme le fait
    l'article. C'est loin d'etre neutre : le Kerr suit I alors que la densite
    suit I^K, donc la moyenne spatiale attenue BEAUCOUP plus le canal plasma
    que le canal Kerr. Avec K = 4 le rapport creux/pic change d'un facteur 4."""
    if spatial_average:
        # moyenne ponderee par l'aire sur le profil transverse de pompe.
        # ATTENTION : seuls les CHAMPS dependant de l'intensite sont moyennes.
        # Ponderer aussi t_fs comprimerait l'axe des temps d'un facteur egal au
        # premier poids -- c'est le bug qui ecrasait toute la dynamique dans
        # les 150 premieres fs.
        AVG = ("phase_rad", "absorption", "N_CB", "N_tr", "I_Wcm2")
        rr = np.linspace(0.0, float(spatial_average), n_r + 1)[1:]
        acc = None
        wsum = 0.0
        for r_ in rr:
            wt = float(r_)                       # poids d'aire 2 pi r dr
            I_r = float(I_peak_Wcm2) * np.exp(-2.0 * r_**2)
            d = _phase_and_absorption_single(cfg, I_r, t_fs, convolve_probe,
                                             linearize)
            if acc is None:
                acc = {k: np.asarray(d[k], float) * wt for k in AVG}
                acc["channels"] = {k: np.asarray(v, float) * wt
                                   for k, v in d["channels"].items()}
                acc["t_fs"] = d["t_fs"]          # jamais pondere
                acc["overdense"] = np.asarray(d["overdense"])
            else:
                for k in AVG:
                    acc[k] = acc[k] + np.asarray(d[k], float) * wt
                for k in acc["channels"]:
                    acc["channels"][k] = acc["channels"][k] + d["channels"][k] * wt
                acc["overdense"] = acc["overdense"] | np.asarray(d["overdense"])
            wsum += wt
        for k in AVG:
            acc[k] = acc[k] / wsum
        for k in acc["channels"]:
            acc["channels"][k] = acc["channels"][k] / wsum
        acc["features"] = _features(acc["t_fs"], acc["phase_rad"])
        return acc
    return _phase_and_absorption_single(cfg, I_peak_Wcm2, t_fs, convolve_probe,
                                        linearize)


def _phase_and_absorption_single(cfg: PumpProbe0D, I_peak_Wcm2, t_fs=None,
                                 convolve_probe=True, linearize=False):
    """delta_phi(tau) [rad] et absorption A(tau) [0-1] vus par la sonde.

        dPhi = (2 pi / lambda) L [Re sqrt(eps_2) - n0]
        A    = 1 - exp[-(2 L omega / c) Im sqrt(eps_2)]

    exactement les Eqs. (3) et (4) de l'article. Si `convolve_probe`, les deux
    sont convoluees par l'enveloppe temporelle de la sonde : l'article
    souligne lui-meme que l'hypothese d'un indice constant pendant
    l'impulsion sonde tombe entre 0 et 200 fs, la ou le signal bascule.
    """
    from keldysh import n_sellmeier
    pop = integrate_populations(cfg, I_peak_Wcm2, t_fs)
    t_fs = pop["t_fs"]

    n0 = float(n_sellmeier(cfg.lambda_probe_m))
    mat = cfg.material
    if mat.n2_m2W != cfg.n2_m2W:
        import copy
        mat = copy.copy(mat)
        mat.n2_m2W = cfg.n2_m2W

    r = mat.response(cfg.lambda_probe_m, n0,
                     rho_e_cm3=pop["N_CB"], rho_s_cm3=pop["N_tr"],
                     I_Wcm2=pop["I_Wcm2"], xpm_factor=cfg.xpm_factor,
                     linearize=linearize)

    L_cm = cfg.overlap_length_m * 1e2
    lam_cm = cfg.lambda_probe_m * 1e2
    phi = 2.0 * np.pi / lam_cm * L_cm * np.asarray(r["dn"], float)
    absorb = 1.0 - np.exp(-np.clip(np.asarray(r["alpha_cm"], float) * L_cm, 0, None))

    # contributions separees, meme normalisation
    chan = {k: 2.0 * np.pi / lam_cm * L_cm * np.asarray(r["dn_" + k], float)
            for k in ("kerr", "drude", "ste", "depletion")}

    if convolve_probe:
        tp = cfg.probe_response_s / np.sqrt(2.0 * np.log(2.0)) * 1e15
        dt = float(np.mean(np.diff(t_fs)))
        g = np.exp(-((np.arange(-4 * tp, 4 * tp + dt, dt)) / tp) ** 2)
        g /= g.sum()
        def _c(y):
            return np.convolve(y, g, mode="same")
        phi, absorb = _c(phi), _c(absorb)
        chan = {k: _c(v) for k, v in chan.items()}

    return dict(t_fs=t_fs, phase_rad=phi, absorption=absorb, channels=chan,
                N_CB=pop["N_CB"], N_tr=pop["N_tr"], I_Wcm2=pop["I_Wcm2"],
                overdense=np.asarray(r["overdense"]),
                features=_features(t_fs, phi))


def _features(t_fs, phi):
    """Les trois reperes de la courbe : pic Kerr, creux plasma, plateau STE."""
    i_pk = int(np.argmax(phi))
    after = np.arange(len(phi)) > i_pk
    i_dip = int(np.argmin(np.where(after, phi, np.inf)))
    tail = t_fs >= t_fs[-1] - 300.0
    return dict(peak_rad=float(phi[i_pk]), peak_t_fs=float(t_fs[i_pk]),
                dip_rad=float(phi[i_dip]), dip_t_fs=float(t_fs[i_dip]),
                plateau_rad=float(np.mean(phi[tail])))


# ================================================================================
#  Traces
# ================================================================================
def plot_delay_scan(cfg: PumpProbe0D, intensities_Wcm2, labels=None,
                    show_channels=False, xlim=(-500, 2000), save=None,
                    absorption=False, spatial_average=None):
    """Une ligne par intensite de pompe, facon Fig. 6 de l'article."""
    import matplotlib.pyplot as plt
    inten = list(intensities_Wcm2)
    labels = labels or [f"{I*1e-12:g} TW/cm$^2$" for I in inten]
    nrow = len(inten)
    fig, ax = plt.subplots(nrow, 2 if absorption else 1,
                           figsize=(11 if absorption else 7, 3.2 * nrow),
                           squeeze=False, sharex=True)
    out = []
    for k, (I, lab) in enumerate(zip(inten, labels)):
        d = phase_and_absorption(cfg, I, spatial_average=spatial_average)
        out.append(d)
        a = ax[k][0]
        a.axhline(0.0, color="0.6", lw=1)
        a.plot(d["t_fs"] * 1e-3, d["phase_rad"], color="k", lw=2, label="total")
        if show_channels:
            for name, col in (("kerr", "crimson"), ("drude", "royalblue"),
                              ("ste", "darkgreen"), ("depletion", "darkorange")):
                a.plot(d["t_fs"] * 1e-3, d["channels"][name], lw=1.2, ls="--",
                       color=col, alpha=0.85, label=name)
        f = d["features"]
        a.set_ylabel("phase shift (rad)")
        a.set_title(f"{lab}   peak {f['peak_rad']:+.2f} / dip {f['dip_rad']:+.2f} "
                    f"/ plateau {f['plateau_rad']:+.2f} rad", fontsize=10)
        a.grid(alpha=0.3)
        if k == 0:
            a.legend(fontsize=8, ncol=2)
        if absorption:
            b = ax[k][1]
            b.plot(d["t_fs"] * 1e-3, d["absorption"], color="darkred", lw=2)
            b.set_ylabel("absorption")
            b.grid(alpha=0.3)
            b.set_title(f"{lab}, A max = {d['absorption'].max():.2f}", fontsize=10)
    for a in ax[-1]:
        a.set_xlabel("optical delay (ps)")
    for row in ax:
        for a in row:
            a.set_xlim(xlim[0] * 1e-3, xlim[1] * 1e-3)
    fig.suptitle(cfg.name, fontsize=11)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=160, bbox_inches="tight")
    return fig, out


def reproduce_martin_fig6(save=None, spatial_average=1.5, verbose=True):
    """Fig. 6 de l'article : SiO2, sonde 618 nm, pompe a 3 et 4 TW/cm^2.

    `spatial_average` est le rayon, en unites du waist de pompe, sur lequel le
    signal est moyenne. L'article precise que chaque point de ses courbes
    temporelles est une moyenne spatiale du profil de la Fig. 3, et cette
    moyenne n'est PAS neutre sur la comparaison : le Kerr suit I, la densite
    suit I^4, donc moyenner attenue quatre fois plus le canal plasma que le
    canal Kerr. Sans elle le creux ressort huit fois trop profond.

    Ce qui se compare vraiment, c'est le RAPPORT creux/pic : l'amplitude
    absolue depend de la longueur de recouvrement effective, que l'article ne
    donne pas (il integre le long d'un trajet incline a 10 degres a travers un
    profil gaussien, pas sur une longueur constante).
    """
    fig, res = plot_delay_scan(MARTIN_SIO2, [3e12, 4e12], show_channels=True,
                               xlim=(-500, 2000), save=save,
                               spatial_average=spatial_average)
    published = [dict(peak=0.67, dip=-0.12, plateau=0.03),
                 dict(peak=1.00, dip=-0.90, plateau=0.15)]
    if verbose:
        print(f"moyenne spatiale sur {spatial_average} w_pompe\n")
        print(f"{'':10s} {'pic (rad)':>18s} {'creux (rad)':>18s} "
              f"{'creux/pic':>18s}")
        for d, pub, lab in zip(res, published, ("3 TW/cm2", "4 TW/cm2")):
            f = d["features"]
            print(f"{lab:10s} {f['peak_rad']:+8.2f} /{pub['peak']:+6.2f} "
                  f"{f['dip_rad']:+8.2f} /{pub['dip']:+6.2f} "
                  f"{f['dip_rad']/f['peak_rad']:+8.2f} /"
                  f"{pub['dip']/pub['peak']:+6.2f}")
            print(f"{'':10s} N_CB max = {d['N_CB'].max():.2e} cm-3, "
                  f"N_tr max = {d['N_tr'].max():.2e} cm-3, "
                  f"A max = {d['absorption'].max():.3f}, "
                  f"plateau = {f['plateau_rad']:+.3f} (publie {pub['plateau']:+.2f})")
        print("\nCe qui marche  : la sequence des trois signes, les densites")
        print("                 (< 1e19 cm-3, la borne annoncee par l'article),")
        print("                 et le rapport creux/pic.")
        print("Ce qui ne marche pas : l'amplitude absolue, environ 4x trop")
        print("                 faible -- longueur de recouvrement effective ;")
        print("                 et le plateau STE, qui sort proche de zero au")
        print("                 lieu d'etre franchement positif. Avec les")
        print("                 forces d'oscillateur de la Table II, le retrait")
        print("                 des oscillateurs de valence compense presque")
        print("                 exactement l'apport des bandes STE.")
    return fig, res


def predict_user_config(intensities_Wcm2=(1e13, 3e13, 5e13), save=None,
                        spatial_average=1.5, verbose=True):
    """La meme courbe pour la manip : pompe 1030 nm, sonde 515 nm.

    Le taux d'ionisation vient du Keldysh du depot, pas d'un sigma_K ajuste.
    `overlap_length_m` de USER_SIO2_1030 est la corde traversee par la sonde ;
    c'est le parametre a caler sur une mesure, tout le reste est fixe.
    """
    fig, res = plot_delay_scan(USER_SIO2_1030, list(intensities_Wcm2),
                               show_channels=True, xlim=(-500, 2500),
                               save=save, spatial_average=spatial_average,
                               absorption=True)
    if verbose:
        L = USER_SIO2_1030.overlap_length_m * 1e6
        print(f"corde utilisee : L = {L:.0f} um  (parametre a caler)")
        for I, d in zip(intensities_Wcm2, res):
            f = d["features"]
            print(f"  I = {I:.1e} W/cm2 : pic {f['peak_rad']:+.3f} rad, "
                  f"creux {f['dip_rad']:+.3f} rad, plateau {f['plateau_rad']:+.3f} rad, "
                  f"T min = {1-d['absorption'].max():.3f}, "
                  f"N_CB max = {d['N_CB'].max():.2e} cm-3")
    return fig, res
