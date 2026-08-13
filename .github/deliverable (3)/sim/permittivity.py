"""Fonction dielectrique du milieu excite, d'apres Martin, Guizard, Daguzan,
Petite et al., Phys. Rev. B 55, 5799 (1997), Sec. III.

Pourquoi ce module
------------------
Le Delta n vu par la POMPE (integre par le solveur, operators.NonlinearOperator)
et celui vu par la SONDE (post-traitement : figures_filament,
web/abel_phase_explorer) etaient ecrits deux fois, a la main, avec des
conventions differentes -- et aucune des deux ne suivait le modele de l'article
de reference. Ce module ecrit la reponse UNE fois et laisse `n = sqrt(eps)`
produire a la fois l'indice et l'absorption, aux deux frequences.

Le modele de l'article
----------------------
Milieu non excite (premiere impulsion sonde, reference) :

    eps_1(w) = 1 + (N0 e^2 / m eps0) f_12 / (w_12^2 - w^2 + i w / tau_12)

Milieu excite (seconde impulsion sonde), Eq. (2) de l'article :

    eps_2(w) = 1 + (e^2/m eps0) (N0 - N_CB - N_tr) f_12/(w_12^2 - w^2 + i w/tau_12)
             + (e^2/eps0) [  N_CB f_CB / m*  * 1/(w^2 + i w/tau_ep)
                           + N_tr f_tr / m   * 1/(w_tr^2 - w^2 + i w/tau_tr) ]
             + chi3_eff E_p^2

    dPhi = (2 pi / lambda) L [ Re(sqrt(eps_2)) - Re(sqrt(eps_1)) ]
    A    = 1 - exp[ -(2 L w / c) Im(sqrt(eps_2)) ]

Quatre choses y figurent que le depot n'avait pas :

1. La DEPLETION de la bande de valence, le facteur (N0 - N_CB - N_tr). Chaque
   electron promu ou piege est retire de l'oscillateur de valence, celui-la
   meme qui fait n0 = 1.46. A 515 nm ce terme vaut 11 % du terme Drude et il a
   le MEME signe : l'ignorer sous-estime la baisse d'indice.
2. Les STE forment DEUX bandes, pas une, avec des forces d'oscillateur
   inferieures a 1 : pour SiO2 l'article donne f = 0.40 a 5.2 eV et f = 0.15 a
   4.2 eV (Table II). Le depot utilisait une bande unique f = 1 a 4.2 eV, ce
   qui surestime le canal STE d'un facteur 2.8 a 515 nm.
3. Les largeurs de bande (1.5 et 1 eV) donnent une absorption STE. Elle reste
   faible a 515 nm -- 4.6 % de l'absorption par porteurs libres -- mais c'est
   maintenant calcule, plus suppose.
4. La masse effective m* = 0.5 m_e dans le terme Drude. Le depot utilisait la
   masse nue, donc un Delta n et une section efficace deux fois trop faibles.

Le developpement au premier ordre en N/N0 que le depot utilisait
(Delta n = -rho / 2 n0 rho_c) est l'Eq. (5) de l'article, donnee "for the sake
of clarity" ; l'article precise que le fit utilise les Eqs. (2)-(4) in extenso
et que le developpement suppose N0 >> N_CB, condition remplie chez eux
(rho < 1e19 cm^-3) mais violee de deux ordres de grandeur par les runs actuels.
D'ou `linearize=False` par defaut ici.

Conventions
-----------
- Champ en exp(-i w t), donc Im(eps) > 0 = absorption.
- Densites en cm^-3, intensites en W/cm^2 (comme le npz et tout le depot).
- alpha renvoye en cm^-1 : exp(-alpha * L_cm) est la transmittance en intensite.
"""

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.constants import (epsilon_0, m_e, c as c_SI,
                             elementary_charge as q_e, hbar)

__all__ = [
    "MaterialResponse", "SIO2_MARTIN1997",
    "omega_of_wavelength", "eV_to_rad_s", "critical_density",
    "drude_delta_eps", "lorentz_delta_eps", "kerr_delta_eps",
    "valence_depletion_delta_eps", "SPM", "XPM",
]

SPM = 1.0   # la pompe sur elle-meme : auto-modulation de phase
XPM = 2.0   # une sonde faible a une autre frequence : modulation croisee


def omega_of_wavelength(lambda_m):
    return 2.0 * np.pi * c_SI / float(lambda_m)


def eV_to_rad_s(E_eV):
    return float(E_eV) * q_e / hbar


def critical_density(lambda_m, meff_rel=1.0):
    """rho_c = eps0 m* w^2 / e^2, en cm^-3."""
    w = omega_of_wavelength(lambda_m)
    return epsilon_0 * meff_rel * m_e * w**2 / q_e**2 * 1e-6


def _wp2(rho_cm3, meff_rel):
    """w_p^2 [rad^2/s^2] pour une densite en cm^-3 et une masse m* = meff_rel m_e."""
    return np.asarray(rho_cm3, float) * 1e6 * q_e**2 / (epsilon_0 * meff_rel * m_e)


# ================================================================================
#  Les quatre canaux, en permittivite
# ================================================================================
def drude_delta_eps(rho_cm3, omega, tau_ep_s, meff_rel=0.5, f_CB=1.0):
    """Porteurs libres de la bande de conduction, terme N_CB de l'Eq. (2).

        d_eps = - f_CB w_p^2 / (w^2 + i w / tau_ep)

    Le signe vient de ce que l'article ecrit +1/(w^2 + i w/tau) avec une
    convention exp(+i w t) ; en exp(-i w t) le terme des porteurs libres est
    negatif sur la partie reelle (l'indice baisse) et positif sur la partie
    imaginaire (absorption). Le facteur de collision w^2 tau^2/(1 + w^2 tau^2)
    sort du denominateur : c'est lui qui manquait cote sonde, ou le Delta n
    etait ecrit -rho/2 n0 rho_c, la limite sans collision w tau >> 1.
    """
    wp2 = _wp2(rho_cm3, meff_rel)
    w = float(omega)
    if tau_ep_s is None or tau_ep_s <= 0:
        return -f_CB * wp2 / w**2 + 0.0j
    return -f_CB * wp2 / (w**2 + 1j * w / tau_ep_s)


def lorentz_delta_eps(rho_cm3, omega, bands, meff_rel=1.0):
    """Population liee (STE), terme N_tr de l'Eq. (2), somme sur les bandes.

        d_eps = sum_j f_j w_p^2 / (w_j^2 - w^2 - i w gamma_j)

    `bands` est une sequence de (E_res_eV, f_osc, gamma_eV). L'electron piege
    est lie, donc la masse est la masse NUE (l'article ecrit N_tr f_tr / m et
    non / m*), d'ou meff_rel = 1 par defaut.
    """
    wp2 = _wp2(rho_cm3, meff_rel)
    w = float(omega)
    out = np.zeros_like(np.asarray(rho_cm3, float), dtype=complex)
    for E_res, f_osc, gamma_eV in bands:
        w_res = eV_to_rad_s(E_res)
        gam = eV_to_rad_s(gamma_eV)
        out = out + f_osc * wp2 / (w_res**2 - w**2 - 1j * w * gam)
    return out


def valence_depletion_delta_eps(rho_removed_cm3, omega, N0_cm3, eps_valence,
                                E_12_eV=None, gamma_12_eV=0.0):
    """Retrait des oscillateurs de valence promus, facteur (N0 - N_CB - N_tr).

    Deux ecritures possibles :

    - `E_12_eV=None` (defaut) : on suppose la resonance de valence tres loin de
      la sonde, donc l'oscillateur de valence vaut exactement eps_valence =
      n0^2 - 1 quelle que soit w, et en retirer une fraction x = N_removed/N0
      coute d_eps = -eps_valence * x. C'est la forme la plus sure, parce
      qu'elle est calee sur le n0 mesure (Sellmeier) et non sur un couple
      (f_12, w_12) ajuste.
    - `E_12_eV` renseigne : oscillateur de Lorentz explicite a w_12, avec la
      force d'oscillateur deduite de eps_valence a la longueur d'onde donnee.
      Utile seulement si on veut la dispersion du terme de deplation.

    A 515 nm et rho = 1e20 cm^-3, ce terme vaut -1.8e-3, soit 11 % du terme
    Drude, avec le meme signe. Il etait absent du depot.
    """
    x = np.asarray(rho_removed_cm3, float) / float(N0_cm3)
    if E_12_eV is None:
        return -float(eps_valence) * x + 0.0j
    w = float(omega)
    w12 = eV_to_rad_s(E_12_eV)
    g12 = eV_to_rad_s(gamma_12_eV)
    # force d'oscillateur telle que la population complete redonne eps_valence
    ref = 1.0 / (w12**2 - w**2 - 1j * w * g12)
    return -float(eps_valence) * x * (ref / ref.real if np.iscomplexobj(ref) else 1.0)


def kerr_delta_eps(I_Wcm2, n0, n2_m2W, xpm_factor=SPM):
    """Terme chi3_eff E_p^2 de l'Eq. (2), ecrit en permittivite.

        d_eps = 2 n0 X n2 I

    `xpm_factor` vaut 1 pour la pompe sur elle-meme et 2 pour une sonde faible
    a une autre frequence : le terme croise du chi3 non degenere compte double
    (Boyd, Nonlinear Optics, ch. 4). ATTENTION cependant : le n2 = 2e-16 cm^2/W
    que l'article ajuste pour SiO2 est deja defini comme l'indice vu par la
    SONDE par unite d'intensite de pompe (leur Eq. (5) s'ecrit n2 I_p sans
    facteur 2). Utiliser ce n2-la avec xpm_factor=2 compterait le facteur deux
    fois. Regle : xpm_factor=2 avec un n2 d'auto-modulation (celui du solveur,
    2.74e-20 m^2/W), xpm_factor=1 avec un n2 ajuste sur une mesure sonde.
    """
    I_SI = np.asarray(I_Wcm2, float) * 1e4                    # W/cm2 -> W/m2
    return 2.0 * float(n0) * float(xpm_factor) * float(n2_m2W) * I_SI + 0.0j


# ================================================================================
#  Jeu de parametres materiau
# ================================================================================
@dataclass
class MaterialResponse:
    """Parametres de la fonction dielectrique d'un dielectrique excite.

    Les valeurs par defaut sont celles de la Table II de Martin et al. (1997)
    pour SiO2. `tau_ep_s` y est donne comme 1/tau_ep = 1.5e15 s^-1, soit
    tau_ep = 0.67 fs -- a comparer aux 1.7 fs du depot (convention Bulgakova)
    et aux 10 fs de Couairon 2005 (ajuste sur la transmission de la pompe).
    Les trois calent la MEME grandeur physique sur trois mesures differentes.
    """
    name: str = "SiO2 (Martin et al., PRB 55, 5799 (1997), Table II)"
    N0_cm3: float = 2.2e22                # densite d'electrons de valence
    meff_rel: float = 0.5                 # m*/m_e dans la bande de conduction
    f_CB: float = 1.0                     # force d'oscillateur des porteurs libres
    tau_ep_s: float = 1.0 / 1.5e15        # collisions electron-phonon, 0.67 fs
    # bandes STE : (E_resonance_eV, f_oscillateur, largeur_gamma_eV)
    ste_bands: Sequence[Tuple[float, float, float]] = field(
        default_factory=lambda: ((5.2, 0.40, 1.5), (4.2, 0.15, 1.0)))
    n2_m2W: float = 0.0                   # a renseigner par l'appelant
    enable_valence_depletion: bool = True

    def response(self, lambda_m, n0_lin, rho_e_cm3=0.0, rho_s_cm3=0.0,
                 I_Wcm2=0.0, xpm_factor=SPM, linearize=False,
                 include=("drude", "ste", "kerr", "depletion")):
        """Reponse complete a la longueur d'onde `lambda_m`.

        Renvoie un dict :
            eps        permittivite complexe du milieu excite
            n_complex  sqrt(eps)
            dn         Re(n) - n0_lin            -> ce qui dephase
            alpha_cm   2 w Im(n)/c en cm^-1      -> ce qui absorbe
            dn_<canal> contribution de chaque canal, meme convention
            overdense  True la ou Re(eps) <= 0 : plus d'indice, le milieu
                       reflechit

        `linearize=True` remplace sqrt(eps) par le developpement au premier
        ordre (Eq. (5) de l'article), qui est ce que faisait le depot. Garde
        pour pouvoir chiffrer l'ecart, pas comme defaut : a rho = 4.4e21 cm^-3
        le developpement se trompe deja de 14 %, et a 2.1e22 il donne
        Delta n = -3.25 alors que Re(eps) est negatif, c'est a dire que le
        milieu ne dephase plus du tout mais reflechit.
        """
        w = omega_of_wavelength(lambda_m)
        n0_lin = float(n0_lin)
        eps_valence = n0_lin**2 - 1.0

        rho_e = np.asarray(rho_e_cm3, float)
        rho_s = np.asarray(rho_s_cm3, float)

        zero = np.zeros_like(np.broadcast_arrays(rho_e, rho_s)[0], dtype=complex)
        d_drude = (drude_delta_eps(rho_e, w, self.tau_ep_s, self.meff_rel, self.f_CB)
                   if "drude" in include else zero)
        d_ste = (lorentz_delta_eps(rho_s, w, self.ste_bands)
                 if "ste" in include else zero)
        d_kerr = (kerr_delta_eps(I_Wcm2, n0_lin, self.n2_m2W, xpm_factor)
                  if "kerr" in include else zero)
        if "depletion" in include and self.enable_valence_depletion:
            d_depl = valence_depletion_delta_eps(rho_e + rho_s, w,
                                                 self.N0_cm3, eps_valence)
        else:
            d_depl = zero

        d_tot = d_drude + d_ste + d_kerr + d_depl
        eps = n0_lin**2 + d_tot

        if linearize:
            dn = np.real(d_tot) / (2.0 * n0_lin)
            kappa = np.imag(d_tot) / (2.0 * n0_lin)
        else:
            n_c = np.sqrt(np.asarray(eps, dtype=complex))
            dn = np.real(n_c) - n0_lin
            kappa = np.imag(n_c)

        def _dn(d):
            return np.real(d) / (2.0 * n0_lin)

        def _al(d):
            return 2.0 * w * (np.imag(d) / (2.0 * n0_lin)) / c_SI * 1e-2

        return dict(
            eps=eps, n_complex=(n0_lin + dn) + 1j * kappa,
            dn=dn, kappa=kappa, alpha_cm=2.0 * w * kappa / c_SI * 1e-2,
            dn_drude=_dn(d_drude), dn_ste=_dn(d_ste),
            dn_kerr=_dn(d_kerr), dn_depletion=_dn(d_depl),
            alpha_drude_cm=_al(d_drude), alpha_ste_cm=_al(d_ste),
            overdense=np.real(eps) <= 0.0, omega=w, n0_lin=n0_lin,
        )

    # ---- diagnostics ----------------------------------------------------
    def f_ste_effective(self, lambda_m):
        """Facteur sans dimension devant rho_s/(2 n0 rho_c) une fois les bandes
        sommees -- l'equivalent du `f_ste` scalaire que le depot utilisait, pour
        comparaison directe."""
        w = omega_of_wavelength(lambda_m)
        d = lorentz_delta_eps(np.array([1.0]), w, self.ste_bands)
        return float(np.real(d)[0] / (_wp2(1.0, 1.0) / w**2))

    def sigma_fca_cm2(self, lambda_m, n0_lin):
        """Section efficace d'absorption par porteur libre, en cm^2."""
        w = omega_of_wavelength(lambda_m)
        d = drude_delta_eps(np.array([1e18]), w, self.tau_ep_s,
                            self.meff_rel, self.f_CB)
        alpha = 2.0 * w * (np.imag(d)[0] / (2.0 * float(n0_lin))) / c_SI * 1e-2
        return alpha / 1e18


SIO2_MARTIN1997 = MaterialResponse()
