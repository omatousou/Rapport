"""Figures destinees au rapport (main.tex).

Chaque fonction produit UNE figure qui illustre UN point precis d'une section
du .tex. Le nom de la section visee est rappele dans la docstring, pour que la
figure et le paragraphe qu'elle appuie restent apparies quand l'un des deux
bouge.

    plot_clamping_equilibrium   -> sec:filamentation  (n2 I = rho / 2 rho_c)
    plot_selffocusing_vs_diffraction -> sec:filamentation (L_c, z_R)
    plot_avalanche_takeover     -> sec:avalanche      (graine MPI puis cascade)
    plot_trapping_sequence      -> sec 9 / sec 10     (libres -> pieges)
    plot_index_channels         -> sec 6 / sec 7      (Kerr + vs Drude -)
    plot_abel_illustration      -> sec:interferometrie (ce que la sonde integre)
    plot_energy_budget          -> sec:propagation    (ou part l'energie)

Tout est calcule depuis le result.npz : aucune de ces fonctions ne relance de
simulation. Les constantes materiau qui ne sont pas dans le npz (n2, tau_c,
U_i...) doivent etre passees explicitement, avec les MEMES valeurs que le run,
sinon la figure raconte une autre physique que celle qui a ete integree.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import (epsilon_0, m_e, c as c_SI,
                             elementary_charge as q_e, hbar)

from figures_filament import (z_um_of, r_um_of, peak_z_um, critical_power,
                              marburger_collapse, entrance_radius,
                              build_abel_matrix, _populations_at, probe_sigma)

__all__ = [
    "critical_density", "avalanche_beta",
    "plot_clamping_equilibrium", "plot_selffocusing_vs_diffraction",
    "plot_avalanche_takeover", "plot_trapping_sequence",
    "plot_index_channels", "plot_abel_illustration", "plot_energy_budget",
    "export_report_figures",
]

_TRAPZ = getattr(np, "trapezoid", None) or np.trapz


# ================================================================================
#  Petites grandeurs derivees
# ================================================================================
def critical_density(wavelength_m, meff_rel=1.0):
    """rho_c = eps0 m* omega^2 / e^2, en cm-3."""
    w = 2.0 * np.pi * c_SI / wavelength_m
    return epsilon_0 * meff_rel * m_e * w**2 / q_e**2 * 1e-6


def avalanche_beta(wavelength_m, tau_c_s, Ui_eV, meff_rel=1.0, n0=1.45):
    """Coefficient d'avalanche beta = sigma / U_i, en cm2/J.

    sigma est la section efficace de Bremsstrahlung inverse A LA POMPE (Drude),
    identique a celle du solveur ; U_i est le potentiel d'ionisation. La
    dependance de sigma en tau_c est non monotone -- sigma ~ tau_c / (1 +
    omega^2 tau_c^2) passe par un maximum en omega tau_c = 1 -- donc changer
    tau_c d'un facteur 6 ne change PAS beta d'un facteur 6.
    """
    w = 2.0 * np.pi * c_SI / wavelength_m
    k = w * n0 / c_SI
    m = meff_rel * m_e
    sigma = (k * q_e**2 * tau_c_s) / (n0**2 * m * epsilon_0 * w
                                      * (1.0 + (w * tau_c_s)**2)) * 1e4   # cm2
    return sigma / (Ui_eV * q_e), sigma


def _peak_plane(res, z_target_um=None):
    z_um = z_um_of(res)
    if z_target_um is None:
        return int(np.argmax(np.asarray(res["Imax_z"]))), z_um
    return int(np.argmin(np.abs(z_um - z_target_um))), z_um


# ================================================================================
#  sec:filamentation -- le clampage est un equilibre, pas un plafond
# ================================================================================
def plot_clamping_equilibrium(res, n2, wavelength_m, meff_rel=1.0,
                              label="", save=None):
    """Confronte les deux termes d'indice sur l'axe, en fonction de z.

    La section ecrit le clampage comme l'egalite n2 I = rho / (2 rho_c) : le
    Kerr focalisant et le plasma defocalisant se compensent. Cette figure
    montre les deux termes separement et leur somme. Tant que la somme reste
    positive le faisceau continue de s'auto-focaliser ; le premier passage par
    zero est le vrai debut du regime de filament, et non le maximum
    d'intensite.
    """
    z_um = z_um_of(res)
    I = np.asarray(res["Imax_z"], float)                    # W/cm2
    iax = len(r_um_of(res)) // 2
    rho = np.asarray(res["rho_rz"], float)[:, iax]          # cm-3, fin d'impulsion
    rho_c = critical_density(wavelength_m, meff_rel)

    dn_kerr = n2 * I * 1e4                                  # n2 en cm2/W -> SI
    dn_plasma = -rho / (2.0 * rho_c)
    dn_net = dn_kerr + dn_plasma

    fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                           gridspec_kw=dict(height_ratios=[2, 1]))

    ax[0].plot(z_um, dn_kerr, color="crimson", lw=2,
               label=r"Kerr  $n_2 I$")
    ax[0].plot(z_um, -dn_plasma, color="royalblue", lw=2,
               label=r"plasma  $\rho_e / 2\rho_c$")
    ax[0].set_yscale("log")
    ax[0].set_ylabel(r"$|\Delta n|$")
    ax[0].legend(loc="upper left")
    ax[0].grid(alpha=0.3, which="both")
    ax[0].set_title(f"clamping balance {label}".strip())

    ax[1].plot(z_um, dn_net, color="k", lw=1.6)
    ax[1].axhline(0.0, color="0.4", lw=1, ls="--")
    ax[1].fill_between(z_um, 0, dn_net, where=dn_net > 0,
                       color="crimson", alpha=0.25, label="net self-focusing")
    ax[1].fill_between(z_um, 0, dn_net, where=dn_net <= 0,
                       color="royalblue", alpha=0.25, label="net defocusing")
    # symlog : sans ca le puits de plasma (1e-1) ecrase completement la zone
    # auto-focalisante (1e-3), qui est justement celle qu'on veut montrer.
    lin = max(float(np.nanmax(np.abs(dn_kerr))) * 1e-2, 1e-12)
    ax[1].set_yscale("symlog", linthresh=lin)
    ax[1].set_ylabel(r"net $\Delta n$")
    ax[1].set_xlabel(r"$z$ [$\mu$m]")
    ax[1].legend(loc="upper right", fontsize=9)
    ax[1].grid(alpha=0.3)

    crossings = np.where(np.diff(np.sign(dn_net)))[0]
    z_cross = z_um[crossings] if len(crossings) else np.array([])
    for zc in z_cross[:8]:
        ax[0].axvline(zc, color="0.6", lw=0.8, ls=":")
        ax[1].axvline(zc, color="0.6", lw=0.8, ls=":")

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=160, bbox_inches="tight")
    return fig, dict(z_um=z_um, dn_kerr=dn_kerr, dn_plasma=dn_plasma,
                     dn_net=dn_net, rho_c_cm3=rho_c, z_crossings_um=z_cross)


# ================================================================================
#  sec:filamentation -- longueur de collapse et longueur de Rayleigh
# ================================================================================
def _beam_radius_vs_z(res, frac=1.0 / np.e**2):
    """Rayon a `frac` du maximum, sur la carte de fluence, plan par plan."""
    r_full = r_um_of(res)
    half = len(r_full) // 2
    r_pos = r_full[half:]
    F = np.asarray(res["fluence_rz"], float)[:, half:]
    w = np.empty(F.shape[0])
    for i, prof in enumerate(F):
        pk = prof.max()
        if pk <= 0:
            w[i] = np.nan
            continue
        below = np.where(prof < frac * pk)[0]
        if len(below) == 0:
            w[i] = r_pos[-1]
        else:
            j = below[0]
            if j == 0:
                w[i] = r_pos[0]
            else:                                    # interpolation lineaire
                y0, y1 = prof[j - 1], prof[j]
                t = (frac * pk - y0) / (y1 - y0) if y1 != y0 else 0.0
                w[i] = r_pos[j - 1] + t * (r_pos[j] - r_pos[j - 1])
    return w


def plot_selffocusing_vs_diffraction(res, w0_m, wavelength_m, n0, n2,
                                     energy_uJ, delta_t_s, begin_m=0.0,
                                     label="", save=None):
    """Rayon simule vs propagation lineaire, avec L_c de Marburger.

    C'est la contrepartie numerique du schema de la section filamentation :
    la courbe lineaire s'ouvre en sqrt(1 + (z/z_R)^2) alors que la courbe
    simulee s'effondre. L'ecart entre les deux EST l'auto-focalisation, et
    l'endroit ou la courbe simulee cesse de descendre est l'arret par le
    plasma.
    """
    z_um = z_um_of(res)
    w_sim = _beam_radius_vs_z(res)

    zR_um = np.pi * n0 * w0_m**2 / wavelength_m * 1e6
    z_lin = z_um - begin_m * 1e6
    w_lin = w0_m * 1e6 * np.sqrt(1.0 + (z_lin / zR_um)**2)

    P_cr = critical_power(n2, wavelength_m, n0)
    tp = delta_t_s / np.sqrt(2.0 * np.log(2.0))
    P_in = energy_uJ * 1e-6 / (tp * np.sqrt(np.pi / 2.0))
    w_in = entrance_radius(w0_m, begin_m, wavelength_m, n0)
    ratio, L_DF, L_c, _ = marburger_collapse(P_in, P_cr, w_in, wavelength_m, n0)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(z_um, w_lin, color="0.55", lw=2, ls="--",
            label=rf"linear propagation ($z_R$ = {zR_um:.0f} $\mu$m)")
    ax.plot(z_um, w_sim, color="crimson", lw=2, label=r"simulation, $1/e^2$ radius")
    if 0 < L_c * 1e6 < z_um[-1]:
        ax.axvline(L_c * 1e6, color="darkgreen", lw=1.5, ls="-.",
                   label=rf"Marburger $L_c$ = {L_c*1e6:.0f} $\mu$m")
    iz = int(np.argmax(np.asarray(res["Imax_z"])))
    ax.axvline(z_um[iz], color="royalblue", lw=1.2, ls=":",
               label=rf"simulated $I$ max ({z_um[iz]:.0f} $\mu$m)")
    ax.set_xlabel(r"$z$ [$\mu$m]")
    ax.set_ylabel(r"radius [$\mu$m]")
    ax.set_yscale("log")
    ax.set_title(f"self-focusing vs diffraction {label}   "
                 rf"($P/P_{{cr}}$ = {ratio:.1f})".strip())
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=160, bbox_inches="tight")
    return fig, dict(z_um=z_um, w_sim_um=w_sim, w_lin_um=w_lin,
                     zR_um=zR_um, L_c_um=L_c * 1e6, P_over_Pcr=ratio,
                     w_min_um=float(np.nanmin(w_sim)))


# ================================================================================
#  sec:avalanche -- la graine multiphotonique puis la cascade
# ================================================================================
def plot_avalanche_takeover(res, wavelength_m, tau_c_s, Ui_eV, meff_rel=1.0,
                            n0=1.45, rho_max_cm3=2.1e22, z_target_um=None,
                            label="", save=None):
    """Densite sur l'axe et taux de creation, au plan le plus intense.

    Le paragraphe avalanche dit que la cascade doit d'abord etre AMORCEE par
    la photoionisation. Cette figure le rend quantitatif : on trace le taux
    total drho/dt (derive de la trace sauvegardee) et le seul terme
    d'avalanche beta I rho. Leur croisement date l'instant ou la multiplication
    par impact prend le pas sur la graine multiphotonique.
    """
    iz, z_um = _peak_plane(res, z_target_um)
    t_fs = np.asarray(res["t_full_fs"], float)
    rho = np.asarray(res["rho_onaxis_t"], float)[iz]
    I = np.asarray(res["I_onaxis_t"], float)[iz]

    beta, sigma = avalanche_beta(wavelength_m, tau_c_s, Ui_eV, meff_rel, n0)
    # meme forme que le solveur : la deplation (1 - rho/rho_max) borne la
    # cascade. L'omettre surestimerait le terme d'avalanche des que rho
    # approche rho_max, et le "reste" attribue a la photoionisation
    # deviendrait negatif par construction.
    depl = np.clip(1.0 - rho / rho_max_cm3, 0.0, 1.0)
    rate_aval = beta * I * rho * depl                            # cm-3 / s
    dt_s = np.gradient(t_fs) * 1e-15
    rate_tot = np.gradient(rho) / dt_s
    rate_mpi = np.clip(rate_tot - rate_aval, 0.0, None)

    fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                           gridspec_kw=dict(height_ratios=[1, 1]))

    ax[0].plot(t_fs, rho, color="k", lw=2, label=r"$\rho_e$")
    ax[0].set_yscale("log")
    ax[0].set_ylabel(r"$\rho_e$ [cm$^{-3}$]")
    ax[0].grid(alpha=0.3, which="both")
    axI = ax[0].twinx()
    axI.plot(t_fs, I, color="darkorange", lw=1.2, alpha=0.8)
    axI.set_ylabel(r"$I$ [W/cm$^2$]", color="darkorange")
    axI.tick_params(axis="y", colors="darkorange")
    ax[0].set_title(f"seeding then avalanche, z = {z_um[iz]:.0f} um {label}".strip())

    ax[1].plot(t_fs, np.clip(rate_mpi, 1e-30, None), color="royalblue", lw=1.8,
               label=r"photoionization = d$\rho_e$/d$t$ $-$ avalanche")
    ax[1].plot(t_fs, np.clip(rate_aval, 1e-30, None), color="crimson", lw=1.8,
               label=rf"avalanche $\beta I \rho_e$  ($\beta$ = {beta:.2f} cm$^2$/J)")
    ax[1].set_yscale("log")
    ax[1].set_ylim(max(1e10, np.nanmax(rate_tot) * 1e-6),
                   np.nanmax(rate_tot) * 3.0)
    ax[1].set_ylabel(r"d$\rho_e$/d$t$ [cm$^{-3}$ s$^{-1}$]")
    ax[1].set_xlabel(r"local $t$ [fs]")
    ax[1].legend(fontsize=9, loc="upper left")
    ax[1].grid(alpha=0.3, which="both")

    # instant de bascule : premier t ou l'avalanche depasse la graine
    over = np.where((rate_aval > rate_mpi) & (rho > rho.max() * 1e-6))[0]
    t_switch = float(t_fs[over[0]]) if len(over) else np.nan
    if np.isfinite(t_switch):
        for a in ax:
            a.axvline(t_switch, color="0.4", lw=1, ls="--")
        ax[1].annotate(f"takeover {t_switch:+.0f} fs", (t_switch, ax[1].get_ylim()[1]),
                       xytext=(6, -14), textcoords="offset points", fontsize=9)

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=160, bbox_inches="tight")
    # gain integre sur l'impulsion : exp(int beta I dt)
    gain = float(np.exp(_TRAPZ(beta * I * depl, t_fs * 1e-15)))
    return fig, dict(z_um=float(z_um[iz]), beta_cm2_J=beta,
                     sigma_cm2=sigma, t_switch_fs=t_switch,
                     avalanche_gain=gain, rho_final=float(rho[-1]))


# ================================================================================
#  sec 9 / sec 10 -- libres puis pieges
# ================================================================================
def plot_trapping_sequence(res, tau_r_s=330e-15, tau_ste_s=1e-12,
                           t_end_ps=2.0, z_target_um=None, label="", save=None):
    """rho_e et rho_STE sur l'axe, de l'impulsion jusqu'a 2 ps.

    Les sections excitons / STE decrivent un transfert : les electrons libres
    disparaissent avec tau_r, la population piegee monte avec le meme tau_r
    puis decroit avec tau_STE. La partie au-dela de la fenetre du solveur est
    integree analytiquement (les equations de population y sont lineaires,
    le champ est nul) -- c'est la meme extrapolation que celle utilisee pour
    les cartes de sonde.
    """
    iz, z_um = _peak_plane(res, z_target_um)
    t_fs = np.asarray(res["t_full_fs"], float)
    rho_e_in = np.asarray(res["rho_onaxis_t"], float)[iz]
    rho_s_in = np.asarray(res["rho_s_onaxis_t"], float)[iz]

    # prolongement analytique jusqu'a t_end_ps
    t_tail = np.linspace(t_fs[-1], t_end_ps * 1000.0, 400)[1:]
    dt = (t_tail - t_fs[-1]) * 1e-15
    e0, s0 = rho_e_in[-1], rho_s_in[-1]
    dr = np.exp(-dt / tau_r_s)
    if tau_ste_s is None:
        s_tail = s0 + e0 * (1.0 - dr)
    else:
        ds = np.exp(-dt / tau_ste_s)
        B = (e0 / tau_r_s) / (1.0 / tau_ste_s - 1.0 / tau_r_s)
        s_tail = s0 * ds + B * (dr - ds)
    e_tail = e0 * dr

    t_all = np.concatenate([t_fs, t_tail]) * 1e-3            # ps
    e_all = np.concatenate([rho_e_in, e_tail])
    s_all = np.concatenate([rho_s_in, s_tail])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(t_all, e_all, color="royalblue", lw=2, label=r"$\rho_e$ free")
    ax.plot(t_all, s_all, color="darkgreen", lw=2, label=r"$\rho_{STE}$ trapped")
    ax.axvline(t_fs[-1] * 1e-3, color="0.6", lw=1, ls="--")
    ax.annotate("end of solver window\n(analytic continuation)",
                xy=(t_fs[-1] * 1e-3, 1.0), xycoords=("data", "axes fraction"),
                xytext=(7, -26), textcoords="offset points",
                fontsize=8, color="0.35")
    ax.set_yscale("log")
    ax.set_ylim(max(e_all.max(), s_all.max()) * 1e-4, None)
    ax.set_xlabel(r"local $t$ [ps]")
    ax.set_ylabel(r"density [cm$^{-3}$]")
    ax.set_title(rf"free $\to$ trapped, z = {z_um[iz]:.0f} $\mu$m   "
                 rf"($\tau_r$ = {tau_r_s*1e15:.0f} fs, "
                 rf"$\tau_{{STE}}$ = {tau_ste_s*1e12:.1f} ps) {label}".strip())
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=160, bbox_inches="tight")
    return fig, dict(z_um=float(z_um[iz]), t_ps=t_all, rho_e=e_all, rho_s=s_all,
                     rho_e_peak=float(e_all.max()), rho_s_peak=float(s_all.max()))


# ================================================================================
#  sec 6 / sec 7 -- competition des signes sur Delta n
# ================================================================================
def plot_index_channels(res, delay_fs=0.0, lambda_probe_m=515e-9, E_tr_eV=4.2,
                        n2=3.54e-20, tau_r_s=330e-15, tau_ste_s=1e-12,
                        n_g=1.4627, r_max_um=30.0, z_target_um=None,
                        yscale="symlog", label="", save=None):
    """Profil radial de Delta n, canal par canal, a un plan et un delai donnes.

    Kerr et STE sont positifs, le plasma est negatif : c'est la lentille
    convergente contre la lentille divergente decrite dans les sections Kerr et
    defocalisation plasma. Le profil du canal Drude est plus etroit que celui
    du Kerr (l'ionisation depend d'une puissance elevee de I), ce qui est la
    raison pour laquelle le plasma creuse le centre du faisceau au lieu de le
    defocaliser en bloc.
    """
    from keldysh import n_sellmeier as _ns

    n0p = _ns(lambda_probe_m)
    nc = epsilon_0 * m_e * (2 * np.pi * c_SI / lambda_probe_m)**2 / q_e**2 * 1e-6
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

    iz = (int(np.argmax(np.asarray(res["Imax_z"]))) if z_target_um is None
          else int(np.argmin(np.abs(z_um - z_target_um))))
    den = 2.0 * n0p * nc

    dn_drude = -rho_e[iz] / den
    dn_ste = f_ste * rho_s[iz] / den
    dn_kerr = n2 * I[iz] * 1e4
    dn_tot = dn_drude + dn_ste + dn_kerr

    m = r_pos <= r_max_um
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(r_pos[m], dn_kerr[m], color="crimson", lw=2, label=r"Kerr  $+n_2 I$")
    ax.plot(r_pos[m], dn_drude[m], color="royalblue", lw=2,
            label=r"Drude  $-\rho_e/2n_0\rho_c$")
    ax.plot(r_pos[m], dn_ste[m], color="darkgreen", lw=2,
            label=rf"STE  $+f\,\rho_{{STE}}/2n_0\rho_c$  ($f$ = {f_ste:.3f})")
    ax.plot(r_pos[m], dn_tot[m], color="k", lw=2.2, ls="--", label="sum")
    ax.axhline(0.0, color="0.5", lw=1)
    if yscale == "symlog":
        # les trois canaux peuvent differer de plusieurs decades (le Kerr
        # domine pendant l'impulsion, le STE longtemps apres) : en lineaire
        # deux des trois courbes seraient confondues avec l'axe.
        span = max(np.abs(dn_kerr).max(), np.abs(dn_drude).max(),
                   np.abs(dn_ste).max())
        ax.set_yscale("symlog", linthresh=max(span * 1e-4, 1e-14))
    ax.set_xlabel(r"$r$ [$\mu$m]")
    ax.set_ylabel(r"$\Delta n$ seen by the probe")
    ax.set_title(rf"index channels, z = {z_um[iz]:.0f} $\mu$m, "
                 rf"delay {delay_fs:+.0f} fs {label}".strip())
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=160, bbox_inches="tight")
    return fig, dict(z_um=float(z_um[iz]), r_um=r_pos, f_ste=f_ste,
                     dn_kerr=dn_kerr, dn_drude=dn_drude, dn_ste=dn_ste,
                     dn_total=dn_tot)


# ================================================================================
#  sec:interferometrie -- ce que la sonde integre reellement
# ================================================================================
def plot_abel_illustration(res, delay_fs=0.0, lambda_probe_m=515e-9,
                           x_half_um=40.0, z_target_um=None, label="",
                           save=None, **probe_kw):
    """Delta n(r) local et sa projection le long de la corde OPL(x).

    La section interferometrie insiste sur le fait que la mesure ne donne pas
    Delta n(r) mais son integrale sur la ligne de visee, et qu'il faut inverser
    une transformee d'Abel pour remonter au profil radial. Cette figure montre
    les deux cotes du probleme sur le meme plan : le profil vrai a gauche, ce
    que l'interferometre enregistre a droite. La projection est plus large et
    plus lisse que le profil : toute structure fine sur l'axe est diluee, ce
    qui est exactement la difficulte de l'inversion.
    """
    from figures_filament import probe_opl_transmittance
    d = probe_opl_transmittance(res, delay_fs, lambda_probe_m=lambda_probe_m,
                                x_half_um=x_half_um, **probe_kw)
    z_um = d["z_um"]
    iz = (int(np.argmax(np.abs(d["opl_nm"]).max(axis=1))) if z_target_um is None
          else int(np.argmin(np.abs(z_um - z_target_um))))

    r_pos, dn = d["r_pos_um"], d["dn_rz"][iz]
    x, opl = d["x_um"], d["opl_nm"][iz]
    lam_nm = lambda_probe_m * 1e9

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    m = r_pos <= x_half_um
    ax[0].plot(r_pos[m], dn[m], color="crimson", lw=2)
    ax[0].axhline(0.0, color="0.5", lw=1)
    ax[0].set_xlabel(r"$r$ [$\mu$m]")
    ax[0].set_ylabel(r"$\Delta n(r)$  (local)")
    ax[0].set_title("true radial profile (simulation)")
    ax[0].grid(alpha=0.3)

    ax[1].plot(x, opl, color="royalblue", lw=2)
    ax[1].axhline(0.0, color="0.5", lw=1)
    ax[1].set_xlabel(r"$x$ [$\mu$m]  (chord)")
    ax[1].set_ylabel("OPL [nm]")
    ax[1].set_title("Abel projection = what the probe measures")
    ax[1].grid(alpha=0.3)
    axp = ax[1].twinx()
    axp.plot(x, 2.0 * np.pi * opl / lam_nm, alpha=0.0)
    axp.set_ylabel(rf"$\varphi$ [rad] at {lam_nm:.0f} nm")

    fig.suptitle(rf"forward Abel transform, z = {z_um[iz]:.0f} $\mu$m, "
                 rf"delay {delay_fs:+.0f} fs {label}".strip())
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=160, bbox_inches="tight")
    return fig, dict(z_um=float(z_um[iz]), r_um=r_pos, dn=dn,
                     x_um=x, opl_nm=opl, phi_rad=2.0 * np.pi * opl / lam_nm)


# ================================================================================
#  sec:propagation -- ou part l'energie
# ================================================================================
def plot_energy_budget(res, label="", save=None):
    """Fractions d'energie cumulees perdues par photoionisation et par plasma.

    L'equation maitresse contient deux termes de perte : le terme de
    photoionisation (qui coute U_i par paire creee) et le terme Drude
    (chauffage des porteurs deja libres). Leur poids relatif dit lequel des
    deux mecanismes domine l'absorption -- et donc si le trou de transmission
    mesure vient de la creation ou du chauffage.
    """
    z_um = z_um_of(res)
    E_mpi = np.asarray(res["E_MPI_z"], float)
    E_pl = np.asarray(res["E_plasma_z"], float)
    E_tot = np.asarray(res["E_total_z"], float)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(z_um, E_mpi, color="royalblue", lw=2, label="photoionization")
    ax.plot(z_um, E_pl, color="crimson", lw=2, label="plasma (Drude)")
    ax.plot(z_um, E_tot, color="k", lw=2.2, ls="--", label="total")
    ax.set_xlabel(r"$z$ [$\mu$m]")
    ax.set_ylabel(r"cumulative fraction of $U_0$")
    ax.set_title(f"energy budget {label}".strip())
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=160, bbox_inches="tight")
    return fig, dict(z_um=z_um, E_mpi=E_mpi, E_plasma=E_pl, E_total=E_tot,
                     total_final=float(E_tot[-1]))


# ================================================================================
#  Tout d'un coup
# ================================================================================
def export_report_figures(res, out_dir, wavelength_m, n0, n2, w0_m, energy_uJ,
                          delta_t_s, tau_c_s, Ui_eV, meff_rel=1.0,
                          begin_m=0.0, rho_max_cm3=2.1e22,
                          tau_r_s=330e-15, tau_ste_s=1e-12,
                          lambda_probe_m=515e-9, E_tr_eV=4.2, label="",
                          probe_kw=None, verbose=True):
    """Genere les sept figures et renvoie {nom: diagnostics}.

    `probe_kw` est passe tel quel aux fonctions qui appellent la sonde, pour
    que tau_c / n2 / tau_r y soient les memes que dans le run.
    """
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pk = dict(probe_kw or {})
    # lambda_probe_m / E_tr_eV / n2 sont deja passes explicitement plus bas :
    # les laisser aussi dans pk leverait "multiple values for argument".
    for k in ("lambda_probe_m", "E_tr_eV", "n2"):
        pk.pop(k, None)
    diag = {}

    _f, diag["clamping"] = plot_clamping_equilibrium(
        res, n2, wavelength_m, meff_rel, label=label,
        save=str(out / "rep_clamping_equilibrium.png"))
    _f, diag["focusing"] = plot_selffocusing_vs_diffraction(
        res, w0_m, wavelength_m, n0, n2, energy_uJ, delta_t_s, begin_m,
        label=label, save=str(out / "rep_selffocusing.png"))
    _f, diag["avalanche"] = plot_avalanche_takeover(
        res, wavelength_m, tau_c_s, Ui_eV, meff_rel, n0,
        rho_max_cm3=rho_max_cm3, label=label,
        save=str(out / "rep_avalanche_takeover.png"))
    _f, diag["trapping"] = plot_trapping_sequence(
        res, tau_r_s, tau_ste_s, label=label,
        save=str(out / "rep_trapping_sequence.png"))
    _f, diag["channels"] = plot_index_channels(
        res, 0.0, lambda_probe_m, E_tr_eV, n2, tau_r_s, tau_ste_s,
        label=label, save=str(out / "rep_index_channels.png"))
    _f, diag["abel"] = plot_abel_illustration(
        res, 0.0, lambda_probe_m, label=label,
        save=str(out / "rep_abel_illustration.png"),
        n2=n2, E_tr_eV=E_tr_eV, **pk)
    _f, diag["energy"] = plot_energy_budget(
        res, label=label, save=str(out / "rep_energy_budget.png"))
    plt.close("all")

    if verbose:
        d = diag
        print(f"rho_c (pompe {wavelength_m*1e9:.0f} nm, m*={meff_rel:g} m_e) = "
              f"{d['clamping']['rho_c_cm3']:.3e} cm-3")
        nz = d["clamping"]["z_crossings_um"]
        print(f"passages Kerr/plasma a l'equilibre : "
              f"{len(nz)} ; premiers = {np.round(nz[:5], 1)}")
        print(f"rayon min simule = {d['focusing']['w_min_um']:.2f} um  "
              f"(z_R = {d['focusing']['zR_um']:.0f} um, "
              f"L_c = {d['focusing']['L_c_um']:.0f} um, "
              f"P/P_cr = {d['focusing']['P_over_Pcr']:.1f})")
        print(f"avalanche : beta = {d['avalanche']['beta_cm2_J']:.2f} cm2/J, "
              f"gain exp(int beta I dt) = {d['avalanche']['avalanche_gain']:.3e}, "
              f"bascule a {d['avalanche']['t_switch_fs']:+.0f} fs")
        print(f"pieges : rho_e max = {d['trapping']['rho_e_peak']:.3e}, "
              f"rho_STE max = {d['trapping']['rho_s_peak']:.3e} cm-3")
        print(f"canaux a z = {d['channels']['z_um']:.0f} um : "
              f"Kerr {d['channels']['dn_kerr'].max():+.3e}, "
              f"Drude {d['channels']['dn_drude'].min():+.3e}, "
              f"STE {d['channels']['dn_ste'].max():+.3e}")
        print(f"OPL projete max = {np.abs(d['abel']['opl_nm']).max():.2f} nm "
              f"= {np.abs(d['abel']['phi_rad']).max():.3f} rad")
        print(f"pertes totales = {d['energy']['total_final']*100:.1f} % de U0")
        print(f"-> {out}")
    return diag
