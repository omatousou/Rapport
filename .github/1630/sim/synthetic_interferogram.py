"""Synthetiser l'interferogramme que la camera aurait enregistre.

Jusqu'ici la comparaison simulation / mesure s'arretait au Delta n projete : on
comparait une grandeur physique du modele a une grandeur DEJA DEPOUILLEE de la
manip. Tout ce que le depouillement fait subir aux donnees -- filtrage de
Takeda, decoupe a l'ouverture numerique, ajustement du plan de fond, rotation,
repliement de phase -- n'affectait qu'un des deux cotes de la comparaison.

Ce module ferme la boucle. Il fabrique, a partir de la simulation, un
interferogramme brut au format exact des fichiers de la manip (`side_sig` et
`side_bg`, 2048 x 2048), de sorte qu'on puisse le passer dans LE MEME code de
depouillement que les vraies images -- `preprocess_side_full` de
web/abel_phase_explorer.py. Les deux cotes subissent alors rigoureusement le
meme traitement, biais compris.

Ce que la synthese modelise :

- l'objet de phase et d'amplitude,  t = exp(i phi - tau_opt/2), construit a
  partir de l'OPL et de l'absorption projetees ;
- le porteur spatial du Nomarski, a 45 degres, de frequence reglable ;
- la PUPILLE de l'objectif : troncature du spectre a NA/lambda. C'est ici
  qu'entre la diffraction, absente de la projection en rayons droits ;
- l'echantillonnage de la camera : pixel de 3.45 um ramene a 0.345 um dans
  l'objet par le grandissement, et integration sur la surface du pixel ;
- le bruit de photons et le bruit de lecture.

Ce qu'elle ne modelise pas : la propagation de la sonde A TRAVERS le milieu
(elle traverse un objet mince), les aberrations, et la depolarisation du
Wollaston. La deviation par la lentille de Kerr n'y est donc pas non plus.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["CameraSide", "SIDE_CAM", "synthesize_interferogram",
           "write_virtual_shot", "roundtrip_check"]


@dataclass
class CameraSide:
    """Voie laterale de l'interferometre, valeurs de abel_phase_explorer."""
    n_px: int = 2048
    pixel_cam_um: float = 3.45
    magnification: float = 10.0
    NA: float = 0.23
    lambda_probe_nm: float = 515.0
    rotation_deg: float = -135.3       # applique au DEPOUILLEMENT
    # Periode du porteur, en pixels camera. NE PAS mettre au hasard : la
    # bande laterale doit sortir du lobe d'ordre zero (rayon 2 r_f) et rester
    # sous Nyquist -- les deux conditions de Kimura rappelees en section 11 du
    # rapport. Avec NA=0.23, M=10, 515 nm et 2048 px, r_f = 316 bins, donc la
    # periode doit tomber entre 2.0 et 3.75 px. A 8 px la bande laterale se
    # noyait dans l'ordre zero et le filtre de Takeda n'en recuperait que la
    # moitie. `check_carrier()` verifie la condition.
    fringe_period_px: float = 3.0
    photons_per_count: float = 500.0   # pour le bruit de grenaille
    read_noise_counts: float = 2.0
    bias_counts: float = 100.0
    full_well_counts: float = 4000.0

    @property
    def pixel_object_um(self):
        return self.pixel_cam_um / self.magnification

    @property
    def carrier_cycles_per_px(self):
        return 1.0 / self.fringe_period_px

    def check_carrier(self, verbose=True):
        """Conditions de separation des ordres, Kimura / section 11 du rapport.

            sqrt(3) r_f < nu0        la bande laterale sort de l'ordre zero
            nu0 + sqrt(2) r_f < sqrt(2) nu_N   elle reste echantillonnable

        avec r_f = NA / (M lambda), le rayon du spectre de l'objet.
        """
        N = self.n_px
        r_f = (self.NA / (self.magnification * self.lambda_probe_nm * 1e-3)) \
              * (N * self.pixel_cam_um)
        nu_N = N / 2.0
        nu0 = N / self.fringe_period_px
        lo, hi = np.sqrt(3) * r_f, np.sqrt(2) * (nu_N - r_f)
        ok = lo < nu0 < hi
        if verbose and not ok:
            print(f"  (!) porteur hors conditions de separation : nu0 = {nu0:.0f} "
                  f"bins, il faut {lo:.0f} < nu0 < {hi:.0f}, soit une periode "
                  f"entre {N/hi:.2f} et {N/lo:.2f} px. Le depouillement de "
                  f"Takeda melangera l'ordre zero et la bande laterale.")
        return dict(ok=ok, nu0=nu0, r_f=r_f, nu0_min=lo, nu0_max=hi,
                    period_min_px=N / hi, period_max_px=N / lo)


def _object_maps(res, delay_fs, cam, x_half_um, **probe_kw):
    """OPL [nm] et profondeur optique sur la grille (z, x) de la simulation."""
    from figures_filament import probe_opl_transmittance
    d = probe_opl_transmittance(res, delay_fs,
                                lambda_probe_m=cam.lambda_probe_nm * 1e-9,
                                x_half_um=x_half_um, **probe_kw)
    T = np.clip(np.asarray(d["transmittance"], float), 1e-12, None)
    return d["z_um"], d["x_um"], np.asarray(d["opl_nm"], float), -np.log(T)


def synthesize_interferogram(res, delay_fs, cam: CameraSide = None,
                             x_half_um=70.0, z_origin_px=None,
                             add_noise=True, seed=0, apply_pupil=True,
                             **probe_kw):
    """Renvoie (side_sig, side_bg), deux images n_px x n_px comme la camera.

    `side_bg` est l'interferogramme de reference, sans excitation : c'est lui
    que le depouillement divise pour eliminer le porteur et les defauts fixes.
    """
    cam = cam or SIDE_CAM
    cam.check_carrier()
    rng = np.random.default_rng(seed)
    N = cam.n_px
    p = cam.pixel_object_um
    lam_um = cam.lambda_probe_nm * 1e-3

    z_um, x_um, opl_nm, tau_opt = _object_maps(res, delay_fs, cam,
                                               x_half_um, **probe_kw)

    # --- placer l'objet dans le champ, oriente pour que la rotation du
    #     depouillement (-135.3 deg) le remette droit ---
    ii = (np.arange(N) - N / 2) * p
    XX, YY = np.meshgrid(ii, ii, indexing="ij")
    th = np.deg2rad(-cam.rotation_deg)
    zc = z_origin_px if z_origin_px is not None else 0.0
    Z = XX * np.cos(th) - YY * np.sin(th) + zc
    R = XX * np.sin(th) + YY * np.cos(th)

    def _interp2(field):
        # interpolation bilineaire de field(z, x) aux points (Z, R)
        iz = np.interp(Z, z_um, np.arange(len(z_um)), left=np.nan, right=np.nan)
        ix = np.interp(R, x_um, np.arange(len(x_um)), left=np.nan, right=np.nan)
        ok = np.isfinite(iz) & np.isfinite(ix)
        out = np.zeros_like(Z)
        if not ok.any():
            return out
        i0 = np.clip(np.floor(iz[ok]).astype(int), 0, len(z_um) - 2)
        j0 = np.clip(np.floor(ix[ok]).astype(int), 0, len(x_um) - 2)
        fz, fx = iz[ok] - i0, ix[ok] - j0
        out[ok] = ((1 - fz) * (1 - fx) * field[i0, j0]
                   + fz * (1 - fx) * field[i0 + 1, j0]
                   + (1 - fz) * fx * field[i0, j0 + 1]
                   + fz * fx * field[i0 + 1, j0 + 1])
        return out

    phi = 2.0 * np.pi * _interp2(opl_nm) / cam.lambda_probe_nm
    tau = np.clip(_interp2(tau_opt), 0.0, None)
    obj = np.exp(1j * phi - 0.5 * tau)

    # --- pupille de l'objectif : c'est ici qu'entre la diffraction ----------
    if apply_pupil:
        kx = 2 * np.pi * np.fft.fftfreq(N, d=p)
        KX, KY = np.meshgrid(kx, kx, indexing="ij")
        pup = (KX**2 + KY**2) <= (2 * np.pi * cam.NA / lam_um) ** 2
        obj = np.fft.ifft2(np.fft.fft2(obj) * pup)
        ref = np.fft.ifft2(np.fft.fft2(np.ones_like(obj)) * pup)
    else:
        ref = np.ones_like(obj)

    # --- porteur du Nomarski, a 45 degres dans le repere camera -------------
    n_px_idx = np.arange(N)
    IX, IY = np.meshgrid(n_px_idx, n_px_idx, indexing="ij")
    carrier = np.exp(1j * 2 * np.pi * cam.carrier_cycles_per_px
                     * (IX + IY) / np.sqrt(2.0))

    def _frame(o):
        amp = 0.5 * cam.full_well_counts
        I = np.abs(o + carrier) ** 2 * amp / 2.0 + cam.bias_counts
        if add_noise:
            I = rng.poisson(np.clip(I, 0, None) * cam.photons_per_count) \
                / cam.photons_per_count
            I = I + rng.normal(0.0, cam.read_noise_counts, I.shape)
        return np.clip(I, 0.0, None).astype(np.float32)

    return _frame(obj), _frame(ref)


SIDE_CAM = CameraSide()


def write_virtual_shot(res, delay_fs, out_path, cam=None, **kw):
    """Ecrit un npz au format des tirs de la manip (side_sig / side_bg).

    Le fichier produit se depouille avec preprocess_side_full comme un vrai.
    """
    sig, bg = synthesize_interferogram(res, delay_fs, cam, **kw)
    out = Path(out_path)
    np.savez_compressed(out, side_sig=sig, side_bg=bg)
    print(f"-> {out.name} ({out.stat().st_size/1e6:.1f} Mo), "
          f"delai {delay_fs:+.0f} fs")
    return out


def roundtrip_check(res, delay_fs=0.0, cam=None, x_half_um=70.0,
                    tmpdir="/tmp", verbose=True, **probe_kw):
    """Verifie que le depouillement retrouve la phase injectee.

    Synthetise un interferogramme depuis la simulation, le passe dans
    `preprocess_side_full` -- le code qui depouille les VRAIES images -- et
    compare la phase reconstruite a celle qu'on avait mise. Tout ecart est
    imputable au depouillement lui-meme, pas au modele : c'est precisement le
    biais que la comparaison directe Delta n contre Delta n ignorait.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "web"))
    from abel_phase_explorer import preprocess_side_full

    cam = cam or SIDE_CAM
    f = Path(tmpdir) / "virtual_shot.npz"
    write_virtual_shot(res, delay_fs, f, cam, x_half_um=x_half_um, **probe_kw)

    z_um, x_um, opl_nm, _ = _object_maps(res, delay_fs, cam, x_half_um,
                                         **probe_kw)
    phi_in = 2.0 * np.pi * opl_nm / cam.lambda_probe_nm

    phase, s_um_per_px, rec_NA = preprocess_side_full(str(f),
                                                      cam.lambda_probe_nm)
    if verbose:
        print(f"depouillement : carte {phase.shape}, "
              f"{s_um_per_px:.3f} um/px, NA_rec = {rec_NA:.3f}")
        print(f"phase injectee    : {phi_in.min():+.4f} a {phi_in.max():+.4f} rad")
        print(f"phase reconstruite: {np.nanmin(phase):+.4f} a "
              f"{np.nanmax(phase):+.4f} rad")
        print(f"amplitude in {np.ptp(phi_in):.4f} rad, "
              f"out {np.nanmax(phase)-np.nanmin(phase):.4f} rad, "
              f"rapport {(np.nanmax(phase)-np.nanmin(phase))/max(np.ptp(phi_in),1e-12):.2f}")
    return dict(phase_out=phase, phi_in=phi_in, s_um_per_px=s_um_per_px,
                z_um=z_um, x_um=x_um, file=f)
