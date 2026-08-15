# filament_1030nm_4uJ — de quoi lancer le notebook

Pompe 1030 nm, sonde 515 nm (interféromètre Nomarski, croisement à 90°),
4 µJ incidents, waist en z = 0, boîte 0 → 350 µm.

## Marche à suivre

1. Dézipper, **sans renommer les dossiers** : le notebook cherche `sim/` et
   `web/` en remontant l'arborescence depuis le répertoire courant.
2. Lancer Jupyter depuis `filament_4uJ/notebooks/`.
3. Exécuter la **cellule d'installation** (la 2ᵉ). Elle pose numpy / scipy /
   matplotlib côté CPU, puis lit `nvidia-smi` et choisit `cupy-cuda12x` ou
   `cupy-cuda11x`.
4. **Redémarrer le noyau** si elle a installé cupy.
5. Exécuter les cellules dans l'ordre.

`run_z0_probe_sweeps.py`, à la racine du zip, est la version script (non
notebook) du même run 4 µJ, avec en plus le balayage HTML Abel à trois
longueurs d'onde de sonde (490/620/690 nm). À lancer depuis la racine du zip
(il remonte l'arborescence pour trouver `sim/`/`web/` comme le notebook).

## Deux niveaux de dépendances

| | pour quoi | fichiers |
|---|---|---|
| CPU seul | tout le post-traitement : relire un `result.npz`, refaire les figures, exporter la courbe | `figures_filament`, `figures_report`, `permittivity`, `virtual_experiment`, `export_curves`, `synthetic_interferogram`, `pump_probe_0d` |
| GPU (cupy) | **lancer** un calcul, `filament_sim.run()` | `filament_sim`, `config`, `grids`, `operators`, `integrator`, `kernels` |

Sans GPU, le notebook s'arrête à la cellule de lancement mais tout le reste
fonctionne sur un `result.npz` déjà calculé.

## Sorties

Le run écrit dans `notebooks/runs_z0/` (ou `run_z0_probe_sweeps.py` écrit
dans `runs_z0_probe_sweeps/`) :

- `z0_350um_4uJ/result.npz` + `params.json` — le calcul (~1 Go)
- `figures/` — planches OPL et transmittance, série de délais
- `figures_rapport/rep_*.png` — les sept figures du rapport
- **`curve_4uJ.csv`** — la courbe δφ(τ), quelques ko : **c'est ce fichier
  qu'il faut partager**
- `result_small.npz` — le run réduit aux cubes utiles au post-traitement
- `runs_z0_probe_sweeps/html/abel_phase_explorer_*_probe_{490,620,690}nm.html`
  — pages autonomes (Plotly), une par sonde, avec panneau densités
  électrons/STE en plus du panneau de phase

## Géométrie de la sonde (important)

La manip croise la pompe à **90°** (incidence normale, imagerie Abel
classique) — pas les 10° de Martin et al. 1997, dont la mesure est
ponctuelle et spatialement moyennée. `sim/pump_probe_0d.py::USER_SIO2_1030`
porte `cross_angle_deg=90.0` en conséquence ; sa fonction reste de vérifier
des ordres de grandeur et des rapports entre canaux, pas de reproduire une
carte (z, x) comparable pixel à pixel à la manip — c'est le pipeline Abel
(`figures_filament.py`, `virtual_experiment.py`, `web/abel_phase_explorer.py`)
qui fait ça.
