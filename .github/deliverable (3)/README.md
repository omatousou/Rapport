# Filamentation femtoseconde SiO2 -- étude d'ablation des termes physiques

Ce dépôt étend le solveur de filamentation (split-step Hankel/FFT + RK4 +
kernel CUDA pour les équations de taux de porteurs, dans l'esprit de
Couairon, Sudrie, Franco, Prade, Mysyrowicz, PRB 71, 125435 (2005)) avec des
interrupteurs indépendants sur chaque terme de l'équation de propagation du
champ, pour pouvoir répondre à "qu'est-ce que ce terme fait réellement au
faisceau ?" par une boucle d'ablation plutôt qu'en le lisant dans le code.

## Structure

- `sim/filament_sim.py` -- solveur (basé sur `NewSim3juillet.py`), avec six
  interrupteurs sur l'équation de propagation du champ (éq. 3) :
  `enable_kerr_instantaneous`, `enable_kerr_raman`, `enable_self_steepening`,
  `enable_photoionization_loss`, `enable_plasma_defocusing`,
  `enable_plasma_absorption`. Les interrupteurs porteurs déjà existants
  (`enable_avalanche`, `enable_recombination`, `enable_ste`) restent
  disponibles. Chaque run écrit `result.npz` **et** `params.json` (optique
  sonde + quels interrupteurs étaient actifs), ce dernier étant nouveau par
  rapport à `NewSim3juillet.py` et nécessaire pour `abel_phase_explorer.py`.

- `notebooks/term_ablation_study.ipynb` -- notebook interactif :
  1. cases à cocher (`ipywidgets`) pour lancer une configuration à la main ;
  2. boucle automatique qui lance `full` (tout activé), un `no_<terme>` par
     terme désactivé isolément, et `linear_only` (les six désactivés), en
     reprenant les runs déjà présents sur disque ;
  3. reproduction des figures fournies : intensité crête vs z (fig. 8),
     densité électronique on-axis vs z (fig. 9), électrons libres vs piégés
     + intensité du pulse à z fixé (fig. 13), contours de fluence + FWHM/2
     vs z (fig. 7), superposés par scénario pour comparer l'effet de chaque
     terme ;
  4. tableau récapitulatif comparatif ;
  5. export vers la page web Abel-transform.

- `web/abel_phase_explorer.py` -- génère une page HTML autonome (Plotly,
  ouverte localement dans un navigateur) qui calcule le déphasage
  interférométrique Δφ(z, x) par transformée d'Abel forward de Δn, avec un
  canal séparé et cochable pour Drude (électrons libres), Kerr (n2·I), STE
  (excitons auto-piégés) et thermique (chauffage phonons au piégeage), plus
  un sélecteur de scénario pour comparer les runs de la boucle d'ablation.
  Garde le pipeline de reconstruction holographique expérimentale de
  `unified_filament_slider_v3.py` intact et réutilisable (case `raw_dir`) ;
  fonctionne en mode simulation seule si les fichiers bruts ne sont pas
  fournis.

## Important : non exécuté sur GPU

Cet environnement de rédaction n'a ni GPU ni `cupy`. Le solveur
(`sim/filament_sim.py`) n'a donc pas pu être exécuté avec le vrai kernel CUDA
d'équations de taux. Chaque interrupteur a été vérifié par relecture
attentive et par un test unitaire séparé (mock `cupy`/`cupyx` en NumPy pur)
qui exerce `NonlinearOperator.split()` pour les six configurations
d'ablation et confirme :
- que désactiver les six termes à la fois donne un terme non-linéaire
  strictement nul (propagation purement linéaire) ;
- que chaque interrupteur modifie effectivement la sortie attendue (phase
  `rhs` et/ou perte `alpha`) sans провoquer de valeurs non-finies ;
- que `enable_self_steepening=False` force l'opérateur `T-hat` à l'identité ;
- que la chaîne complète `filament_sim.run()` -> `result.npz` + `params.json`
  -> `abel_phase_explorer.build_explorer_html()` s'exécute sans erreur.

Le pipeline complet (boucle d'ablation notebook + kernel CUDA réel + page
web) reste à valider sur une machine avec GPU avant une étude de production
sur grille fine.

## Architecture du solveur (`sim/`)

Le solveur était un fichier unique de ~1000 lignes ; il est découpé en modules
à responsabilité unique, avec `filament_sim.py` comme **point d'entrée** (le
« main ») qui les assemble et expose `run()`.

```
keldysh.py      taux de photoionisation de Keldysh + ses limites analytiques,
                dispersion de Sellmeier. Aucune dépendance interne, pas de
                cupy -> exécutable seul : `python sim/keldysh.py` lance la
                suite de validation.
config.py       dataclass Config (tous les paramètres) + code_fingerprint()
kernels.py      noyau CUDA des équations de taux des porteurs (éq. 6-7)
grids.py        grilles Hankel/temps/fréquence, dispersion, LUT, enveloppes
operators.py    demi-pas linéaire + terme non linéaire (éq. 3)
integrator.py   buffers d'enregistrement, marche en z, écriture result.npz
filament_sim.py POINT D'ENTREE : run(), FIELD_TOGGLES, et ré-export de tout
```

Dépendances (acycliques, vérifiées) :

```
keldysh, kernels   (feuilles)
config      -> keldysh
grids       -> keldysh, config
operators   -> config, kernels
integrator  -> config, grids, operators
filament_sim -> tous
```

`filament_sim.py` ré-exporte l'ensemble de la surface publique, donc
`from filament_sim import run, Config, ...` continue de fonctionner à
l'identique : **le notebook n'a pas eu à changer**. Le découpage a été vérifié
numériquement neutre (22 tableaux comparés avant/après, écart relatif max
0.000e+00, bit à bit identique).

`code_fingerprint()` hache désormais **tous** les fichiers sources, pas
seulement deux : éditer n'importe quel module invalide les `result.npz` en
cache.
