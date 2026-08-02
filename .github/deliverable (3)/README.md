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
