# Martin et al. 1997 : la recette, l'audit du code, le plan

Référence : P. Martin, S. Guizard, Ph. Daguzan, G. Petite, P. D'Oliveira,
P. Meynadier, M. Perdrix, *Subpicosecond study of carrier trapping dynamics in
wide-band-gap crystals*, Phys. Rev. B **55**, 5799 (1997).

---

# Partie 1 — La recette

## 1.1 Ce qui est fait, physiquement

Une pompe intense crée des paires électron-trou dans un diélectrique à grand
gap. Une sonde faible, retardée, mesure la modification de la fonction
diélectrique. Le signal passe par trois régimes successifs :

| | origine | signe | échelle de temps |
|---|---|---|---|
| 1 | Kerr, `∝ I_pompe` | + | durée de recouvrement pompe-sonde |
| 2 | électrons de conduction | − | apparaît à la fin de l'impulsion |
| 3 | excitons auto-piégés | + | monte en `τ`, persiste |

## 1.2 Comment la mesure est faite

**Interférométrie dans le domaine spectral**, pas Nomarski. C'est une
différence de principe avec la manip du dépôt.

- Ti:Sa 790 nm, 120 fs, 2 mJ, 20 Hz.
- Pompe doublée à **395 nm** (`ħω = 3.14 eV`), énergies 2–22 µJ.
- Diamètre de pompe **à 1/e** : 34, 44 ou 58 µm → `2×10¹¹` à `4×10¹²` W/cm².
- Sonde à **618 nm** ou 790 nm.
- Un Michelson fabrique **deux impulsions sonde identiques** séparées de
  12 ou 18 ps. La première passe **avant** la pompe (référence), la seconde
  **après**. Elles interfèrent à la sortie d'un spectromètre.
- Spectre : `I(ω) = I₀(ω)[1 + T + 2√T·cos(ωΔt + ΔΦ)]`. La distorsion des
  franges donne `ΔΦ`, la perte de contraste donne `T`.
- Axe horizontal du capteur = longueur d'onde ; axe vertical = **la coordonnée
  transverse réelle**, résolution ~1 µm. Une transformée de Fourier **ligne par
  ligne** donne `ΔΦ(r)` et `T(r)`.
- Échantillons : quartz, 500 µm d'épaisseur, pureté < 1 ppm.
- Moyenne sur **3 tirs**, échantillon déplacé après chaque tir.
- Sensibilité : **2×10⁻² rad** en phase, **±5 %** en absorption.

## 1.3 Comment les données des graphes sont produites

C'est le point que j'avais raté, et il est décisif.

**Géométrie** : la sonde croise la pompe à **10°**. Pour chaque coordonnée
transverse `r`, les Eqs. (3) et (4) sont **intégrées le long du trajet de la
sonde** à travers le profil gaussien de pompe. Il n'y a pas de longueur `L`
unique : la longueur effective dépend du canal.

Sur l'axe, avec une pompe de 52 µm de diamètre à 1/e :

| canal | dépendance | longueur effective |
|---|---|---|
| Kerr | `I¹` | 265 µm |
| plasma, STE | `I⁴` | 133 µm |

**Puis** chaque point des courbes temporelles (Figs. 4, 6, 7) est obtenu en
**intégrant spatialement sur `r`** le profil `ΔΦ(r)` du type de la Fig. 3.

Les deux intégrations combinées donnent exactement

```
signal(canal en I^K)  ∝  π w² / (K sin θ)
```

soit un facteur **`1/K`**. Un canal en `I⁴` est donc pesé **4 fois moins** que
le Kerr, par pure géométrie.

`ΔΦ(r,t) = (2π L / λ)·Δn(r,t)`, `L ≈ 300 µm` — mais ce `L` est une commodité
d'exposé, pas ce qui est utilisé dans l'ajustement.

## 1.4 Le modèle d'indice, Sec. III

Milieu non excité :
```
ε₁(ω) = 1 + (N₀e²/mε₀)·f₁₂/(ω₁₂² − ω² + iω/τ₁₂) = n₀²
```

Milieu excité, **Eq. (2)** :
```
ε₂(ω) = 1 + (e²/mε₀)(N₀ − N_CB − N_tr)·f₁₂/(ω₁₂² − ω² + iω/τ₁₂)
          + (e²/ε₀)[ N_CB f_CB/m* · 1/(ω² + iω/τ_e-p)
                   + N_tr f_tr/m  · 1/(ω_tr² − ω² + iω/τ_tr) ]
          + χ³_eff E_p²
```

Observables, **Eqs. (3) et (4)** :
```
ΔΦ = (2π/λ)·L·[Re√ε₂ − Re√ε₁]
A  = 1 − exp[−(2Lω/c)·Im√ε₂]
```

Approximations que l'article énonce lui-même :
- contribution **électronique seule** ; les trous sont négligés car
  `m_trou ≫ m_e` — invalide si la sonde est résonante avec une transition de
  trou piégé, ce qui n'est pas le cas ici ;
- `f₁₂` et `ω₁₂` sont des **paramètres effectifs** représentant la somme de
  toutes les transitions ;
- les bandes STE sont des **raies uniques**, `f_tr` est « an adjustable
  effective parameter » ;
- le développement au premier ordre (leur Eq. 5) n'est donné **que pour
  l'exposé** : l'ajustement utilise les Eqs. (2)–(4) *in extenso*, condition
  `N₀ ≫ N_CB` remplie car `ρ < 10¹⁹ cm⁻³` chez eux ;
- l'hypothèse « l'indice ne varie pas pendant l'impulsion sonde » est
  **explicitement invalide entre 0 et 200 fs**, et ils l'assument.

## 1.5 Les valeurs, SiO₂ (Table II)

| grandeur | symbole | valeur |
|---|---|---|
| indice non linéaire | `n₂` | 2×10⁻¹⁶ cm²/W |
| densité initiale | `N₀` | 2.2×10²² cm⁻³ |
| ordre multiphotonique | `n` | 4 |
| section efficace | `σ⁽⁴⁾` | 2.3×10⁻¹¹⁴ |
| force d'oscillateur CB | `f_CB` | 1 |
| masse effective CB | `m*` | **0.5 mₑ** |
| collisions électron-phonon | `1/τ_e-p` | **1.5×10¹⁵ s⁻¹** (soit 0.67 fs) |
| temps de piégeage | `τ` | **150 fs** |
| forces d'oscillateur des pièges | `f_tr` | **0.4 et 0.15** |
| énergies des pièges | `ω_tr` | **5.2 et 4.2 eV** |
| largeurs des pièges | `1/τ_tr` | **1.5 et 1 eV** |

Équations cinétiques SiO₂, **Eq. (8)** — pas d'avalanche :
```
dN_CB/dt = N₀ σ⁽⁴⁾ F⁴ − N_CB/τ
dN_tr/dt = N_CB/τ
```

Faits mesurés à retenir :
- gap du quartz **10 eV**, 4 photons à 3.14 eV = 12.5 eV ;
- passage au positif à **550 fs**, identique à 3 et 4 TW/cm² alors que les
  densités diffèrent d'un facteur 5 — c'est la preuve d'une cinétique
  **non séquentielle** ;
- absorption à 618 nm : forte au début (collisions e-phonon), **< 10 %** aux
  temps longs, « indicating that the probe photon energy is far from the
  resonances of the trapping center ».

> ⚠️ La transcription `.tex` fournie en début de discussion porte
> « ~150 fs » pour ce passage au positif. Le PDF dit **550 fs**. 150 fs est le
> temps de piégeage `τ`, une grandeur différente.

---

# Partie 2 — Audit du code

## 2.1 Chaîne du solveur (GPU, inchangée)

| fichier | rôle |
|---|---|
| `config.py` | tous les paramètres + `code_fingerprint` |
| `keldysh.py` | taux de photoionisation, Sellmeier |
| `grids.py` | grilles Hankel/FFT, `σ_ω`, coefficient d'avalanche, préfacteur STE |
| `kernels.py` | noyau CUDA des équations de taux |
| `operators.py` | opérateurs linéaire et non linéaire, Eq. (3) de Couairon |
| `integrator.py` | RK4 + Strang, écriture de `result.npz` et `params.json` |
| `filament_sim.py` | `run()`, façade |

Sort : cubes `(z, r, t)` de `rho_e`, `rho_STE`, `I`.

## 2.2 Chaîne de post-traitement sonde

| fichier | rôle | conforme à la recette ? |
|---|---|---|
| `permittivity.py` | `ε(ω)` complexe, 4 canaux, `n = √ε` | **oui**, Eq. (2) |
| `figures_filament.probe_opl_transmittance` | coupe oblique + Abel + Δn, α | géométrie **Nomarski**, pas celle de l'article |
| `virtual_experiment.sample_as_experiment` | instrument : NA, convolution sonde, bruit | lit **un pixel**, pas d'intégration spatiale |
| `pump_probe_0d.py` | modèle 0D pour reproduire la Fig. 6 | `L` **unique**, moyenne spatiale ad hoc |
| `synthetic_interferogram.py` | interférogramme brut + vrai dépouillement | Nomarski, correct pour la manip |
| `export_curves.py` | réduction npz, extraction de la courbe mesurée | — |

## 2.3 Ce que chaque écart produit

Comparaison avec Mouskeftaras Fig. 1 (mêmes trois repères) :

| | mesuré | simulé | facteur |
|---|---|---|---|
| creux/pic | −0.219 | −0.738 | 3.4 |
| plateau/creux | −0.429 | −0.032 | 13 |
| écart pic→creux | 250 fs | 450 fs | 1.8 |

**Le facteur 3.4 est géométrique, pas physique.** La double intégration
(trajet + espace) pèse un canal en `I^K` d'un facteur `1/K`. Avec `K = 4` :

```
mon creux/pic ponctuel          -0.738
corrigé de la géométrie (×1/4)  -0.185
mesuré                          -0.219      écart résiduel 16 %
```

**Le facteur 13 sur plateau/creux est physique.** STE et plasma suivent tous
deux `I⁴`, donc leur rapport est insensible à la géométrie. Il ne reste que le
couple `(f_STE, m*)`.

**Le facteur 1.8 sur le timing est instrumental** : sonde de 263 fs contre une
structure de 250 fs. Non résoluble en l'état.

---

# Partie 3 — Plan de modification

Ordonné par dépendance. Chaque étape a son test.

### Étape 1 — Géométrie de l'article dans `pump_probe_0d`

Remplacer `L` unique + `spatial_average` ad hoc par la vraie double
intégration : trajet incliné à `θ` à travers une pompe gaussienne, puis
intégration sur `r`.

*Test* : le rapport des intégrales spatiales entre un canal `I¹` et un canal
`I⁴` doit valoir exactement `1/4`, et `π w²/(K sin θ)` analytiquement.

### Étape 2 — Recalage des paramètres sur la Table II

`τ = 150 fs` (et non 330), pas d'avalanche, `K = 4`, `σ⁽⁴⁾`, `m* = 0.5`,
`τ_e-p = 0.67 fs`, sonde 618 nm, pompe 395 nm.

*Test* : `N_CB` sous 10¹⁹ cm⁻³, passage au positif à **550 fs** aux deux
intensités.

### Étape 3 — Calibrer `f_STE` sur `plateau/creux`

Seul paramètre encore libre une fois 1 et 2 faits. Contrainte :
`plateau/creux = 0.43`. Ajouter une fonction qui inverse cette relation et
renvoie le jeu de bandes.

*Test* : la Fig. 6 reproduite doit donner le bon signe **et** le bon rapport
aux deux intensités.

### Étape 4 — Reporter le résultat sur la manip

Une fois `f_STE` calibré sur une mesure, l'appliquer à la géométrie **Nomarski**
(Abel, sonde transverse) — qui reste la bonne pour le dépôt.

*Test* : `compare_to_measurement` contre la courbe δφ(τ) extraite des frames.

### Étape 5 — Intégration spatiale optionnelle côté `virtual_experiment`

Ajouter la lecture intégrée sur une fenêtre en `r`, en plus de la lecture
ponctuelle, pour pouvoir comparer aux deux conventions.

---

## Ce qui reste ouvert

- **Longueur d'onde sonde de la figure Mouskeftaras.** `f_eff` varie d'un
  facteur 3 entre 490 et 790 nm ; la cible de l'étape 3 en dépend.
- **`τ_c`** : trois calibrations concurrentes, 0.67 / 1.7 / 10 fs.
- **`ρ_e` du solveur** reste ~10× au-dessus de ce que la transmittance
  implique — problème du terme source, indépendant de tout ce qui précède.
