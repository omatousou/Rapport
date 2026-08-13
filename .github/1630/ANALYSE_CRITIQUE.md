# Analyse critique du solveur vis-à-vis de Couairon et al., PRB 71, 125435 (2005)

Document de travail. Chaque affirmation est soit vérifiée numériquement (la
commande est donnée), soit explicitement marquée comme non tranchée.

Références utilisées :

| clé | référence |
|---|---|
| **[C05]** | A. Couairon, L. Sudrie, M. Franco, B. Prade, A. Mysyrowicz, *Filamentation and damage in fused silica induced by tightly focused femtosecond laser pulses*, PRB **71**, 125435 (2005) |
| **[M04]** | S.S. Mao, F. Quéré, S. Guizard, X. Mao, R.E. Russo, G. Petite, P. Martin, *Dynamics of femtosecond laser interactions with dielectrics*, Appl. Phys. A **79**, 1695 (2004) |
| **[S]** | Sakurai et al., *Quantum Electronics* (modélisation ablation/piégeage, table de paramètres silice) |

---

## 1. Convention SI vs gaussienne — la question posée

L'hypothèse était que [C05] écrit son équation de propagation en unités
gaussiennes alors que le code travaille en SI. **Vérification faite : il n'y a
pas de conversion gaussienne à faire.** [C05] utilise une convention
*normalisée en intensité*, pas gaussienne, et la traduction déjà présente dans
le code est correcte terme par terme.

La preuve est dans l'article lui-même : Sec. III écrit « …and intensity
`E₀² = 2P_in/πw₀²` », donc **|𝓔|² EST l'intensité** chez eux. C'est cohérent
avec `k₀n₂|𝓔|²` qui doit être un nombre d'onde : avec n₂ en m²/W, il faut bien
que |𝓔|² soit en W/m².

Le code, lui, propage un champ `u` en V/m et reconstruit l'intensité par
`I = invE2·|u|²` avec `invE2 = ½n₀cε₀·10⁻⁴`. Les trois termes non linéaires ont
été revérifiés analytiquement :

**Kerr.** `kerr_pref = 3χ⁽³⁾ω₀²/(8kc²)` avec `χ⁽³⁾ = (4/3)ε₀n₀²c·n₂`, donc
`kerr_pref = ε₀n₀n₂ω₀/2`. En substituant `|u|² = 2I/(ε₀n₀c)` :

```
kerr_pref·|u|² = (ε₀n₀n₂ω₀/2)·2I/(ε₀n₀c) = n₂ω₀I/c = k₀n₂I     ✔ identique à [C05] Eq. (4)
```

**Plasma.** `σ` du code = `k e²τ_c/(n₀²mε₀ω₀(1+ω₀²τ_c²))`, à comparer à
[C05] Eq. (5) `σ = ke²/(n₀²ω₀²ε₀m)·ω₀τ_c/(1+ω₀²τ_c²)` — algébriquement la même
expression ✔. Et le terme de défocalisation `(σ/2)(iω₀τ_c)ρ` se réduit bien,
pour ω₀τ_c ≫ 1, à la forme standard `kρ/(2n₀²ρ_c)` avec `ρ_c = ε₀mω₀²/e²` ✔.

**Perte par photoionisation.** `photo = W_PI·U_i/(n₀cε₀|u|²) = W_PI·U_i/(2I)` ✔
(unités m⁻¹).

**Conclusion :** aucun facteur 4π ni ε₀ manquant. Cette piste est fermée.

---

## 2. Grandeurs vérifiées numériquement

Toutes reproduites à mieux que 1 % sans ajustement :

| grandeur | article | code | commande |
|---|---|---|---|
| n₀(800 nm) | 1.45 | 1.4533 | Sellmeier |
| k″(800 nm) | 361 fs²/cm | 361.6 | dérivée 2ᵈᵉ de Sellmeier |
| P_cr | 1.98 MW | 1.980 | λ₀²/(2πn₀n₂) |
| z_f (Rayleigh) | 5.7 µm | 5.71 | πw_f²n₀/λ₀ |
| w(d) entrée | 13.2 µm | 13.18 | w_f√(1+d²/z_f²) |
| énergie du pulse | 1.1 µJ | 1.0998 | intégration de la fluence |
| γ à I = 3.5e13 | 1 | 1.000 | ω₀√(mU_i)/(eE) |
| W_MPI à 3.5e13 | 3.7e34 | 3.71e34 | σ₆I⁶ρ_at |
| Fig. 7 : >1 J/cm², 0.45 µJ | 45 → 90 µm | 44.5 → 92.5 | `fluence_level_extent` |
| Fig. 7 : >1 J/cm², 1.1 µJ | 25 → 110 µm | 27.3 → 110.5 | `fluence_level_extent` |

Les deux dernières lignes sont les seuls chiffres que [C05] donne explicitement
sur la Fig. 7 (Sec. V), et ce sont les plus contraignants : ils testent
simultanément l'auto-focalisation, le clamping et l'équilibre Kerr/plasma.

---

## 3. Corrections apportées

### 3.1 Waist du faisceau : 1.1 → 1.0 µm  *(impact fort)*

`PAPER_PARAMS` utilisait `w0 = 1.1e-6`, la valeur **mesurée** de la Table I pour
l'objectif 20×. Mais la Sec. III dit « w_f = 1 µm is the beam waist,
z_f = πw_f²n₀/λ₀ = 5.7 µm » — et 5.7 µm n'est compatible qu'avec 1.0 µm (avec
1.1 µm on obtient 6.9 µm). Les **simulations** de l'article utilisent donc
1.0 µm. Comme I_pic ∝ 1/w_f², c'était ~20 % d'erreur sur l'intensité.

### 3.2 Normalisation en énergie  *(impact 6.4 %)*

`Config` calculait `I₀ = 2E/(πw₀²·FWHM)`, alors que [C05] pose
`P_in = E_in/(t_p√(π/2))` avec `t_p = FWHM/√(2ln2)`. Le rapport
`t_p√(π/2)/FWHM = 1.0644` : le faisceau portait 6.4 % d'énergie en trop.

*Note :* [C05] se contredit sur ce point. Les étiquettes de la Fig. 3 (7.81 MW
à 1.25 µJ, 6.88 MW à 1.1 µJ, …) correspondent toutes à `P = E/FWHM`
(rapport constant 6.25e12 W/J sur les six énergies), pas à la formule de la
Sec. III (6.46 MW à 1.1 µJ). J'ai gardé la Sec. III, seule version qui conserve
l'énergie. L'écart sur P/P_cr (3.26 vs 3.47) déplace le foyer non linéaire de
0.13 µm via Marburger, donc c'est sans conséquence sur la position.

### 3.3 Opérateur T̂ : T̂¹ → T̂² sur le Kerr

[C05] Eq. (4) met `T̂²` sur le crochet Kerr mais seulement `T̂¹` sur la perte par
photoionisation. Le code partageait un unique `T̂¹` pour les deux. Séparé en
deux transformées.

### 3.4 Filtre spectral : bande ω ≤ 0  *(nouveau, impact numérique fort)*

`T̂ = 1 + Ω/ω₀` vaut exactement `ω/ω₀`, donc **négatif** partout où la fréquence
absolue est négative : 25 % de la grille à Nt = 2048, **38 % à Nt = 4096**.
Raffiner la grille temporelle aggrave le problème — signature d'un opérateur
mal posé, pas d'une sous-résolution. Et depuis le passage à `T̂²`, cette région
non physique était amplifiée jusqu'à ×9 par pas (×25 à Nt = 4096) et réinjectée.

Ajout d'un masque spectral à bords tanh limitant le champ à λ ∈ [0.18, 5] µm,
soit exactement la fenêtre où le fit de Sellmeier (donc `delta_k`) est défini —
c'est déjà le domaine auquel `omega_safe` bornait la dispersion. Vérifié :
|T̂| dans la bande ω ≤ 0 chute de 1.011 à 1.5e-5, le côté bleu légitime
(ω/ω₀ jusqu'à 3.0, soit 266 nm) reste intact. Piloté par
`enable_spectral_filter` (défaut `True`) ; désactivé → comportement identique
à l'ancien, donc A/B possible.

### 3.5 Taux de Keldysh — fausse piste, corrigée. Le taux reste celui d'origine.

**Ce que j'avais cru trouver.** Le taux d'ionisation *décroît* quand l'intensité
augmente, en 23 points entre 1e11.5 et 1e14.2 W/cm², pentes log-log jusqu'à
**−136**, avec des décrochements à 4.5e12, 3.2e13, **6.6e13** et 1.0e14 W/cm².
L'intensité crête des runs étant 5.95e13 et 7.22e13, celui de 6.6e13 tombe entre
les deux. J'en avais conclu à un bug de transcription de l'Eq. (8) et remplacé
l'argument de Dawson `Φ(√(β(n+2ν)))` par `Φ(√(2β(n+ν)))`, qui rend bien W
continu aux fermetures de canal.

**C'était faux.** La limite multiphotonique de Keldysh tranche : quand γ → ∞,

```
β = π²/(4K(Ξ)E(Ξ)) → π²/(4(π/2)²) = 1        (vérifié : 1.00000 à γ=100)
exp(-α)             → e²/(16γ²)              (vérifié : rapport 0.99995)
```

Avec β → 1, le terme n=0 de la forme **publiée** vaut `Φ(√(2ν))` — c'est
exactement le taux multiphotonique de Keldysh du manuel. Ma variante « continue »
donnait `Φ(√ν)`, incompatible avec cette limite. Mesuré contre
`W_MPI = σ₆I⁶ρ_at` à 1e12 W/cm² : forme publiée **1.018**, ma variante 0.872.

**Donc les cusps sont physiques** : ce sont les fermetures de canal. À chaque
passage de x par un entier l'ordre photonique monte d'une unité, et à
l'intérieur d'une branche le terme n=0 décroît en `√ν` quand ν → 0 (cusp de
seuil en racine carrée), ce qui peut dépasser la croissance de `exp(-α⟨x+1⟩)` et
faire baisser W localement.

**État actuel.** L'argument publié `(n+2ν)` est restauré, et le taux est
**numériquement identique à la version d'origine** (écart relatif max 6.5e-13
sur 1e11–1e15 W/cm², vérifié contre une réimplémentation propre de l'ancienne
formule). Rien n'a changé sur la Fig. 2. Ce qui est ajouté :

- `monotone` (défaut **False**) : enveloppe monotone optionnelle. C'est une
  **régularisation numérique, pas une correction de physique** — un taux avec
  des décrochements d'un facteur plusieurs dans la plage 5–7e13 où le filament
  clampe rend le clamping sensible au côté du cusp où tombe une maille. À
  activer seulement pour *tester* si les cusps affectent un run donné.
- `beta_den` (défaut **2**, valeur historique du code) : l'article signale
  lui-même l'ambiguïté — « our quantity β is divided by 4, whereas the
  corresponding quantity in Ref. 34 is divided by 2 ». Les deux diffèrent d'un
  facteur ~1.5 : W(3.5e13) = 4.30e32 (β_den=2) contre 6.44e32 (β_den=4).
- `validate()` : vérifie la convention de sommation *structurellement* (β → 1 et
  exp(−α) → e²/16γ²), pas seulement par comparaison de valeurs.

**Validation.** `python sim/keldysh.py` :

```
beta a gamma=100               : 1.00000   [-> 1]
exp(-alpha) / [e^2/16gamma^2]  : 0.99995   [-> 1]
W/W_MPI a 5e11/1e12/2e12       : 1.062 / 1.018 / 0.911   [-> 1, article Sec. III]
ordre photonique a basse I     : 5.89                    [-> 6]
gamma a 3.5e13 W/cm2           : 1.000                   [article: 1]
W(3.5e13), beta_den=2 / =4     : 4.30e+32 / 6.44e+32     [article cite 1.6e32]
points decroissants, brut      : 23  (pente min -136.49)
```

**Leçon.** Le test « W → σ₆I⁶ρ_at » seul ne suffisait pas à trancher (0.872 vs
1.018, les deux « proches de 1 » sur un graphe log à 10 décades). Ce sont les
asymptotes **structurelles** de Keldysh (β → 1, exp(−α) → e²/16γ²) qui
départagent, parce qu'elles contraignent la forme de la somme et pas seulement
sa normalisation.

### 3.6 Résonance Lorentz du STE : 5.2 → 4.2 eV  *(impact ×1.84 sur le déphasage)*

[M04] — l'article même dont `figures_article.py` reproduit les figures — écrit

```
Δn = N_STE e²/(2 n₀ m ε₀) · 1/(ω_tr² − ω²)
```

et précise que ω_tr est « the resonance frequency of the STE's **first excited
level (~4.2 eV in SiO2)** ». Les 5.2 eV largement cités sont le **sommet de la
bande d'absorption** du STE (Fig. 12 de [M04] : « rise time of the 5.2 eV
absorption band »), mesuré en absorption transitoire — ce n'est pas la résonance
à utiliser pour le changement d'**indice**.

À 490 nm (2.53 eV), `f_STE = ω²/(ω_tr²−ω²)` passe de 0.310 à 0.570, soit un
**facteur 1.84** sur la contribution STE au déphasage sonde. La formule elle-même
est déjà exactement celle de [M04], vérifié par substitution de
`ρ_c' = ε₀mω'²/e²`.

Signe cohérent : ω' < ω_tr donc Δn > 0, ce qui correspond bien au déphasage
positif observé dans SiO₂ par [M04] (Fig. 10).

### 3.7 Canal de décroissance du STE (nouveau, optionnel)

[S] tabule pour la silice un temps de piégeage `t_tr = 150 fs` **et** un temps de
relaxation `t_r = 1 ps`. Le code n'avait pas ce second canal : une fois piégés,
les STE ne repartaient que par ré-ionisation laser. Ajout de
`Config.tau_ste` (défaut `None` = ancien comportement) qui insère `−ρ_s/τ_ste`
dans le noyau CUDA. Pertinent surtout pour le déphasage sonde aux délais
picoseconde.

### 3.8 Polarisabilité liée du STE dans l'équation de champ  *(nouveau)*

Un STE n'est **pas** un porteur de bande de conduction : c'est une paire de
liaisons pendantes (`≡Si•` / `•O−Si≡`, rupture homolytique du pont Si−O−Si)
dont les orbitales non appariées restent **dans le gap**. Conséquence pour un
modèle à deux populations :

| | état | mobile | Drude ? |
|---|---|---|---|
| ρ_e | bande de conduction | oui | oui, σ(τ_c, m*) |
| ρ_s | liaisons pendantes, dans le gap | non (hopping) | **non** |

**Ce qui était déjà correct.** `Integrator.step()` ne passait que `self.rho` à
`split()` ; `self.rho_s` n'y entrait jamais. Donc l'absorption plasma
(`alpha += plasma_pref·ρ`) et la défocalisation
(`−plasma_pref·(plasma_phase−1)·ρ·u`) ne voient que ρ_e. Les STE ne
contribuaient déjà ni au bremsstrahlung inverse ni au `−ρ/2ρ_c` ✔.

**Ce qui manquait.** Les STE polarisent quand même, via une résonance **liée**
au premier niveau excité (§3.6) :

```
dn_STE = + [ω²/(ω_tr² − ω²)] · ρ_s/(2 n₀ ρ_c)
```

Ce terme n'existait que dans le post-traitement, et **uniquement pour la sonde
à 490 nm** : la pompe ne le voyait jamais pendant la propagation. Or à 800 nm
(1.55 eV, loin sous 4.2 eV) le facteur vaut f_STE = 0.1576 et

| ρ_s | Δn_STE à 800 nm | vs Δn_Kerr(5e13) = 1.77e-2 |
|---|---|---|
| 1e20 | +3.11e-3 | 18 % |
| 5e20 | +1.56e-2 | 88 % |

Signe **positif** (ω < ω_tr) : ça s'ajoute à l'auto-focalisation Kerr — et c'est
physiquement le changement d'indice permanent que [C05] identifie comme dommage
de type I (modification d'indice sans fracture).

Ajouté comme terme de **phase pure, sans perte** (la pompe à 1.55 eV est très
loin de la bande d'absorption STE, donc pas d'absorption à un photon à
compter), piloté par `enable_ste_index` (défaut `True`).

Vérifications : `ste_pref` retombe exactement à 0 si `enable_ste=False` **ou**
`enable_ste_index=False`, donc la reproduction de [C05] (`enable_ste=False`) est
inchangée ; `Δn_STE(ρ_s=1e20)` reconstruit depuis `ste_pref` donne 3.1131e-3
contre 3.113e-3 attendu analytiquement ; et `rho_s=0` est un no-op bit-à-bit
face à l'ancien appel à deux arguments.

---

## 4. Le modèle STE du code est celui de Mao et al.

Confirmation utile : les équations implémentées dans le noyau CUDA sont
exactement celles de [M04] (Sec. 1.1.3) :

```
[M04]  dN/dt     = aIN + σ N I^m + σ_x N_STE I^{m_x} − N/τ_x
       dN_STE/dt = −σ_x N_STE I^{m_x} + N/τ_x
```

```
code   dρ_e/dt = W_PI·depl + β_g I ρ_e depl + (W_STE + β_s I ρ_e)(ρ_s/ρ_at) − ρ_e/τ_r
       dρ_s/dt = ρ_e/τ_r − (W_STE + β_s I ρ_e)(ρ_s/ρ_at) − ρ_s/τ_ste
```

terme par terme : MPI ↔ `W_PI`, avalanche `aIN` ↔ `β_g I ρ_e`, ré-ionisation
STE `σ_x N_STE I^{m_x}` ↔ `(W_STE + β_s I ρ_e)(ρ_s/ρ_at)`, piégeage `−N/τ_x`
↔ `−ρ_e/τ_r`. La seule généralisation est le remplacement de la loi de puissance
à `m_x` photons par un taux de Keldysh au gap `U_s` — plus cohérent, puisque le
canal valence utilise déjà Keldysh.

**Point de vocabulaire important :** le `τ_r = 150 fs` que [C05] appelle
« recombinaison » et le `t_tr = 150 fs` que [M04]/[S] appellent
« self-trapping » sont **le même processus**. [C05] le dit d'ailleurs :
« plasma recombination is dominated by fast trapping of carriers into localized
states below the band gap ». Les deux littératures convergent sur 150 fs.

---

## 5. Écarts subsistants, non corrigés

### 5.1 Intensité et densité hors bande

Sur les runs de production (après §3.1–3.4, **avant** §3.5–3.7) :

| | article | 0.45 µJ | 1.1 µJ |
|---|---|---|---|
| I_max | 5 ± 0.5e13 W/cm² | 5.95e13 | 7.22e13 |
| ρ_e max | 2–4e20 cm⁻³ | 7.49e20 | 1.04e21 |
| pertes | ~40–50 % (Fig. 12) | 11.1 % | 18.0 % |

I_max croît bien avec l'énergie, donc **pas de plafond artificiel** — c'était
une inquiétude que j'avais soulevée à tort. Mais les trois grandeurs sont hors
bande dans le même sens : trop d'intensité, trop d'électrons, pas assez
d'absorption. **Ces chiffres datent d'avant la correction Keldysh §3.5**, qui
modifie le taux précisément dans la plage 5–7e13 : elle doit être remesurée
avant toute autre conclusion. C'est l'action suivante.

### 5.2 Coefficient d'avalanche : tension entre les deux littératures

Avec `τ_c = 1e-14 s` (valeur que [C05] ajuste sur ses propres mesures de
transmission), `β_g = σ/U_i = 0.912 cm²/J`. Or [S] tabule `a = 4 cm²/J` pour la
silice — un facteur **4.4** d'écart. Avec `τ_c = 1.7e-15 s` (défaut du code pour
ton expérience) on obtient 5.06 cm²/J, cohérent avec [S].

Non corrigé : on reproduit [C05], qui a déterminé τ_c = 1e-14 sur ses données.
Mais cela explique en partie des pertes trop faibles (avalanche faible → moins
de plasma → moins d'absorption Bremsstrahlung), et c'est une piste sérieuse
pour §5.1.

### 5.3 Masse dans σ : ambiguïté de l'article

[C05] Eq. (5) écrit `σ = ke²/(n₀²ω₀²ε₀m)…` sans préciser quel `m`. Le seul `m`
défini dans l'article est la masse réduite 0.64 mₑ. Mais leur `ρ_c ≡ ω₀²ε₀m*/e²
= 1.74e21 cm⁻³` n'est numériquement correct qu'avec `m* = mₑ` (vérifié). Le code
utilise `mₑ` (`meff_drude_rel = 1.0`), ce qui est cohérent avec ρ_c. Prendre
0.64 mₑ multiplierait σ par 1.56 (plus d'absorption **et** plus de
défocalisation, donc clamping plus bas — la direction voulue pour §5.1). Le
paramètre est exposé : `meff_drude_rel` est testable directement.

### 5.4 Termes de l'Eq. (2) non implémentés — quantifiés, jugés négligeables

L'Eq. (2) complète est `Û ∂Ê/∂z = i[∇⊥²/2k + (Û + L̂/2k)L̂]Ê + iN`, soit après
division par Û :

```
∂Ê/∂z = i[−ρ²/(2kÛ) + L̂ + L̂²/(2kÛ)]Ê + iN/Û
```

- `L̂²/(2kÛ)` **absent**. Rapport au terme L̂ conservé : `L̂/2k ≈ 2e-7`
  (L̂ ≈ 5.4 m⁻¹ à la largeur spectrale du pulse, k = 1.139e7 m⁻¹). Négligeable
  même avec un élargissement spectral ×10. Ne pas toucher.
- `1/Û` sur le terme non linéaire **absent**. Û = 1 + (n_g/n₀)(Ω/ω₀) s'écarte de
  1 de 0.7 % à la largeur spectrale nominale, mais atteint 2.0 à Ω/ω₀ = 1. Donc
  négligeable sur le cœur du spectre, réel sur les ailes fortement décalées. À
  garder en tête si l'élargissement spectral devient important.
- Facteur de déplétion `(1−ρ/ρ_max)` **présent dans le code, absent de l'Eq. (6)
  de [C05]**. À ρ ≤ 4e20 contre ρ_at = 2.1e22 il vaut ≥ 0.98 : ~2 %. Conservé
  comme régularisation (il empêche ρ > ρ_at), écart documenté.

### 5.5 Énergie de ré-ionisation des STE non prélevée sur le champ

Le terme de perte du champ (`photo`) n'utilise que `W_PI` (bande de valence) ;
`W_STE` n'apparaît que dans le noyau CUDA. Autrement dit la ré-ionisation des
STE **crée des porteurs sans coûter d'énergie au faisceau**. C'est une
inconsistance de bilan énergétique, plus petite que §5.1 (elle ne joue que là où
ρ_s est déjà substantiel) mais réelle. Correction propre : ajouter
`W_STE·U_s/(2I)·(ρ_s/ρ_at)` à `alpha`. Non fait pour l'instant — à traiter après
avoir remesuré §5.1, pour ne pas mélanger deux changements de bilan d'énergie
dans la même mesure.

### 5.6 W_PI : facteur ~2.7 sur une valeur ponctuelle

[C05] cite `W_PI = 1.6e32 cm⁻³s⁻¹` à I = 3.5e13 ; le code donne 4.3e32 (forme
brute). Aucune des deux conventions sur β que l'article mentionne lui-même
(« notre β est divisé par 4, celui de Keldysh par 2 ») ne redonne 1.6e32 :
j'obtiens 4.30e32 et 6.44e32. Comme l'asymptote multiphotonique est, elle,
correcte à ~13 % — test bien plus contraignant — je considère la normalisation
validée et le 1.6e32 comme probablement relevé sur leur propre figure log.
À noter que [C05] signale que σ₆ varie de 1.5e-71 à 3e-67 dans la littérature,
soit 4 ordres de grandeur : un facteur 2.7 n'est pas discriminant.

---

## 6. Différences avec Bulgakova, Stoian & Rosenfeld

*« Laser-induced modification of transparent crystals and glasses »* modélise la
même physique (silice, 800 nm, focalisation en volume) mais avec un jeu
d'équations et de paramètres sensiblement différent. C'est donc un **second test
indépendant** du solveur, et non une redite. Leur Eq. (19) est la NLSE, (21) le
taux de porteurs.

### 6.1 Écarts sur l'équation de propagation

| | [C05] Eq. (2)/(4) | Bulgakova Eq. (19) |
|---|---|---|
| couplage espace-temps | `Û(Ω) ∂/∂z` avec `Û = 1 + k₁Ω/k₀` | `T̂⁻¹` devant la diffraction |
| opérateur de choc | `T̂²` sur le Kerr, `T̂¹` sur la perte PI | `T̂` sur le Kerr, `T̂⁻¹` sur le plasma |
| gap d'ionisation | `U_i = 9 eV` **fixe** | `E_g = E_g0 + e²𝓔²/(2cn₀ε₀m_rω₀²)` = **E_g0 + U_p** |
| déplétion | absente de l'Eq. (6) | `(n_a/n_lat)` présente |
| dénominateur avalanche | `U_i` | `(1 + m_r/mₑ)E_g` = **1.5 E_g** |

Les deux premières lignes sont des façons différentes d'écrire la même
correction au premier ordre en Ω/ω₀ ; elles ne diffèrent qu'au second ordre.

**La ligne qui compte est le gap pondéromoteur.** Chiffré :

| I (W/cm²) | U_p | E_g = 9 + U_p | écart |
|---|---|---|---|
| 1e13 | 0.82 eV | 9.82 eV | +9 % |
| 3e13 | 2.47 eV | 11.47 eV | +27 % |
| 5e13 | 4.11 eV | 13.11 eV | +46 % |
| 7e13 | 5.76 eV | 14.76 eV | +64 % |

À l'intensité de clamping, Bulgakova ionise donc à travers un gap ~45 % plus
grand que Couairon. Comme W_PI ~ I^K avec K ≈ E_g/ħω₀, cela **abaisse
fortement** le taux à haute intensité et donc l'électron produit. Le solveur
suit [C05] (gap fixe) : un écart est attendu et **normal** sur leurs figures à
haute intensité, ce n'est pas un bug.

### 6.2 Écarts sur les paramètres matériau

| | [C05] | Bulgakova |
|---|---|---|
| n₂ | 3.54e-16 cm²/W | 2.48e-16 |
| m_r | 0.64 mₑ | 0.5 mₑ |
| ω₀τ_c | 23.6 (τ_c = 10 fs) | **3** |
| n_lat / ρ_at | 2.1e22 cm⁻³ | 6.6e22 |
| τ piégeage | 150 fs | 150 fs ✔ (même valeur) |
| w / d / durée | 1.0 µm / 75 µm / 160 fs | 0.9 µm / 90 µm / 120 fs |

`ω₀τ_c` diffère d'un facteur ~8, ce qui rebascule complètement l'arbitrage
absorption / défocalisation plasma (cf. §5.2) : à ω₀τ_c = 3 l'absorption plasma
est bien plus forte qu'à 23.6.

### 6.3 Taux multiphotonique : les deux σ₆ ne sont pas comparables

- [C05] : `W = σ₆I⁶ρ_at` avec σ₆ = 9.6e-70 s⁻¹cm¹²/W⁶ → **W = 2.02e-47·I⁶**
- Bulgakova (table) : σ₆ = 6e8 cm⁻³ps⁻¹(TW/cm²)⁻⁶ → **W = 6.00e-52·I⁶**

soit un rapport **3.4e4**. Une partie vient du gap pondéromoteur (leur K
effectif n'est pas 6), une autre du fait que leur table concerne l'ULE et non la
silice pure. À ne pas utiliser comme point de comparaison directe — [C05]
signale d'ailleurs que σ₆ court sur 4 ordres de grandeur dans la littérature.

### 6.4 Ce que Bulgakova ajoute et que [C05] n'a pas

Photoémission électronique et champ électrostatique auto-cohérent, explosion
coulombienne, modèle à deux températures et transport thermique, déformations
élastoplastiques. Hors du périmètre de ce solveur, qui s'arrête au dépôt
d'énergie.

### 6.5 Reproduction des Figs. 11 et 12

Implémentées : `fig11_bulgakova` et `fig12_bulgakova` dans `figures_article.py`,
pilotées par la section « 1bis » du notebook (`BULGAKOVA_PARAMS`).

Trois grandeurs ont dû être enregistrées en plus dans le solveur, toutes
locales en (r, z) :

- `absorbed_rz` — densité d'énergie absorbée `∫2αI dt` (J/cm³), α venant de
  `NonlinearOperator.loss_rates` (MPI + bremsstrahlung inverse) → Fig. 11b
- `absorbed_rz_bins` — la même chose sur 4 tranches temporelles
  (`absorb_time_bins_fs`) → Fig. 12
- `Ipeak_rz` — intensité crête **par rayon** (`Imax_z` était le max sur r *et* t,
  une autre grandeur) → Fig. 11c
- `rho_rz_at` — densité électronique à un instant donné (`rho_snapshot_t_fs`,
  +50 fs chez eux), là où `rho_rz` stocke le max sur le temps → Fig. 11d

Vérifié : la somme des 4 tranches redonne le total à 1e-7 près (arrondi float32),
`rho_rz_at ≤ rho_rz` partout, et `max(Ipeak_rz) = max(Imax_z)`.

Niveaux de contour repris de leurs figures : fluence 0.20/0.80/1.4/1.9 J/cm²,
énergie absorbée 50/600/1200 J/cm³, intensité 2e12/7e12/3e13 W/cm²,
densité 1e15/1e17/1e19/3e20 cm⁻³ ; z ∈ [0, 150] µm, r ∈ [−5, +5] µm, trait blanc
au foyer géométrique (90 µm).

---

## 7. Que faire ensuite

1. **Vider le cache et relancer** — obligatoire : `load_scenario_npz` relit un
   `result.npz` sans vérifier la version du code, donc un run en cache masque
   tous les changements ci-dessus. `run_health_check` affiche désormais les
   interrupteurs lus dans `params.json` pour détecter ce cas.
   ```python
   import shutil
   for d in ("couairon2005_1p1uJ", "couairon2005_0.45uJ", "couairon2005_1uJ"):
       shutil.rmtree(OUT_ROOT / d, ignore_errors=True)
   ```
   Puis **redémarrer le kernel** (sinon `filament_sim` déjà importé reste en
   mémoire).

2. **Remesurer §5.1.** Attention : le taux de Keldysh est finalement
   **inchangé** (§3.5 était une fausse piste, corrigée). Les changements qui
   agissent réellement sur le clamping sont le waist (§3.1), la normalisation
   en énergie (§3.2), T̂² (§3.3) et le filtre spectral (§3.4).

3. **Si I_max et ρ_e restent hauts**, tester dans cet ordre — chaque paramètre
   est déjà exposé :
   - `meff_drude_rel = 0.64` (§5.3) → σ ×1.56, clamping plus bas
   - `tau_c` (§5.2) → arbitre absorption vs défocalisation
   - `KeldyshSiO2(..., beta_den=4)` (§3.5) → W ×1.5, la convention que
     l'article dit utiliser
   - `monotone=True` (§3.5) → dit si les cusps de fermeture de canal jouent
   - convergence : `Nt` 2048 → 4096, puis `Nr` 3000 → 6000

4. **Ne pas ajuster n₂, P_cr, k″, w_f, l'énergie** : tous vérifiés à mieux que
   1 % (§2). Si un écart persiste, il n'est pas là.
