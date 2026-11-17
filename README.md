# TP Cartographie et localisation avec le lidar LD06

Louis Delfosse, EIGSI La Rochelle. Module « Cartographie et localisation appliquée à la robotique mobile ».

Chaîne complète en Python : acquisition des scans du lidar LD06, nuages de points, grid map, carte vectorielle, localisation par scan matching (sur grid map, sur carte vectorielle, entre scans), évaluation chiffrée, SLAM (grid map et carte vectorielle) évalué sur les scans réels et en simulation à pose connue.

Le compte rendu détaillé (choix, résultats, limites) est dans `rapport_TP_lidar_LD06.html`.

## Installation

```
pip install numpy pandas matplotlib pyserial pytest
```

Python 3, testé sous Windows avec VS Code. `pyserial` n'est nécessaire que pour `acquisition.py`.

## Conventions

- Une **pose** est un triplet `(x en m, y en m, θ en degrés)` dans le repère de la carte.
- Le lidar fournit des distances en **mm** ; tout le reste du code est en **mètres**.
- Passage polaire → cartésien : `x = d·cos(a)`, `y = −d·sin(a)` (le LD06 tourne dans le sens horaire).
- Un scan est d'abord exprimé dans le repère du robot (lidar en 0, 0), puis transformé dans le repère de la carte avec sa pose.
- Les 7 positions de mesure sont `pos_A.csv` à `pos_G.csv` (dossier `data/`), 3 tours chacune. Les évaluations n'utilisent que le premier tour.

## Organisation du dépôt

### Jour 1 : extraction, visualisation, cartographie

| Fichier | Rôle |
|---|---|
| `acquisition.py` | Lit le lidar sur le port série (`COM7`, 115 200 bauds), enregistre 3 tours dans un CSV. Utilise `tours.py` (détection d'un nouveau tour). |
| `tours.py` | Détecteur de tour : cumule l'angle parcouru (les angles du LD06 décroissent) et signale un tour complet à 360°. |
| `viewer.py` | Affiche plusieurs scans en couleurs ; sliders pour régler la pose de chaque scan à la main ; sauvegarde dans `data/poses_ajustees.csv`. |
| `voir_positions.py` | Trace les positions de mesure A à G. |
| `gridmap.py` | Filtrage des mesures (`scan_local`), création et mise à jour de la grid map (Bresenham : cellules traversées = libres, cellule d'impact = occupée). |
| `carte_globale.py` | Grid map globale à partir des 7 scans et de leurs poses ajustées. |
| `segments.py` | Outils sur segments : angle, proximité, fusion, intersection, découpe (avec tolérances numériques). |
| `extraction.py` | Nuage de points → segments (groupes, découpe récursive, ajustement par moindres carrés). |
| `carte_vectorielle.py` | Construit la carte vectorielle à partir de plusieurs scans, écrit `data/carte_vectorielle.txt`. |
| `test_segments.py` | 18 tests pytest sur `segments.py`, dont les cas limites demandés par le sujet. |

### Jour 2 : localisation (scan matching)

| Fichier | Rôle |
|---|---|
| `localisation_grid.py` | Localisation d'un scan sur une grid map : recherche exhaustive, puis multi-résolution (affinage à 1 cm / 0,25° avec score interpolé). |
| `localisation_scan.py` | Localisation d'un scan par rapport à un scan de référence (4 paires de positions proches). |
| `localisation_vecto.py` | Localisation d'un scan sur la carte vectorielle (distance des points aux segments) : mêmes deux méthodes. Mode `verite` pour le test de cohérence. |
| `test_verite_terrain.py` | Test de cohérence sur grid map : tous les scans sont dans la carte, décalages aléatoires. |
| `verif_loc.py` | Vérification visuelle d'une localisation (référence en bleu, pose trouvée en rouge). |
| `bilan.py` | Tableau récapitulatif des trois types de référence (erreurs, temps, taux de réussite à 5 et 10 cm) → `data/bilan_localisation.csv`. |

### Jour 3 : SLAM

| Fichier | Rôle |
|---|---|
| `slam.py` | SLAM sur grid map et sur carte vectorielle, évalué sur les 7 scans réels. Contient la composition de poses, l'odométrie bruitée, l'a priori d'odométrie. |
| `simulation.py` | Salle simulée + lidar virtuel : scans à pose **exacte**. Expérience principale et étude de sensibilité à la qualité de l'odométrie. |
| `test_slam.py` | 7 tests pytest (poses, odométrie, a priori, scan simulé). |

## Utilisation, dans l'ordre

```
python acquisition.py data/pos_A.csv        # une fois par position (robot branché)
python viewer.py data/pos_*.csv             # régler les poses, bouton "Sauver poses"
python voir_positions.py
python gridmap.py data/pos_A.csv            # grid map d'un seul fichier
python carte_globale.py                     # grid map des 7 scans
python extraction.py data/pos_A.csv 0       # segments d'un tour
python carte_vectorielle.py                 # carte vectorielle globale
python -m pytest                            # tous les tests (25)

python localisation_grid.py                 # écrit data/resultats_grid.csv
python localisation_scan.py                 # écrit data/resultats_scan_scan.csv
python localisation_vecto.py                # écrit data/resultats_vecto.csv
python localisation_vecto.py verite         # test de cohérence
python test_verite_terrain.py
python verif_loc.py pos_B.csv
python bilan.py                             # écrit data/bilan_localisation.csv

python slam.py 20                           # SLAM réel, 20 tirages d'odométrie (~7 min)
python simulation.py 5                      # simulation à pose connue, 5 tirages (~12 min)
python simulation.py sens 12                # sensibilité à l'odométrie, 12 tirages (~7 min)
```

Les scripts doivent être lancés depuis la racine du dépôt (ils lisent `data/`). Les tirages de bruit utilisent des graines fixes : les résultats sont reproductibles.

## Fichiers de données (`data/`)

- `pos_A.csv` … `pos_G.csv` : scans bruts (`scan_id, t_ms, angle_deg, dist_mm, intensity`).
- `poses_ajustees.csv` : poses réglées à la main avec `viewer.py` (`fichier, x_m, y_m, theta_deg`).
- `poses_mesurees.csv` (facultatif) : poses mesurées au mètre ruban, même format. Si ce fichier existe, il sert de référence pour les scans qu'il contient. **Il n'existe pas dans ce dépôt : aucune pose n'a été mesurée au mètre ruban.**
- `carte_vectorielle.txt` : segments `x1,y1,x2,y2`.
- `resultats_*.csv`, `bilan_localisation.csv` : résultats des évaluations du jour 2.
- `slam_poses.csv`, `carte_salle_slam.txt` : poses estimées et carte vectorielle de la salle produites par le SLAM (tirage 0).
- `resultats_slam.csv`, `resultats_slam_simu.csv`, `sensibilite_odometrie.csv` : résultats du jour 3.

## Évaluation du jour 2

- Poses initiales décalées de la référence : `(dx, dy, dθ)` ∈ {(0,0,0), (0,2 m,0), …, (0,8 m,0)}. Le dernier décalage sort volontairement de la fenêtre de recherche (±0,50 m, ±15°) ; `bilan.py` l'écarte du bilan.
- Succès : erreur de position < 10 cm et erreur angulaire < 2°.
- La référence est la pose ajustée à la main (voir « Limites »). Le temps de calcul est mesuré pour chaque localisation.

## SLAM (jour 3)

### Principe

Pour chaque nouveau scan `k` :

1. **Prédiction** : `pose_init` = pose estimée de `k-1` composée avec le déplacement relatif fourni par l'odométrie.
2. **Recalage** : le scan est localisé sur la carte construite jusqu'ici (recherche exhaustive puis multi-résolution, comme au jour 2).
3. **Mise à jour** : le scan, à la pose recalée, est ajouté à la carte (grid map : `mettre_a_jour` ; carte vectorielle : absences, extraction, fusion).

La première pose est fixée (origine du repère). Il n'y a **pas de fermeture de boucle ni de relocalisation globale** : une erreur de recalage plus grande que la fenêtre de recherche (±0,50 m, ±15°) n'est pas rattrapée.

### Odométrie simulée

Le robot a été déplacé à la main entre les 7 positions : il n'y a pas d'odométrie réelle. Elle est **simulée** : déplacement relatif entre poses consécutives (poses ajustées pour les scans réels, vérité exacte en simulation) auquel on ajoute un bruit gaussien, `σ = σ_fixe + glissement × distance` par axe, plus un bruit de cap. Réel : 2 cm + 3 %, 1° par pas. Simulation : 1 cm + 5 %, 0,8° par pas.

### A priori d'odométrie (grid map)

Le recalage d'un scan le long d'un mur droit est mal contraint : le score est presque plat et son maximum peut tomber n'importe où dans la fenêtre. Sans correction, le SLAM sur grid map perd la trajectoire (une erreur de recalage de quelques dizaines de cm pollue la carte, et tout dérive ensuite). On pénalise donc le score par la distance de Mahalanobis à la pose prédite :

`score − κ · N · (écart / σ_pas)²`, avec `σ_pas = 2 cm + 5 % × distance du pas`, `κ = 0,07`, `N` = nombre de points du scan.

La pénalité est faible quand l'odométrie est incertaine (grands pas) et forte quand elle est fiable (petits pas). `κ` a été choisi en balayant {0,03 ; 0,07 ; 0,15} sur la simulation et sur les scans réels : **ce réglage est donc optimisé sur les mêmes données que celles du bilan, les résultats avec a priori sont un peu optimistes.**

### Résultats : 7 scans réels (20 tirages d'odométrie)

Référence = poses ajustées à la main. Ordre des scans : E, A, G, B, D, F, C (plus proche voisin). Succès par pose : erreur < 10 cm et < 2°.

| Méthode | Erreur position moyenne | Erreur angulaire moyenne | Erreur dernière pose | Poses réussies | Temps par scan |
|---|---|---|---|---|---|
| Odométrie seule | 34,8 cm | 1,09° | 47,8 cm | 6 % | – |
| SLAM grid map, sans a priori | 9,9 cm | 0,47° | 12,1 cm | 84 % | 0,10 s |
| SLAM grid map, avec a priori | 9,7 cm | 0,46° | 12,2 cm | 87 % | 0,10 s |
| SLAM carte vectorielle | 5,5 cm | 0,37° | 7,7 cm | 100 % | 1,81 s |

Figure : `illustation_avancement/slam_reel.png`.

### Résultats : simulation à pose connue (46 poses, 5 tirages)

Vérité terrain exacte. Salle simulée = contour de la vraie salle (dimensions mesurées) + deux obstacles ; lidar virtuel : 450 faisceaux, bruit 1,5 cm, 3 % de faisceaux sans retour.

| Méthode | Erreur position moyenne | Erreur angulaire moyenne | Erreur dernière pose | Poses réussies | Temps par scan |
|---|---|---|---|---|---|
| Odométrie seule | 40,7 cm | 3,43° | 57,4 cm | 6 % | – |
| SLAM grid map (avec ou sans a priori) | 2,7 cm | 0,15° | 2,6 cm | 100 % | 0,07 s |
| SLAM carte vectorielle | 3,2 cm | 0,29° | 3,6 cm | 98 % | 2,33 s |

Figures : `illustation_avancement/slam_simulation.png`, `slam_derive.png`.

### Sensibilité à la qualité de l'odométrie (grid map, 12 tirages par niveau)

Pourcentage de **trajectoires entières** réussies (toutes les poses sous 10 cm et 2°). Glissement : σ position = 1 cm + glissement × distance.

| Glissement | 3 % | 5 % | 10 % | 15 % | 20 % |
|---|---|---|---|---|---|
| Sans a priori | 67 % | 83 % | 75 % | 67 % | 33 % |
| Avec a priori | 92 % | 100 % | 100 % | 83 % | 58 % |

Les échecs sont brutaux : une fois la carte polluée, la trajectoire est perdue. La courbe n'est pas strictement monotone (12 tirages seulement par niveau).

## Limites

- **Pas de vérité terrain mesurée sur le robot.** Les erreurs « réelles » sont mesurées par rapport aux poses ajustées à la main avec `viewer.py`, dont l'incertitude est de l'ordre de quelques cm (sur les scans réels, le score de la pose trouvée dépasse presque toujours celui de la pose ajustée). Les erreurs de 5 à 10 cm sur données réelles incluent donc cette incertitude. Les chiffres exacts viennent de la simulation.
- **Odométrie simulée**, pas mesurée (voir plus haut).
- **Pas de fermeture de boucle** ni de relocalisation : un décalage initial supérieur à la fenêtre de recherche n'est pas rattrapé (sur les scans réels, un tirage sur 20 perd toute la trajectoire du SLAM sur grid map dès le premier pas, avec 55 à 62 cm d'erreur ; le SLAM sur carte vectorielle rattrape ce même tirage).
- La carte vectorielle de la salle (`data/carte_salle_slam.txt`) a des trous dans les murs : l'étape « absence de détection » coupe les segments traversés par un rayon, ce qui retire aussi des morceaux de murs vus de loin sous un angle rasant.
- La salle simulée n'est pas la vraie salle.

## Reste à faire

- Finaliser le compte rendu (jour 3, limites) et la conclusion personnelle.
- Bonus facultatif : déplacer le robot automatiquement et cartographier en temps réel.
