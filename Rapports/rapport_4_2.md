# Rapport §4.2 — Modélisation et comparaison d'architectures
## Projet CastagNet (MSc BIHAR, ESTIA — MESPR)

*Ce rapport fait suite au rapport §4.1 (qualité et complétude de la donnée), qui documente la constitution du jeu d'images d'entraînement (`img_dataset_exam`) et la décision d'abandonner la correspondance T/B sur le dataset labellisé au profit d'une architecture à deux modèles indépendants.*

---

## 1. Stratégie retenue avant entraînement

### 1.1 Deux modèles indépendants (T et B)

Conséquence directe de la décision prise au §4.1 : aucune correspondance fiable entre les images T et B du dataset labellisé n'a pu être établie dans le temps imparti. Deux modèles sont donc entraînés séparément, chacun sur les images d'une seule position de caméra, chacun capable de classifier indépendamment Conforme / NON Conforme / PIETRA.

### 1.2 Traitement des images AMBIGU

Les images classées AMBIGU par le détecteur de vide (§4.1) sont **incluses** dans le jeu d'entraînement, au même titre que les non-vides. Ce choix s'appuie sur l'analyse contre la vérité terrain humaine (rapport §4.1) : 97,2 % des images AMBIGU sont en réalité de vraies châtaignes — des fragments limites pour le détecteur colorimétrique de vide, mais pas nécessairement problématiques pour un modèle de classification. Seul le vide franc (statut `True`) est exclu, destiné à un lot **« repasse »** distinct (collecte et modèle spécifique à envisager ultérieurement, hors périmètre de ce rapport).

### 1.3 Trois classes cibles, pas deux

Le modèle est entraîné sur les trois classes prévues par le cahier des charges (Conforme / NON Conforme / PIETRA), malgré un déséquilibre modéré (facteur ~2,3 entre la classe majoritaire et les deux autres, cf. rapport §4.1). Une fusion en deux classes (Conforme / Autre) a été envisagée puis écartée : elle équilibrerait artificiellement les classes mais ferait disparaître une distinction potentiellement utile (PIETRA correspond à une destination commerciale réelle — la bière corse du même nom — pas seulement à un rebut). Les métriques Conforme-vs-Autre exigées par le cahier des charges restent de toute façon calculables a posteriori à partir de la matrice de confusion d'un modèle 3 classes, sans coût supplémentaire.

### 1.4 Contrainte matérielle : entraînement et déploiement séparés

L'entraînement s'effectue sur un Mac Apple M1 Pro (16 Go de mémoire unifiée), sans contrainte mémoire significative. La contrainte forte du cahier des charges (GPU de production GTX 1060, 3 Go de VRAM, partagés avec la gestion de 12 flux caméra Ethernet) ne s'applique qu'au modèle **déployé**, pas à l'entraînement. Par prudence, un budget mémoire de déploiement d'environ 1,5 Go a été retenu comme hypothèse de travail (partage avec la gestion des flux caméra), sans qu'il soit possible de le vérifier précisément à ce stade — seule une mesure sur le matériel cible ou une machine équivalente (prévue au §4.3) pourra le confirmer.

Cette contrainte a orienté la conception du CNN « from scratch » vers un modèle volontairement frugal : 3 blocs de convolution (16→32→64 canaux), *global average pooling* plutôt qu'une couche dense de grande taille en sortie, résolution d'entrée réduite (128×128) — les images étant déjà bien cadrées (cf. §4.1, crop tangent au cercle), une haute résolution n'est pas jugée nécessaire.

### 1.5 Approches comparées

Trois approches sont comparées pour chaque caméra :

1. **CNN « from scratch » léger**, sur le dataset déséquilibré tel quel (baseline, imposée par la consigne).
2. **Le même CNN**, avec les classes minoritaires (NON Conforme, PIETRA) doublées par flip horizontal — un rééquilibrage à faible coût, les châtaignes n'ayant pas d'asymétrie gauche/droite sémantique.
3. **Un classifieur ML classique** (Random Forest) sur des caractéristiques géométriques et colorimétriques déjà calculées lors des étapes précédentes (fraction de brun, aire, largeur/hauteur du contour détecté, position du centre de masse), avec pondération de classe (`class_weight='balanced'`) comme équivalent du rééquilibrage par flip — le flip horizontal n'ayant pas de sens sur des caractéristiques déjà quasi invariantes par symétrie.

### 1.6 Suivi des expériences (MLflow)

L'ensemble des runs est journalisé dans MLflow (backend SQLite local) : un *experiment* par approche (`cnn_baseline`, `cnn_flip_augmente`, `ml_classique`), un *run* par combinaison approche/caméra. Sont loggés : hyperparamètres, courbes de perte/exactitude par époque, exactitude finale sur le jeu de test, précision/rappel/F1 par classe, matrice de confusion (image), importance des caractéristiques (modèle ML), et le modèle entraîné lui-même.

*Point technique résolu en cours de campagne* : la sauvegarde du modèle PyTorch échouait sur l'accélération MPS (Apple Silicon) du fait d'un conflit d'appareil lors du traçage au format `pt2` (nouveau format par défaut de MLflow). Corrigé en repassant explicitement le modèle sur CPU et en forçant le format de sérialisation `pickle` avant sauvegarde — sans incidence sur les métriques déjà loggées avant ce point, ni sur le rechargement ultérieur du modèle.

---

## 2. Résultats de la campagne d'entraînement

Six runs ont été exécutés (3 approches × 2 caméras), avec arrêt anticipé (patience de 6 époques sans amélioration de l'exactitude de validation) :

| Run | Exactitude test |
|---|---|
| cnn_baseline_camB | 74,7 % |
| cnn_flip_camB | 75,3 % |
| ml_rf_camB | 49,6 % |
| cnn_baseline_camT | 72,3 % |
| cnn_flip_camT | 78,2 % |
| ml_rf_camT | 47,2 % |

### 2.1 Le CNN domine largement le ML classique

Le classifieur Random Forest obtient une exactitude proche ou inférieure à la simple prédiction de la classe majoritaire (51,3 % pour B, 48,2 % pour T) — un résultat nettement moins bon que celui obtenu sur la tâche vide/non-vide (§4.1, 90,3 % d'exactitude) avec des caractéristiques similaires. Ceci confirme que les caractéristiques géométriques/colorimétriques simples, pertinentes pour détecter la présence d'une châtaigne, ne suffisent pas à distinguer Conforme / NON Conforme / PIETRA — une tâche qui repose sur des indices visuels plus fins (texture, coloration localisée) qu'un CNN apprend directement, contrairement à un jeu de caractéristiques fixées à la main.

### 2.2 Effet du rééquilibrage par flip

Le flip horizontal des classes minoritaires améliore l'exactitude dans les deux cas, modestement pour la caméra B (+0,6 point) et plus nettement pour la caméra T (+5,9 points). Un seul run ayant été exécuté par configuration, une partie de cet écart — en particulier pour T — peut relever du bruit d'entraînement plutôt que d'un effet systématique ; l'absence de répétitions avec graines aléatoires différentes ne permet pas d'estimer un intervalle de confiance sur ce gain.

### 2.3 Point d'observation : instabilité des courbes de validation

L'exactitude de validation oscille fortement d'une époque à l'autre pour l'ensemble des runs CNN (écarts de plusieurs dizaines de points parfois observés entre deux époques consécutives). L'early stopping, basé sur le meilleur point observé, absorbe cette instabilité sans faire échouer l'entraînement, mais suggère qu'un taux d'apprentissage plus faible ou un ordonnancement (*scheduler*) pourrait stabiliser et potentiellement améliorer les résultats — piste non explorée dans le temps imparti à cette campagne.

---

## 3. Le critère qui compte : précision et rappel sur Conforme

L'exactitude globale n'est pas la métrique du cahier des charges GRPTMC, qui impose des seuils spécifiques sur la seule classe Conforme : **précision ≥ 95 %** et **rappel ≥ 85 %**. Ces métriques ont été extraites pour chacun des six runs :

| Run | Exactitude | Précision Conforme | Rappel Conforme | Conforme au cahier des charges ? |
|---|---|---|---|---|
| cnn_flip_camT | 78,2 % | 83,3 % | 90,3 % | Non (précision insuffisante) |
| cnn_flip_camB | 75,3 % | 81,9 % | 87,5 % | Non (précision insuffisante) |
| cnn_baseline_camB | 74,7 % | 77,3 % | 93,6 % | Non (précision insuffisante) |
| cnn_baseline_camT | 72,3 % | 85,6 % | 80,1 % | Non (précision et rappel insuffisants) |
| ml_rf_camB | 49,6 % | 60,5 % | 65,4 % | Non (très insuffisant) |
| ml_rf_camT | 47,2 % | 58,2 % | 59,7 % | Non (très insuffisant) |

**Le rappel est globalement proche de la cible** (trois des quatre runs CNN dépassent déjà 85 %), mais **la précision est systématiquement le facteur limitant** (77 % à 86 % pour les CNN, contre 95 % exigés). Concrètement, trop de fruits NON Conforme ou PIETRA sont encore classés à tort « Conforme » — l'erreur que le cahier des charges juge la plus grave. Ce résultat est attendu à ce stade : l'entraînement utilise une fonction de perte standard, sans pondération reflétant l'asymétrie de coût du cahier des charges.

---

## 4. Diagnostic de l'insuffisance de précision

### 4.1 Décomposition des faux positifs Conforme

Sur le modèle `cnn_baseline_camB`, les 249 faux positifs Conforme du jeu de test (prédiction Conforme, vraie classe différente) se décomposent ainsi :

| Vraie classe | Nombre | Part des faux positifs |
|---|---|---|
| PIETRA | 195 | 78,3 % |
| NON Conforme | 54 | 21,7 % |

Les erreurs sont massivement concentrées sur PIETRA plutôt que sur NON Conforme.

### 4.2 Explication physique

Le critère PIETRA repose sur la présence de tan (fine peau interne, rousse et amère, devant être retirée) non correctement ôté. Il s'agit bien d'un défaut visuel réel — non d'un critère purement commercial comme initialement supposé avant cette analyse (le nom PIETRA fait référence à la bière corse du même nom, à base de farine de châtaigne, vers laquelle ces fruits sont réorientés ; il ne s'agit cependant pas d'un critère de destination déconnecté de l'apparence, mais bien la conséquence commerciale d'un défaut visuel identifié).

Point déterminant pour l'interprétation des résultats : **le tan peut n'être visible que sur une seule face du fruit**. Un modèle n'observant qu'une seule caméra peut donc voir une face parfaitement saine d'un fruit par ailleurs classé PIETRA — non par erreur d'apprentissage, mais parce que l'information disqualifiante est physiquement absente de son champ de vision. Ceci constitue une **limite structurelle du dispositif mono-caméra**, cohérente avec la concentration des erreurs observée sur PIETRA plutôt que sur NON Conforme (un fruit pourri présentant en général des signes visibles sous tous les angles).

### 4.3 Ajustement du seuil de décision : insuffisant seul

Avant de conclure à une limite structurelle, l'hypothèse d'un simple mauvais réglage du seuil de décision (argmax standard, sans marge) a été testée sur `cnn_baseline_camB` : balayage du seuil de probabilité requis pour retenir Conforme, de 0,30 à 0,98.

Résultat : les courbes de précision et de rappel se croisent vers un seuil de 0,62–0,64 (précision et rappel tous deux proches de 84–85 %), mais **aucun seuil ne satisfait simultanément les deux critères**. Lorsque la précision atteint la cible de 95 % (seuil ≈ 0,92), le rappel s'est déjà effondré à 30 %. Ce résultat confirme que le problème n'est pas un mauvais réglage du seuil sur ce modèle, mais un manque de pouvoir de discrimination intrinsèque depuis une seule vue — cohérent avec l'explication physique de la section 4.2.

---

## 5. Stratégie de fusion T/B

### 5.1 Principe

Puisque le tan peut être invisible d'un seul côté, une fusion des prédictions T et B pour une même châtaigne (assumée être physiquement le même fruit vu sous deux angles, hypothèse valide en production où les deux flux caméra d'un poste sont synchronisés en temps réel) est le levier le plus directement justifié pour améliorer la précision sur Conforme — une caméra peut rattraper ce que l'autre ne voit pas.

### 5.2 Règle de fusion (vote strict)

Règle retenue et implémentée :

- les deux modèles prédisent Conforme → **Conforme** ;
- au moins un modèle prédit NON Conforme → **NON Conforme** (priorité absolue : aucun défaut grave ne doit passer, même détecté par une seule caméra) ;
- tout le reste (désaccord sans NON Conforme) → **PIETRA** (filet de sécurité, capture notamment le cas où une seule caméra voit le tan).

Cette règle a été vérifiée unitairement sur les neuf combinaisons possibles de prédictions.

### 5.3 Estimation théorique du gain (sous hypothèse d'indépendance)

Aucune paire T/B réelle n'étant disponible sur le dataset labellisé (cf. rapport §4.1), le gain de la fusion a été **simulé** à partir des matrices de confusion réelles des deux modèles `cnn_baseline` (T et B), sous l'hypothèse que les erreurs de classification des deux caméras sont indépendantes conditionnellement à la vraie classe du fruit — hypothèse plausible (le côté où le tan est visible n'a pas de raison de dépendre de la caméra) mais non vérifiée empiriquement.

Résultat de la simulation, pondérée par les proportions réelles des classes dans le dataset :

- **Précision Conforme estimée : 94,4 %** (à 0,6 point du seuil de 95 %)
- **Rappel Conforme estimé : 74,9 %** (10,1 points sous le plancher de 85 %)

La précision se rapproche fortement de la cible, mais le rappel chute significativement sous l'effet de la règle d'unanimité : mathématiquement, le rappel fusionné correspond au **produit des rappels individuels** des deux modèles sur Conforme (0,801 × 0,936 = 0,750, vérifié analytiquement). Le modèle T, dont le rappel Conforme individuel (80,1 %) est plus faible que celui de B (93,6 %) — 18,6 % des vrais Conforme y sont classés à tort PIETRA —, est le facteur limitant de cette multiplication.

*Cette estimation reste théorique et n'a pas été optimisée davantage (balayage de seuils par caméra avant fusion) : le temps disponible a été jugé mieux employé sur une refonte qualitative de la règle de fusion plutôt que sur l'optimisation d'un résultat simulé, non mesuré sur données réelles.*

### 5.4 Raffinement proposé : fusion pondérée par la confiance

Le vote strict traite de façon identique un cas où les deux modèles sont très confiants (98 %/98 %) et un cas où un modèle est très confiant en Conforme (98 %) tandis que l'autre hésite légèrement en faveur de PIETRA (51 %) — les deux étant classés Conforme dans le premier cas, PIETRA dans le second, alors que le second cas mérite sans doute d'être reconsidéré.

Une variante a été formulée : le veto NON Conforme reste un vote dur et prioritaire (inchangé, l'erreur grave ne doit pas être assouplie), mais la décision Conforme/PIETRA repose sur la **moyenne géométrique des probabilités Conforme des deux modèles**, comparée à un seuil (0,5 par défaut — valeur arbitraire, à documenter comme telle). Cette formulation a été vérifiée sur des cas illustratifs construits à la main : le cas motivant (98 % / 51 % PIETRA) bascule bien en Conforme, tandis que le veto NON Conforme reste respecté même face à un Conforme très confiant de l'autre côté.

Cette variante n'a pas été intégrée à la simulation théorique de la section 5.3 (arbitrage de temps, cf. remarque ci-dessus) ; elle est documentée comme piste de raffinement pour la suite.

---

## 6. Comparaison avec des architectures pré-entraînées

### 6.1 Justification et protocole

Le dossier `training/` mentionné dans l'énoncé pour cette comparaison s'est révélé inaccessible (erreur du sujet, confirmée en cours de session). Trois architectures pré-entraînées publiques (poids ImageNet, `torchvision`) ont été retenues à la place, choisies pour couvrir un éventail de tailles pertinent au regard de la contrainte de déploiement : **MobileNetV3-Small**, **MobileNetV2** et **EfficientNet-B0**.

Stratégie de finetuning : le backbone (extracteur de caractéristiques) est **gelé** (poids ImageNet conservés tels quels), seule la tête de classification est réentraînée. Choix justifié par le volume de données modeste par caméra (~10 000 images) au regard de réseaux à plusieurs millions de paramètres, et par la contrainte de temps. Chaque architecture utilise son prétraitement recommandé (`weights.transforms()` de torchvision), avec une résolution d'entrée de 224×224 imposée par les poids préentraînés — contre 128×128 pour le CNN from scratch.

**⚠️ Erreur de protocole identifiée a posteriori** : les résultats de la section 6.2 ont été obtenus sur le dossier d'**images brutes** (`images/`, non recadrées), et non sur `img_dataset_exam` (images traitées, crop tangent au cercle du §4.1), par confusion de chemin lors du lancement. Contrairement au CNN from scratch (section 2, entraîné sur images traitées), **la comparaison de la section 6.2/6.3 mélange donc deux prétraitements différents** — elle reste informative sur l'ordre de grandeur, mais sa conclusion (6.3) doit être lue avec cette réserve : une partie de l'écart observé pourrait provenir du prétraitement plutôt que de la seule architecture. Cette confusion a été détectée en préparant l'expérience de la section 6.4, et a motivé la construction d'une comparaison à isoprétraitement (backbone gelé sur images traitées, en cours au moment de la rédaction).

### 6.2 Résultats (backbone gelé, images BRUTES)

| Modèle | Caméra | Exactitude test | Taille (Mo) | Latence (ms/image) |
|---|---|---|---|---|
| **CNN from scratch (flip)** | B | **75,3 %** | **0,10** | **0,52** |
| **CNN from scratch (flip)** | T | **78,2 %** | **0,10** | **0,53** |
| CNN from scratch (baseline) | B | 74,7 % | 0,10 | 0,52 |
| CNN from scratch (baseline) | T | 72,3 % | 0,10 | 0,52 |
| EfficientNet-B0 (finetuné) | T | 71,2 % | 15,58 | 9,72 |
| EfficientNet-B0 (finetuné) | B | 69,3 % | 15,58 | 9,39 |
| MobileNetV3-Small (finetuné) | T | 69,8 % | 5,93 | 6,08 |
| MobileNetV2 (finetuné) | B | 68,6 % | 8,73 | 8,40 |
| MobileNetV3-Small (finetuné) | B | 68,2 % | 5,93 | 6,52 |
| MobileNetV2 (finetuné) | T | 67,5 % | 8,73 | 7,94 |
| Random Forest (ML classique) | B | 49,6 % | ~93 | — |
| Random Forest (ML classique) | T | 47,2 % | ~119 | — |

*(latences mesurées sur le device d'entraînement local, Apple M1 Pro — indicatives, pas une mesure sur le matériel de production cible, à faire au §4.3)*

![Compromis précision / taille du modèle](comparaison_taille_precision.png)

### 6.3 Conclusion partielle (à isoprétraitement près) : le CNN from scratch l'emporte sur les trois axes

**Sous réserve de la confusion de prétraitement notée en 6.1** — la conclusion de cette section a été réexaminée à la section 6.4 —, contrairement à l'intuition qu'un transfert de connaissances depuis ImageNet apporterait un avantage, **le CNN from scratch, pourtant volontairement minimaliste, dépasse les trois architectures pré-entraînées en exactitude** (75,3–78,2 % contre 67,5–71,2 %, soit **6 à 7 points d'écart** avec la meilleure d'entre elles, EfficientNet-B0), tout en étant :

- **~156 fois plus léger** (0,10 Mo contre 15,58 Mo pour EfficientNet-B0) ;
- **~18 fois plus rapide** à l'inférence (0,52 ms contre 9,5 ms par image).

Le classifieur ML classique (Random Forest) cumule à l'inverse le pire des deux mondes : la précision la plus faible de toute la campagne (47–50 %) et, de façon inattendue, les modèles les plus volumineux (93–119 Mo, sans limite de profondeur d'arbre) — soit environ **1000 fois plus lourd** que le CNN from scratch pour un résultat très inférieur. Ce constat sur le ML classique n'est pas affecté par la confusion de prétraitement (le Random Forest utilise ses propres caractéristiques géométriques, recalculées séparément sur `img_dataset_exam`) et reste donc pleinement valide.

### 6.4 Hypothèse de décorrélation de domaine et test par dégel partiel

**Hypothèse formulée** : la sous-performance des architectures pré-entraînées pourrait s'expliquer par une décorrélation entre les caractéristiques apprises sur ImageNet (images naturelles : objets, scènes, animaux) et la texture spécifique de cette tâche (gros plan macro sur une seule surface, cadrage circulaire artificiel introduit par le crop du §4.1). Le simple entraînement d'une tête linéaire sur des caractéristiques gelées reviendrait alors à utiliser ces caractéristiques comme une « sonde linéaire » : les résultats de la section 6.2 mesureraient directement à quel point elles sont corrélées aux classes visées, indépendamment de toute confusion de prétraitement.

**Protocole de test** : dégel partiel d'EfficientNet-B0 (meilleure architecture pré-entraînée de la section 6.2) — les 3 derniers blocs (indices 6, 7, 8 sur 9, portant les caractéristiques de plus haut niveau, ~3,15 millions de paramètres sur 4,0 millions au total) sont dégelés et réentraînés, les blocs 0 à 5 (bas niveau : contours, dégradés) restent gelés. Taux d'apprentissage différenciés : 1e-4 pour le backbone dégelé (déjà pré-entraîné, à ne pas perturber brutalement), 1e-3 pour la nouvelle tête (entraînée de zéro) — pratique standard de finetuning. Cette fois exécuté correctement sur les images **traitées** (`img_dataset_exam`).

**Résultat** :

| Modèle | Caméra | Prétraitement | Exactitude test | Taille (Mo) | Latence (ms/image) |
|---|---|---|---|---|---|
| EfficientNet-B0 (gelé) | B | Brut | 69,3 % | 15,58 | 9,39 |
| EfficientNet-B0 (gelé) | T | Brut | 71,2 % | 15,58 | 9,72 |
| EfficientNet-B0 (gelé) | B | Traité | 71,4 % | 15,58 | 10,65 |
| EfficientNet-B0 (gelé) | T | Traité | 70,5 % | 15,58 | 10,43 |
| **EfficientNet-B0 (dégel partiel)** | B | Traité | **78,8 %** | 15,58 | 10,11 |
| **EfficientNet-B0 (dégel partiel)** | T | Traité | **78,9 %** | 15,58 | 10,30 |
| CNN from scratch (flip) | B | Traité | 75,3 % | 0,10 | 0,52 |
| CNN from scratch (flip) | T | Traité | 78,2 % | 0,10 | 0,53 |

Le dégel partiel améliore l'exactitude de 9 à 18 points par rapport au gel complet, et **dépasse même le CNN from scratch** sur les deux caméras (bien que de peu côté T). Ceci confirme que le gel complet du backbone était bien un facteur limitant majeur — cohérent avec l'hypothèse de décorrélation — mais au prix d'un coût de ressources bien supérieur (~156 fois plus lourd, ~19 fois plus lent que le CNN from scratch).

**Isolation de l'effet du prétraitement (matrice 2×2 complétée)** : la comparaison à isostratégie de dégel (gelé sur brut vs gelé sur traité) montre un écart négligeable — +2,1 points pour la caméra B (69,3 % → 71,4 %), -0,7 point pour la caméra T (71,2 % → 70,5 %) —, compatible avec le bruit d'un run unique plutôt qu'un effet réel du prétraitement. **Le gain de 7 à 8 points observé entre gelé et dégel partiel (tous deux sur images traitées) est donc attribuable presque intégralement au dégel du backbone, pas au crop.** Ce constat est spécifique à ce backbone pré-entraîné gelé/dégelé ; il ne remet pas en cause l'utilité du prétraitement pour les autres usages du pipeline (détection vide/non-vide du §4.1, caractéristiques géométriques du classifieur ML classique, qui dépendent toutes deux directement du cercle détecté par le crop).

*Limite restante : le dégel partiel n'a été testé que sur images traitées, pas sur images brutes — une interaction entre dégel et prétraitement ne peut donc pas être totalement exclue, bien que peu probable au vu du résultat ci-dessus sur le backbone gelé.*

**Second biais identifié en cours de comparaison, distinct du simple cadrage** : les dossiers `images` (brut) et `img_dataset_exam` (traité) ne contiennent pas rigoureusement la même population de fruits.

- `images` (brut) : 22 531 images valides sur 35 254, soit exactement les 12 723 vraies « Vide » de `labels_principal.csv` écartées (36,1 % du total) — le filtrage s'appuie ici directement sur la vérité terrain humaine.
- `img_dataset_exam` (traité) : 20 931 images valides, seulement 93 écartées (correspondant précisément aux 44 faux négatifs + 49 AMBIGU-en-réalité-vides identifiés au §4.1) — cohérent avec un filtrage fondé sur la détection automatique, pas sur la vérité terrain.

L'écart entre les deux (22 531 − 20 931 = 1 600) correspond exactement aux faux positifs « vide » mesurés au §4.1 : de vraies châtaignes que le détecteur colorimétrique automatique a classées à tort en vide franc, et qui n'ont donc jamais atteint l'étape de crop. Ces 1 600 fruits sont majoritairement des fragments (`chunk=True` à 64 %, cf. §4.1).

**Conséquence** : la comparaison brut/traité en cours ne teste pas uniquement l'effet du cadrage — elle teste aussi, de façon confondue, l'effet d'une population différente (le traité exclut systématiquement une partie des fragments à faible surface visible, non pas à cause du crop lui-même mais d'un effet de sélection en amont, dans la chaîne de détection vide/non-vide). Un écart de performance observé entre brut et traité pourrait donc refléter en partie une différence de difficulté intrinsèque des deux populations, pas seulement l'effet du prétraitement. Cette limite est documentée ici faute de temps pour la corriger (par exemple en reconstituant une version « brute filtrée sur vérité terrain » strictement comparable à `img_dataset_exam`).

### 6.5 Synthèse et conclusion révisée

En tenant compte des résultats de la section 6.4 — le dégel partiel apporte un gain réel et propre à l'adaptation du backbone, indépendant du prétraitement —, deux options apparaissent désormais compétitives sur la précision, avec un compromis clair côté ressources :

- Le **CNN from scratch** reste imbattable en frugalité (156 fois plus léger, 19 fois plus rapide que toute alternative pré-entraînée testée), pour une précision très proche du meilleur résultat obtenu par ailleurs.
- Le **dégel partiel d'EfficientNet-B0** obtient la meilleure précision brute de toute la campagne, mais à un coût de ressources qui reste probablement excessif au regard du budget mémoire de déploiement retenu (~1,5 Go, à partager entre deux modèles T et B simultanés) et de la contrainte de latence de production (12 flux caméra à traiter sans accumulation de retard).

**Le choix final entre les deux dépendra de la marge réelle de ressources disponible sur le matériel de production**, à établir au §4.3. À ressources très contraintes, le CNN from scratch reste le choix par défaut le plus sûr ; si la latence/mémoire mesurée sur le matériel cible laisse une marge suffisante, le dégel partiel d'EfficientNet-B0 mérite d'être reconsidéré pour son gain de précision.

---

## 7. Limites et points ouverts

- **Aucune mesure réelle de la fusion T/B** : toute la section 5 repose sur une simulation sous hypothèse d'indépendance, faute de paires T/B disponibles sur le dataset labellisé. Une vraie mesure nécessitera soit une correspondance T/B (non obtenue sur ce dataset, cf. §4.1), soit des données de production synchronisées.
- **Un seul run par configuration** : les écarts observés (notamment le gain du flip sur la caméra T) ne sont pas accompagnés d'une estimation de variance ; plusieurs graines aléatoires seraient nécessaires pour trancher entre effet réel et bruit d'entraînement.
- **Courbes de validation instables** : piste d'amélioration non explorée (taux d'apprentissage, ordonnancement) qui pourrait à la fois stabiliser et améliorer les résultats de tous les runs CNN.
- **Seuils arbitraires non optimisés** : le seuil de la fusion pondérée (0,5) et les seuils de décision individuels n'ont pas fait l'objet d'une recherche systématique, par choix (cf. 5.3) plutôt que par contrainte technique.
- **Budget mémoire de déploiement (1,5 Go)** : hypothèse de travail prudente, non vérifiée sur le matériel réel ni sur un équivalent — à mesurer au §4.3.
- **Finetuning complet non testé** : seul un dégel partiel (3 derniers blocs) a été testé sur EfficientNet-B0 (section 6.4) ; un dégel plus profond, ou étendu à MobileNetV2/V3, pourrait encore changer la conclusion, mais n'a pas été exploré par manque de temps.
- **Dégel partiel non testé sur images brutes** : seule la combinaison dégel partiel × traité a été mesurée (78,8/78,9 %) ; une interaction entre dégel et prétraitement reste théoriquement possible, bien que peu probable au vu de l'absence d'effet du prétraitement sur le backbone gelé (cf. section 6.4).
- **Comparaison brut/traité confondue avec une différence de population** : `img_dataset_exam` exclut systématiquement ~1 600 fruits (majoritairement des fragments) que le détecteur vide/non-vide automatique a mal classés en amont — ces fruits sont bien présents dans le dossier brut. Un écart observé entre brut et traité peut donc refléter une différence de difficulté de la population plutôt que l'effet du seul cadrage (cf. section 6.4).
- **Latences mesurées sur M1 Pro, pas sur le matériel cible** : les latences comparées en section 6 sont indicatives (ordres de grandeur relatifs entre modèles) ; une mesure sur la GTX 1060 ou un équivalent reste à faire au §4.3.
- **Stratégie du lot « repasse »** : proposée mais non spécifiée (traitement matériel, calendrier de ré-analyse) ; relève d'une décision opérationnelle hors du périmètre de ce rapport.
