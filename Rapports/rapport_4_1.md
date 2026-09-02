# Rapport §4.1 — Qualité et complétude de la donnée
## Projet CastagNet (MSc BIHAR, ESTIA — MESPR)

---

## 1. Introduction

### 1.1 Contexte

Le projet **CastagnIA** vise le développement d'un système de tri automatique de châtaignes sèches pour le compte du GRPTMC (Groupement Régional des Producteurs et Transformateurs de Châtaignes et Marrons Corses). La ligne de production repose sur 12 caméras (6 emplacements, chacun équipé d'une vue du dessus — caméra T — et d'une vue du dessous — caméra B), et classe chaque fruit en quatre catégories : Conforme, NON Conforme, PIETRA et Vide.

Le cahier des charges impose des exigences de qualité asymétriques : un rappel minimal de 85 % et une précision minimale de 95 % sur la catégorie Conforme, reflétant la priorité donnée par le groupement à la qualité du produit fini (farine de châtaigne) sur le rendement matière. Le matériel de production cible (GPU GTX 1060 3 Go de VRAM) impose par ailleurs des contraintes fortes sur le choix et le déploiement des modèles.

### 1.2 Travail demandé (§4.1)

Le volet 4.1 de l'épreuve porte sur la qualité et la complétude de la donnée labellisée, et comprend notamment :

- le découpage d'un extrait vidéo de la machine de tri en images, puis le recadrage autour de chaque châtaigne détectée ;
- la recherche et la justification d'une méthode reliant les deux vues (caméra T et caméra B) d'une même châtaigne, aussi bien pour cet extrait vidéo que pour les images déjà labellisées du dataset (35 254 images fournies) ;
- la constitution d'un dépôt Git organisant ces éléments ;
- la production d'un rapport de qualité du dataset actualisé.

Ce document rend compte du travail réalisé sur ces points, jusqu'à l'obtention d'un jeu d'images prêt pour l'entraînement des modèles (volet §4.2, traité séparément).

---

## 2. Traitement de l'extrait vidéo de démonstration

### 2.1 Données de départ

Deux fichiers vidéo (DV, 720×576, 25 fps, 30 secondes) ont été fournis, correspondant chacun à une caméra d'un même poste de la machine : `attachment.avi` (caméra B) et `attachment-2.avi` (caméra T). Le dispositif filmé présente les échantillons un par un à travers un hublot circulaire, de façon cyclique (28 passages sur les 30 secondes, période moyenne d'environ 26,7 frames).

### 2.2 Détection des cycles par la pastille de référence

Une pastille verte, fixée à côté du hublot et solidaire du mécanisme, balaie verticalement l'image à chaque cycle indépendamment de la présence ou non d'une châtaigne. Ce repère a été retenu comme base de la segmentation en cycles plutôt que la détection directe de la châtaigne, car il reste détectable même sur les passages à vide :

- détection par seuillage de couleur (HSV) avec filtrage sur l'aire minimale du contour ;
- segmentation en cycles par recherche des minima locaux de la distance de la pastille à la ligne médiane horizontale de l'image, avec fusion des minima trop proches.

Cette méthode a permis d'isoler de façon fiable les 28 cycles sur chacune des deux caméras.

### 2.3 Classification vide / non-vide / AMBIGU

Pour chaque cycle, la présence d'une châtaigne est déterminée par la fraction de pixels de couleur "brun châtaigne" (seuillage HSV) dans la zone du hublot. Un seuil unique s'étant révélé instable pour les cas de fragments partiellement hors cadre (un même cycle physique basculant de "présent" à "vide" selon de légères variations de méthode), deux seuils ont été retenus : en deçà du seuil bas, le cycle est classé vide ; au-delà du seuil haut, non-vide ; entre les deux, le statut est marqué **AMBIGU** et exclu des comparaisons automatiques.

### 2.4 Méthode de correspondance T/B

Deux pistes ont été écartées avant d'aboutir à la méthode retenue :

- le **numéro de frame absolu** n'est pas fiable, les deux vidéos pouvant démarrer et se terminer à des instants différents (latence de déclenchement, désynchronisation matérielle) ;
- l'**apparence de la châtaigne** (couleur, forme) n'est pas non plus un critère stable, une même châtaigne pouvant se présenter différemment selon l'angle de vue ou bouger entre les deux prises.

La méthode retenue s'appuie sur l'absence de châtaigne : un cycle vide l'est nécessairement dans les deux vues simultanément, puisqu'il s'agit du même passage physique. Une séquence binaire vide/non-vide est donc construite pour chaque caméra, et le décalage entre les deux séquences est recherché automatiquement en maximisant le nombre de correspondances vide-vide confirmées, sous la contrainte stricte qu'aucun conflit (un vide face à un non-vide) ne soit toléré — un tel conflit indiquant un décalage incorrect plutôt qu'un cas limite à accepter.

Cette méthode a été appliquée avec succès sur les deux vidéos de démonstration : le décalage retenu est corroboré par deux correspondances vide-vide indépendantes, avec un espacement identique entre elles (signature peu compatible avec une coïncidence), et confirmé a posteriori par une correspondance de châtaigne repérée visuellement en dehors du calcul.

### 2.5 Recadrage des images extraites

Plusieurs approches de recadrage ont été testées avant de retenir une solution :

| Approche | Résultat |
|---|---|
| Bounding box variable autour du contour brun détecté | Fonctionnelle mais taille de sortie non uniforme, sensible au bruit sur les cycles vides |
| Taille fixe centrée sur un cercle détecté par transformée de Hough | Détection du cercle insuffisamment précise |
| Cercle centré sur le milieu de l'image, rayon égal à la distance pastille–centre | Fonctionnelle, mais taille de sortie légèrement variable |
| **Carré fixe 350×350, centré horizontalement, marge verticale asymétrique selon la caméra** | **Retenue** : aucun échec ni remplissage nécessaire sur l'ensemble testé |

La règle finale associe une marge plus généreuse du côté où la châtaigne est susceptible de déborder du hublot (bas de l'image pour la caméra T, haut pour la caméra B), avec un filtre optionnel de suppression des pixels achromatiques (blanc/gris/noir) pour isoler la châtaigne du métal environnant.

---

## 3. Application au dataset labellisé (35 254 images)

### 3.1 Constat initial

Le dataset fourni est composé d'images déjà recadrées individuellement, nommées selon la convention `annee_label_Cam_{T|B}_numCamera_numEchantillon.jpg`. Contrairement à l'extrait vidéo, ces images ne constituent pas un flux continu et synchronisé : elles correspondent à des échantillons sélectionnés et exportés indépendamment pour chaque caméra en vue de la constitution du jeu de labellisation.

### 3.2 Tentatives de correspondance T/B

Plusieurs approches ont été explorées, par analogie avec la méthode validée sur les vidéos :

1. **Décalage constant sur le numéro d'échantillon.** Le taux de conflit vide/non-vide observé reste stable, entre 27 % et 38 %, sur l'ensemble des décalages testés. Aucun décalage ne s'approche d'un taux de conflit nul, ce qui indique que le numéro d'échantillon n'est vraisemblablement pas un compteur synchronisé entre les deux caméras pour ce sous-ensemble de données.

2. **Ancrage sur la première image non-vide de chaque liste, avec fenêtre de recherche restreinte.** Cette approche améliore la robustesse de la recherche (élimine les décalages aberrants portés par une unique coïncidence) mais ne résout pas le problème de fond : le taux de conflit résiduel reste trop élevé pour conclure à une correspondance fiable à l'échelle du dataset.

3. **Appariement par similarité de caractéristiques visuelles** (fraction de brun, taille et position du contour détecté), avec correction d'un biais systématique de caméra identifié en cours d'analyse (les valeurs mesurées diffèrent significativement entre T et B pour des raisons d'angle de vue et d'éclairage, indépendamment de la châtaigne réelle) et appariement bijectif par algorithme d'affectation optimal. Cette approche a été validée avec succès sur un échantillon contrôlé de quatre images, retrouvant exactement la correspondance identifiée par inspection visuelle. Sa généralisation fiable à l'ensemble du dataset (calibration du biais sur un échantillon représentatif, validation manuelle d'un sous-ensemble) représente cependant un volume de travail disproportionné au regard du poids de ce point au barème.

### 3.3 Décision retenue

Compte tenu de la contrainte de temps impartie à l'épreuve et du poids relatif de ce point de méthode dans l'évaluation, il a été décidé de **ne pas chercher à établir de correspondance T/B sur le dataset déjà labellisé**. Cette décision a une conséquence directe sur l'architecture de modélisation prévue au §4.2 :

- deux modèles seront entraînés séparément, chacun sur les images d'une seule position de caméra (T ou B), chacun capable de classifier indépendamment une image en Conforme / NON Conforme / PIETRA / Vide ;
- un post-traitement combinera les deux sorties pour produire la décision finale par châtaigne, selon une stratégie de fusion à définir en tenant compte de l'asymétrie de coût du cahier des charges (une piste envisagée consiste à ne retenir la classe Conforme que si les deux modèles convergent).

Cette architecture reste par ailleurs cohérente avec le contexte de production réel, où les deux flux caméra d'un même poste sont synchronisés en temps réel au moment de l'inférence — la difficulté rencontrée ici est propre à l'export du jeu de données d'entraînement, non au fonctionnement de la ligne en production.

---

## 4. Classification et préparation finale des images d'entraînement

### 4.1 Classification vide / non-vide / AMBIGU sur l'ensemble du dataset

La méthode de classification établie sur les vidéos (fraction de pixels bruns dans la zone du hublot, matérialisée ici par un cercle vert complet plutôt qu'une pastille) a été appliquée à l'ensemble des 35 254 images. Les seuils, initialement calibrés sur l'échantillon de démonstration, se sont révélés trop permissifs à l'échelle du dataset réel et ont été resserrés, ramenant la proportion de « Vide » détectée à 40,4 %.

Répartition obtenue sur les 35 254 images, avec les seuils ajustés :

| Statut | Nombre | Proportion |
|---|---|---|
| Vide | 14 230 | 40,4 % |
| Non-vide | 19 249 | 54,6 % |
| AMBIGU | 1 775 | 5,0 % |

### 4.2 Validation contre la vérité terrain relue par un humain

Le fichier `labels_principal.csv`, comportant une colonne `label_principal` établie par relecture humaine, a permis de valider ce résultat a posteriori. Il en ressort que la proportion réelle d'images « Vide » dans le dataset est de **36,1 %** (12 723 images sur 35 254) — une valeur proche des 40,4 % obtenus automatiquement, bien plus cohérente que ne le laissait supposer l'hypothèse initiale d'un taux de « Vide » nul déduite du seul rapport d'inventaire (qui ne comptabilisait que le label porté par le nom de fichier, non la relecture humaine).

La comparaison point par point donne les résultats suivants (hors cas AMBIGU) :

| Métrique | Valeur |
|---|---|
| Rappel sur la classe Vide | 99,7 % |
| Précision sur la classe Vide | 88,8 % |
| Exactitude globale (True/False uniquement) | 90,3 % |

La quasi-totalité des vraies images vides sont correctement détectées (rappel de 99,7 %). L'essentiel de l'écart de précision provient d'un phénomène identifié précisément : parmi les 1 600 images classées à tort « Vide » alors qu'elles contiennent une châtaigne, 64 % sont marquées `chunk=True` (fragment/débris) dans la relecture humaine, contre 14,3 % sur l'ensemble du dataset — soit une sur-représentation d'un facteur 4,5. Un fragment de petite taille présente en effet une surface brune visible minime, insuffisante pour dépasser le seuil de détection, bien qu'un relecteur humain l'identifie correctement comme une châtaigne réelle. Cette source d'erreur est ponctuelle et bien circonscrite plutôt que le signe d'un biais méthodologique généralisé.

Par ailleurs, sur les 1 775 images classées AMBIGU, 1 726 (97,2 %) sont en réalité non-vides d'après la relecture humaine : la zone AMBIGU capte donc majoritairement des cas limites de châtaignes réelles (probablement des fragments également), conformément à l'usage prévu pour ce statut — signaler les cas incertains pour relecture plutôt que de trancher arbitrairement.

*Note méthodologique : une hypothèse formulée avant l'accès à `labels_principal.csv` — une couleur atypique des échantillons PIETRA en raison de leur nature « minérale », qui aurait pu expliquer un taux de « Vide » élevé — a été invalidée : le label PIETRA désigne les châtaignes destinées à la bière corse du même nom (fabriquée à base de farine de châtaigne). La véritable explication du taux de « Vide » élevé, identifiée grâce à la vérité terrain, est le phénomène des fragments (`chunk`) décrit ci-dessus.*

*Précision complémentaire apportée en cours de projet (§4.2) : le critère PIETRA n'est pas dénué de fondement visuel comme d'abord supposé — il correspond à la présence du tan (fine peau interne rousse et amère) non correctement retiré. C'est bien un défaut réel, détectable visuellement, mais qui présente une particularité importante pour la suite : le tan peut n'être visible que sur une seule face du fruit, l'autre pouvant avoir l'aspect d'une châtaigne parfaitement conforme. Cette caractéristique s'est révélée déterminante pour l'architecture de décision finale (cf. §4.2, section 10).*

### 4.3 Matrice de confusion et analyse par catégorie

![Matrice de confusion](matrice_confusion.png)

![Histogramme de comparaison](histogramme_comparaison.png)

La matrice de confusion, détaillée par catégorie réelle (Vide / Conforme / NON Conforme / PIETRA), fait apparaître une hétérogénéité notable du taux d'erreur (Vide ou AMBIGU prédit à tort) selon le label :

| Label réel | Prédit Vide à tort | Prédit AMBIGU à tort |
|---|---|---|
| Conforme | 3,7 % | 5,3 % |
| NON Conforme | 12,8 % | 10,3 % |
| PIETRA | 7,8 % | 9,5 % |

Les catégories NON Conforme et PIETRA présentent un taux d'erreur environ deux fois supérieur à Conforme. Deux hypothèses ont été envisagées pour expliquer cet écart : une confusion liée à la taille des fragments (mécanisme déjà identifié en 4.2), ou une confusion de teinte propre à ces catégories — les châtaignes NON Conforme (pourries) tirant vers le marron foncé/noir, et les PIETRA conservant potentiellement leur peau foncée, deux teintes plus proches de l'intérieur sombre du hublot vide que le brun clair d'une châtaigne Conforme.

La vérification par croisement avec la colonne `chunk` (fragment/débris) donne un résultat nuancé, différent selon la catégorie :

| Label réel | Chunk parmi « Vide » prédit à tort | Chunk parmi « AMBIGU » prédit à tort |
|---|---|---|
| Conforme | 84,3 % | 73,3 % |
| PIETRA | 85,9 % | 68,8 % |
| NON Conforme | 38,9 % | 28,8 % |

PIETRA suit en réalité le même schéma que Conforme : l'écrasante majorité de ses erreurs s'explique par la présence d'un fragment, invalidant l'hypothèse d'une confusion de teinte pour cette catégorie. En revanche, **NON Conforme se distingue nettement** : moins de 40 % de ses erreurs sont associées à un fragment, ce qui laisse la majorité des cas inexpliqués par la taille. Ce résultat est cohérent avec l'hypothèse de la teinte : une châtaigne dégradée (pourrissement, brunissement prononcé) se rapproche davantage de la couleur sombre de l'intérieur du hublot vide, réduisant la fraction de pixels détectés comme « brun châtaigne » au sens de la plage colorimétrique utilisée, indépendamment de sa taille réelle. Cette hypothèse n'a pas été vérifiée directement (par exemple en mesurant la luminosité moyenne des échantillons NON Conforme mal classés) faute de temps disponible dans cette session, mais constitue la piste la plus probable au vu des éléments recueillis.

### 4.4 Répartition des labels dans le jeu d'entraînement retenu

Le sous-ensemble d'images conservées pour l'entraînement (`img_dataset_exam`, celles classées non-vides) a été croisé avec la vérité terrain pour en analyser la composition. Sur les 19 249 images retenues, 44 (0,2 %) sont en réalité des faux négatifs (véritablement vides d'après la relecture humaine, mais non détectées comme telles) ; en excluant ces cas résiduels, la répartition des trois catégories réelles est la suivante :

![Répartition des labels dans le jeu d'entraînement](camembert_labels_entrainement.png)

| Label | Nombre | Proportion |
|---|---|---|
| Conforme | 9 885 | 51,5 % |
| PIETRA | 4 941 | 25,7 % |
| NON Conforme | 4 379 | 22,8 % |

Un déséquilibre modéré est observé (facteur d'environ 2,3 entre la classe majoritaire et les deux autres), cohérent avec les proportions déjà notées dans le rapport d'inventaire du dataset complet. Ce point est à prendre en compte au §4.2 (pondération de classes ou fonction de perte adaptée), conformément au point de vigilance méthodologique retenu en début de projet.

![Répartition des labels par caméra](histogramme_labels_par_camera.png)

La répartition est par ailleurs cohérente entre les deux caméras (écart de 1 à 2 points de pourcentage par catégorie entre T et B), ce qui est un point rassurant pour l'architecture à deux modèles indépendants retenue en section 3.3 : les deux modèles seront entraînés sur une distribution de classes très similaire, sans déséquilibre supplémentaire propre à l'une ou l'autre caméra.

### 4.5 Recadrage final des images non-vides

Les images de position T et B du dataset étant déjà recadrées par la source autour du hublot (matérialisé par un cercle vert), le contenu utile est intégralement contenu à l'intérieur de ce cercle. Le recadrage final retient donc le carré tangent au cercle sur ses quatre côtés, sans distinction T/B et sans marge superflue. La taille de sortie varie légèrement d'une image à l'autre selon le rayon détecté, sans incidence sur la suite du traitement (un redimensionnement uniforme intervient au moment de l'entraînement).

Seules les images classées non-vides sont recadrées et conservées ; le résultat est stocké dans un dossier dédié (`img_dataset_exam`), constituant la base d'images sur laquelle porteront les entraînements de modèles.

---

## 5. Rapport qualité complémentaire (vérité terrain)

Cette section exploite les colonnes de `labels_principal.csv` non encore mobilisées dans les sections précédentes (`reviewed`, `labeled_by`), pour répondre aux points du rapport qualité qui restaient ouverts : taux de relecture, répartition par année et caméra, hétérogénéité du processus de labellisation historique.

### 5.1 Taux de relecture

**100 % des 35 254 images sont marquées comme relues** (`reviewed = True`), sans exception par année (20 266/20 266 pour 2025, 14 988/14 988 pour 2026). Aucun reliquat non relu ne subsiste dans le dataset labellisé fourni — point positif, à distinguer du reliquat de ~500 images de l'année 2026 mentionné en tout début de projet et depuis traité (cf. état d'avancement initial).

### 5.2 Répartition par année et par caméra

| Caméra | 2025 (T) | 2025 (B) | 2025 Total | 2026 (T) | 2026 (B) | 2026 Total |
|---|---|---|---|---|---|---|
| 1 | 1509 | 1826 | 3335 | 1279 | 1197 | 2476 |
| 2 | 1546 | 1822 | 3368 | 1300 | 1285 | 2585 |
| 3 | 1663 | 1934 | 3597 | 1277 | 1283 | 2560 |
| 4 | 1660 | 1736 | 3396 | 1269 | 1127 | 2396 |
| 5 | 1505 | 1840 | 3345 | 1251 | 1317 | 2568 |
| 6 | 1563 | 1662 | 3225 | 1243 | 1160 | 2403 |

Le volume par caméra est homogène au sein de chaque année (écart maximal d'environ 11 % entre caméras). En revanche, la répartition des **labels** par caméra fait apparaître un écart net entre les deux années :

| Année | Conforme | NON Conforme | PIETRA | Vide |
|---|---|---|---|---|
| 2025 (plage par caméra) | 34–42 % | 12–20 % | 11–16 % | 25–42 % |
| 2026 (plage par caméra) | 19–25 % | 12–16 % | 18–26 % | 34–50 % |

La part de PIETRA double presque d'une année sur l'autre, celle de Conforme diminue d'autant, et le taux de Vide augmente également. Ce constat, qui pourrait à première vue suggérer un changement réel de qualité de récolte entre les deux campagnes, doit être relativisé au vu de l'analyse par labelleur (section 5.3) : une partie de cet écart est vraisemblablement un artefact du processus de labellisation plutôt qu'un phénomène agricole.

### 5.3 Hétérogénéité du processus de labellisation historique

Le champ `labeled_by` (texte libre, non contraint) recense neuf identifiants distincts pour 35 254 lignes :

| Labelleur | Nombre d'images | Années couvertes | Répartition des labels | Taux `chunk` |
|---|---|---|---|---|
| NicoG | 12 018 (34,1 %) | 2025 (1876) + 2026 (10 142) | Vide 35 %, Conforme 33 %, NON Conforme 17 %, PIETRA 15 % | 8,4 % |
| Tilyah | 11 080 (31,4 %) | 2025 uniquement | Conforme 45 %, Vide 31 %, PIETRA 23 %, **NON Conforme 0 %** | 27,2 % |
| *(vide, non renseigné)* | 7763 (22,0 %) | 2025 (5694) + 2026 (2069) | Vide 37 %, NON Conforme 35 %, Conforme 20 %, PIETRA 9 % | 9,2 % |
| nico | 2000 (5,7 %) | 2026 uniquement | Vide 43 %, PIETRA 43 %, Conforme 14 % | 7,3 % |
| nico h | 1000 (2,8 %) | 2025 (988) + 2026 (12) | NON Conforme 54 %, Vide 45 %, Conforme 1 % | 5,5 % |
| auto_empty_v1 | 596 (1,7 %) | 2025 (101) + 2026 (495) | Vide 100 % | 0,0 % |
| Nico | 500 (1,4 %) | 2025 uniquement | NON Conforme 69 %, Vide 31 % | 7,2 % |
| El_Ingeniero_3000 | 270 (0,8 %) | 2026 uniquement | Vide 60 %, PIETRA 19 %, NON Conforme 14 %, Conforme 7 % | 18,9 % |
| popas1 | 27 (0,1 %) | 2025 uniquement | NON Conforme 93 %, Vide 7 % | 22,2 % |

Trois constats concrets ressortent de ce tableau, chacun corrigible par un changement de processus plutôt que par un retraitement de la donnée existante :

1. **Une labelleuse n'a jamais attribué le label NON Conforme** (Tilyah, 0 % sur 11 080 images, soit 31 % du dataset total). Comparé aux 12 à 20 % observés chez les autres contributeurs sur des volumes comparables, cet écart est trop important pour être le fruit du seul hasard d'échantillonnage. Deux hypothèses restent ouvertes sans plus d'information sur le protocole de collecte : cette personne a pu travailler sur un lot pré-filtré ne contenant pas de fruits pourris, ou son interprétation du critère NON Conforme diverge de celle des autres labelleurs. Cette incertitude explique à elle seule une part significative de l'écart 2025/2026 relevé en section 5.2, puisque Tilyah est absente du labelling 2026.

2. **Le taux de fragments (`chunk`) varie d'un facteur 3 à 4 selon le labelleur** (27,2 % chez Tilyah et 22,2 % chez popas1, contre 7 à 9 % chez NicoG, Nico et nico). Rien n'indique que ces trois personnes aient reçu des lots de fruits physiquement différents ; l'écart traduit plus vraisemblablement des critères différents pour juger qu'un fragment est trop petit pour être analysé entièrement — un défaut de guide de labellisation partagé plutôt qu'une réalité terrain.

3. **L'identité des labelleurs n'est pas normalisée** : « Nico », « nico », « nico h » et « NicoG » sont vraisemblablement la même personne sous quatre graphies différentes (champ texte libre, sans liste de valeurs contrôlée), et 22 % des lignes n'ont aucun labelleur renseigné (champ vide). Un identifiant de pré-labellisation automatique (`auto_empty_v1`, 596 images, 100 % Vide) est par ailleurs mélangé au même champ que les labelleurs humains, sans distinction explicite.

### 5.4 Proposition d'évolution du schéma de labellisation collaborative

Sur la base des constats ci-dessus, et du risque de conflits Git évoqué dans l'énoncé pour un CSV partagé édité par plusieurs personnes, quatre évolutions concrètes sont proposées.

#### 5.4.1 Fusion automatisée des CSV individuels (mécanisme détaillé)

Le problème de fond avec un unique `labels_principal.csv` édité concurremment n'est pas seulement le risque de blocage Git (deux pushs successifs sur le même fichier), mais surtout un risque silencieux : Git fusionne le texte ligne à ligne sans connaître la sémantique d'un CSV. Si deux personnes ajoutent, indépendamment et sans le savoir, une ligne pour la **même image** avec des labels différents, Git ne détecte aucun conflit (ce sont deux lignes distinctes du point de vue texte) et le fichier fusionné se retrouve avec deux entrées contradictoires pour un même fruit, sans qu'aucune alerte ne soit levée.

**Solution proposée** : chaque contributeur travaille sur son propre fichier CSV (`labels_<nom>.csv`, même schéma que `labels_principal.csv`), jamais partagé avec les autres. Un script de consolidation (livré, `consolider_labels.py`) fusionne l'ensemble de ces fichiers individuels en un unique `labels_principal.csv`, **généré et jamais édité à la main**, avec une gestion explicite de trois cas :

- **doublon inoffensif** (même image, mêmes valeurs dans deux fichiers) : dédupliqué silencieusement ;
- **conflit réel** (même image, valeurs différentes selon le contributeur) : **aucune ligne n'est retenue automatiquement** — les versions en désaccord sont isolées dans un fichier `conflits_a_resoudre.csv` séparé, exclues du fichier maître tant qu'un arbitrage humain n'a pas tranché ;
- **ligne malformée** (label hors des quatre valeurs autorisées, colonne booléenne invalide, champ obligatoire vide) : isolée dans `lignes_invalides.csv`, jamais fusionnée silencieusement.

Ce mécanisme élimine structurellement le risque de conflit Git au sens strict (des fichiers distincts ne peuvent pas entrer en conflit), tout en rendant visibles et traçables les véritables désaccords de labellisation entre contributeurs — désaccords que Git, par nature, ne peut pas détecter puisqu'il ignore que la colonne `filename` identifie une image de façon unique.

**Flux opérationnel** : chaque contributeur pousse uniquement son propre fichier individuel (jamais de conflit possible à ce niveau) ; une exécution du script de consolidation (déclenchable manuellement, ou automatisée via une GitHub Action se lançant à chaque push d'un fichier `labels_*.csv`) régénère `labels_principal.csv` et le pousse à son tour, disponible immédiatement pour tous via un simple `git pull`. L'automatisation complète par GitHub Action n'a pas été mise en œuvre dans le temps imparti à cette session ; le script de consolidation, lui, est fonctionnel et livré avec ce rapport.

#### 5.4.2 Autres évolutions proposées

2. **Un identifiant de labelleur contraint** (liste déroulante ou énumération validée à la saisie dans `labeling_tool/`), pour éliminer les variantes orthographiques d'une même personne et rendre le champ `labeled_by` réellement exploitable pour un contrôle qualité inter-annotateurs.
3. **Une colonne dédiée pour distinguer labellisation humaine et pré-labellisation automatique** (booléen `auto_labeled` par exemple), plutôt que d'encoder cette information dans le même champ texte libre que l'identité humaine (cas actuel de `auto_empty_v1`).
4. Une piste plus structurante mais non chiffrée ici faute de temps : formaliser un guide de labellisation avec des exemples visuels de référence pour les critères ambigus (seuil de taille définissant un `chunk`, limite entre NON Conforme et PIETRA), pour réduire la variabilité inter-labelleur mise en évidence en section 5.3.

---

## 6. État d'avancement et suite

Le jeu d'images destiné à l'entraînement des modèles est constitué, et la fiabilité de la classification vide/non-vide a été validée contre une vérité terrain relue par un humain (90,3 % d'exactitude, source d'erreur principale identifiée et circonscrite selon la catégorie : fragments pour Conforme et PIETRA, probable confusion de teinte pour NON Conforme). Les points du rapport qualité initialement en suspens ont été traités en section 5 : taux de relecture (100 %), répartition par année/caméra, et hétérogénéité du processus de labellisation historique (variabilité inter-labelleur documentée, proposition d'évolution du schéma collaboratif). Les points suivants restent ouverts et seront traités séparément :

- l'analyse de la répartition des labels (Conforme / NON Conforme / PIETRA) au sein des images non-vides retenues pour l'entraînement (`img_dataset_exam`) ✅ **fait** (cf. section 4.4) — déséquilibre modéré (~2,3x) à anticiper au §4.2 ;
- le taux de relecture, la répartition par année/caméra et l'hétérogénéité du processus de labellisation ✅ **fait** (cf. section 5) ;
- la décision de traitement des fragments (`chunk`) mal classés « Vide » : les réintégrer au jeu d'entraînement non-vide (en s'appuyant sur `labels_principal.csv` plutôt que sur la détection automatique), ou les exclure comme cas marginaux ;
- la relecture manuelle d'un sous-ensemble des images marquées AMBIGU ;
- la vérification de l'hypothèse de confusion de teinte pour la catégorie NON Conforme (mesure de luminosité moyenne, à comparer entre catégories) ;
- **nouveau point identifié en section 5.3** : vérifier auprès de l'équipe si l'absence totale de NON Conforme chez la labelleuse Tilyah (31 % du dataset) reflète un lot pré-filtré ou une divergence de critère — impact potentiel sur la fiabilité du label NON Conforme pour environ un tiers des données 2025 ;
- la définition de la stratégie de fusion des sorties des deux modèles T et B (volet §4.2) ;
- l'entraînement et la comparaison des architectures de modélisation, prévus lors de la prochaine session de travail.
