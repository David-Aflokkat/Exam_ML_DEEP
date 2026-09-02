# Rapport §4.3 — Export et compatibilité production
## Projet CastagNet (MSc BIHAR, ESTIA — MESPR)

*Ce rapport fait suite aux rapports §4.1 (qualité de la donnée) et §4.2 (modélisation et comparaison d'architectures), qui laissaient en suspens le choix final entre deux candidats : le CNN « from scratch » (frugal, 0,10 Mo) et le dégel partiel d'EfficientNet-B0 (plus précis de 7 à 8 points, mais ~156 fois plus lourd et ~19 fois plus lent).*

---

## 1. Décision finale : le CNN from scratch

Le modèle retenu pour l'export et le déploiement est le **CNN « from scratch » avec rééquilibrage par flip horizontal** (`cnn_flip`), pour chacune des deux caméras (T et B). Ce choix est motivé par :

- la contrainte matérielle de production, la plus stricte du cahier des charges (GPU GTX 1060, 3 Go de VRAM, partagés avec la gestion de 12 flux caméra Ethernet) ;
- l'écart de précision avec le dégel partiel d'EfficientNet-B0 (7 à 8 points) jugé insuffisant pour justifier un coût de ressources ~156 fois supérieur, en l'absence de mesure sur le matériel cible confirmant qu'une telle marge est disponible ;
- une contrainte de temps ne permettant pas d'aller au bout de l'investigation du second candidat (l'export ONNX d'EfficientNet-B0 a d'ailleurs révélé un problème d'exporteur, cf. section 4).

Ce choix reste réversible si une évolution ultérieure du projet démontre une marge de ressources suffisante sur le matériel réel pour absorber le coût du dégel partiel.

---

## 2. Export ONNX

Le modèle `cnn_flip` a été exporté au format ONNX pour les deux caméras (`cnn_flip_camB.onnx`, `cnn_flip_camT.onnx`), avec l'ancien exporteur PyTorch basé sur TorchScript (`dynamo=False`) plutôt que le nouvel exporteur basé sur `torch.export`, activé par défaut dans les versions récentes de PyTorch — ce dernier s'est révélé instable sur des architectures plus complexes (cf. section 4), et l'exporteur legacy reste la voie la plus éprouvée pour un CNN standard de ce type.

### 2.1 Vérification de l'export

L'export n'a pas été considéré validé sur le seul critère « l'export ne plante pas » : les sorties du modèle ONNX ont été comparées numériquement à celles du modèle PyTorch d'origine, sur 20 entrées aléatoires identiques pour chaque caméra.

| Caméra | Écart max (logits) | Désaccords de classe | Résultat |
|---|---|---|---|
| B | 2,29 × 10⁻⁵ | 0 / 20 | Validé |
| T | 2,29 × 10⁻⁵ | 0 / 20 | Validé |

L'écart observé est très inférieur à la tolérance retenue (10⁻⁴) et aucun désaccord de classification n'apparaît sur l'échantillon testé : l'export ONNX reproduit fidèlement le comportement du modèle entraîné.

---

## 3. Analyse de latence et compatibilité avec la cadence de production

### 3.1 Latence mesurée

| Caméra | Latence ONNX Runtime (CPU, M1 Pro) |
|---|---|
| B | 0,39 ms/image |
| T | 0,39 ms/image |

Mesure effectuée sur processeur (pas d'accélération GPU/MPS), volontairement pour donner un ordre de grandeur conservateur en l'absence d'accès au matériel de production cible (GTX 1060 3 Go). ONNX Runtime a été préféré à une mesure PyTorch brute pour rester représentatif d'un environnement de déploiement réel.

### 3.2 Cadence requise en production

Le cahier des charges GRPTMC exprime la cadence de production en kg/h (100 kg de châtaignes sèches/heure), sans indication du poids unitaire d'un fruit permettant une conversion directe et fiable en fréquence de passages. En l'absence de cette donnée, la fréquence de passage par poste caméra observée sur l'extrait vidéo de démonstration (§4.1 : 28 passages sur 30 secondes) a été retenue comme hypothèse de travail — à confirmer ou ajuster avec une donnée de cadence réelle si elle devient disponible.

| Paramètre | Valeur |
|---|---|
| Fréquence par poste (hypothèse, d'après la vidéo démo) | 0,933 châtaigne/s |
| Nombre de postes | 6 |
| Fréquence totale de châtaignes | 5,6 châtaignes/s |
| Inférences nécessaires par châtaigne (vues T + B) | 2 |
| **Fréquence totale d'inférences nécessaire** | **11,2 inférences/s** |
| Budget de temps disponible par inférence | 89,3 ms |

### 3.3 Marge disponible

| Scénario | Latence supposée | Marge (budget / latence) | Charge résultante |
|---|---|---|---|
| Mesurée (M1 Pro, CPU) | 0,39 ms | **229×** | 0,44 % |
| Pessimiste ×10 | 3,9 ms | 22,9× | 4,4 % |
| Pessimiste ×50 | 19,5 ms | 4,6× | 21,8 % |
| Pessimiste ×100 | 39,0 ms | 2,3× | 43,7 % |

Même sous l'hypothèse volontairement pessimiste d'un matériel de production 100 fois plus lent que le M1 Pro utilisé pour la mesure — un écart bien supérieur à ce qu'on peut raisonnablement attendre entre un Mac récent et une GTX 1060, aussi ancienne soit-elle, face à un modèle de 0,10 Mo — la marge reste positive (2,3×), avec une charge de calcul de 43,7 % du temps disponible. Sur la mesure brute, la marge est considérable (229×, moins de 0,5 % de charge).

**Conclusion** : la latence d'inférence du modèle retenu n'est pas un facteur de risque pour la cadence de production, avec une marge très confortable même dans des scénarios dégradés.

---

## 4. Difficulté rencontrée : export ONNX d'EfficientNet-B0

Lors de la tentative d'export du second candidat (EfficientNet-B0, dégel partiel), le nouvel exporteur ONNX de PyTorch (basé sur `torch.export`, actif par défaut) a produit un écart numérique très important par rapport au modèle PyTorch d'origine (écart maximal de l'ordre de 3 880 sur les logits, très supérieur à la tolérance, avec un désaccord de classification sur l'échantillon de vérification) — signe d'un problème dans la traduction du graphe pour cette architecture, vraisemblablement lié aux blocs squeeze-excitation ou à une capture incorrecte du mode évaluation par ce nouvel exporteur, encore récent.

Un correctif a été identifié (forcer l'ancien exporteur via `dynamo=False`) mais n'a pas pu être validé faute de temps disponible en fin de session. Ceci n'affecte pas le modèle finalement retenu (CNN from scratch), dont l'export a été validé sans difficulté avec le même exporteur legacy dès le premier essai. Ce point reste à vérifier si le dégel partiel d'EfficientNet-B0 est reconsidéré à l'avenir (cf. section 1).

---

## 5. Recommandation finale

Le CNN « from scratch » avec rééquilibrage par flip horizontal est recommandé pour le déploiement, un modèle distinct par position de caméra (T et B), avec la stratégie de fusion des deux sorties définie au §4.2 (vote strict : Conforme si et seulement si les deux modèles sont d'accord, NON Conforme si l'un des deux le détecte, PIETRA sinon).

Points forts de cette recommandation :
- Export ONNX validé numériquement sur les deux caméras.
- Marge de latence très confortable face à la cadence de production, y compris sous hypothèses pessimistes sur le matériel cible.
- Empreinte mémoire minimale (0,10 Mo par modèle), cohérente avec la contrainte de 3 Go de VRAM partagés entre le modèle et la gestion de 12 flux caméra.

Limites explicitement assumées, à garder à l'esprit pour la suite du projet :
- La précision sur Conforme (rappel proche de la cible, précision en-deçà du seuil de 95 % exigé par le cahier des charges, cf. §4.2) n'est pas résolue par ce choix d'architecture ; elle dépend des leviers identifiés au §4.2 (fusion T/B, ajustement de seuil, pondération de la loss).
- La cadence de production utilisée en section 3.2 est une hypothèse dérivée de la vidéo de démonstration, non une donnée officielle du cahier des charges — à confirmer.
- La latence mesurée n'est pas une mesure sur le matériel de production réel (GTX 1060) ; la marge calculée, bien que confortable, reste théorique tant qu'une mesure sur ce matériel ou un équivalent n'a pas été faite.
- L'export ONNX du second candidat (EfficientNet-B0, dégel partiel) n'a pas été finalisé (section 4) ; à reprendre si ce candidat est reconsidéré.
