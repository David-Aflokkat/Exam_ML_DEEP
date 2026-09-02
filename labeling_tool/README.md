# Outil de labelisation

Petite application Streamlit pour relire les images et renseigner `labels_principal.csv`
(le fichier de labels utilisé pour entraîner le classifieur — distinct de
`labels_masked.csv`, qui ne doit pas être modifié).

## Installation

```
pip install -r labeling_tool/requirements.txt
```

## Initialiser / mettre à jour le fichier de labels

À lancer une fois pour créer `labels_principal.csv` (déjà fait), et à relancer
si de nouvelles images sont ajoutées dans `images/` (n'écrase jamais les
lignes déjà labelisées) :

```
python labeling_tool/build_labels_csv.py
```

## Lancer l'outil

```
streamlit run labeling_tool/app.py
```

## Utilisation

1. Renseigne ton nom/pseudo dans la barre latérale (enregistré dans la colonne
   `labeled_by`).
2. Filtre par **année**, **catégorie** (déduite du nom de fichier), **position
   caméra** (T/B) et **numéro de caméra** pour travailler par paquets.
3. Pour chaque image :
   - coche les cases **Plusieurs châtaignes** / **Morceau / débris** /
     **Qualité mixte** si applicable,
   - clique sur le label principal (**Conforme**, **NON Conforme**, **PIETRA**,
     **Vide**) si le label pré-rempli (déduit du nom de fichier) doit être
     corrigé, ou sur **Suivant — confirmer le label affiché** si le label
     pré-rempli est correct.
4. Dans les deux cas, la ligne est immédiatement sauvegardée dans
   `labels_principal.csv` (`reviewed=True`, `labeled_by=ton pseudo`) et
   l'image suivante s'affiche.

Décocher "Afficher seulement les images non revues" permet de revenir en
arrière et de corriger une image déjà labelisée.

## Schéma de `labels_principal.csv`

| Colonne | Description |
|---|---|
| `filename` | nom du fichier image |
| `year`, `cam_position`, `cam_num`, `sample_num` | déduits du nom de fichier |
| `label_filename` | label déduit du nom de fichier (référence, jamais modifié) |
| `label_principal` | **cible d'entraînement** : `Conforme` / `NON Conforme` / `PIETRA` / `Vide` |
| `multiple` | plusieurs châtaignes visibles sur l'image |
| `chunk` | présence d'un morceau / débris (pas une châtaigne entière) |
| `mixed_quality` | mélange de qualités sur la même image (ex: 1 conforme + 1 piétra) |
| `reviewed` | l'image a été relue/validée par quelqu'un de l'équipe |
| `labeled_by` | pseudo de la personne ayant validé la ligne |

## Travailler à plusieurs

Pour limiter les conflits git sur `labels_principal.csv`, chaque personne peut
se réserver une tranche (ex : une année + une caméra) via les filtres, puis
committer/pousser son travail régulièrement :

```
git add labels_principal.csv
git commit -m "Labelisation: caméra 3, 2025"
git push
```
