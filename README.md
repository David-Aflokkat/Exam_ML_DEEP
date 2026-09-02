# Dataset Châtaignes — Images

Images de châtaignes extraites de vidéos de la machine de tri, pour l'entraînement
d'un classifieur **Conforme** / **NON Conforme** / **PIETRA**.

## Format des noms de fichiers

```
annee_label_Cam_{T|B}_{numCamera}_{numEchantillon}.jpg
```

- `T` = caméra du dessus (Top), `B` = caméra du dessous (Bottom)

## Mise en place (première fois)

1. Cloner ce repo
2. Installer DVC avec le support Google Drive :
   ```
   pipx install 'dvc[gdrive]'
   ```
3. Récupérer les images :
   ```
   dvc pull
   ```
   La première fois, un lien va s'afficher pour autoriser l'accès au Drive partagé
   "Châtaigne Corse" (compte Google de l'équipe) — ouvrir le lien dans le navigateur
   et valider.

## Workflow quotidien

- Récupérer la dernière version des images :
  ```
  dvc pull
  ```
- Après avoir ajouté ou modifié des images dans `images/` :
  ```
  dvc add images
  git add images.dvc
  git commit -m "Ajout de nouvelles images"
  dvc push
  ```

## Labelisation

Le fichier `labels_principal.csv` contient le label d'entraînement
(`label_principal` : Conforme / NON Conforme / PIETRA / Vide) et des tags
(`multiple`, `chunk`, `mixed_quality`) pour chaque image. Voir
[`labeling_tool/README.md`](labeling_tool/README.md) pour l'outil Streamlit
de relecture/labelisation.

## Contenu du dépôt

- `images/` — les images du dataset (suivies par DVC, pas par git)
- `images.dvc` — pointeur de version vers le dataset (suivi par git)
- `labels_masked.csv` — annotations vide/châtaigne par image (`label`, `hot_frac`, `max_blob`), ne pas modifier
- `labels_principal.csv` — labels d'entraînement (`label_principal`) et tags, mis à jour via `labeling_tool/`
- `labeling_tool/` — outil Streamlit de labelisation (voir son README)
- `Rapport_Dataset_Chataignes.pdf` — rapport statistique (répartition par label, année, caméra)
