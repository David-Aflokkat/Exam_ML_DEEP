"""
CastagNet - EfficientNet-B0 (backbone gelé) sur un dossier d'images au choix
================================================================================

Version réduite de finetune_pretrained.py : UNIQUEMENT EfficientNet-B0,
backbone entièrement gelé (comme dans la campagne initiale), pour comparer
à isostratégie de dégel les images brutes (dossier `images/`) et les
images traitées (`img_dataset_exam/`, crop tangent au cercle du §4.1).

Rappel des résultats déjà obtenus :
    - backbone gelé × images BRUTES (dossier `images/`) :
          efficientnet_b0_camB : test_acc = 0.693
          efficientnet_b0_camT : test_acc = 0.712
    - backbone PARTIEL dégelé × images TRAITÉES (`img_dataset_exam/`) :
          efficientnet_b0_partiel_camB : test_acc = 0.788
          efficientnet_b0_partiel_camT : test_acc = 0.789

Point manquant pour la comparaison 2x2 (gelé/dégelé x brut/traité) : le
backbone GELÉ sur images TRAITÉES -- c'est ce que ce script doit produire
en le lançant avec le dossier `img_dataset_exam/` (PAS `images/`, qui a
déjà été utilisé par erreur pour la toute première campagne).

Usage
-----
    python3 finetune_efficientnet_frozen_only.py
    (demande le dossier des images -- brut OU traité, au choix -- et
     labels_principal.csv)
"""

import sys

import mlflow

sys.path.insert(0, ".")
from train_all_models import (
    build_index, split_stratified, get_device, VALID_LABELS, FLIP_CLASSES,
    log, LOG_PATH, run_experiment_safe,
)
from finetune_pretrained import train_pretrained


def main():
    img_dir = input("Dossier des images (brut = 'images', traité = 'img_dataset_exam') : ").strip()
    labels_csv = input("Chemin de labels_principal.csv : ").strip()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    open(LOG_PATH, "a").close()
    log(f"=== EfficientNet-B0 (backbone gelé) sur {img_dir} ===")

    device = get_device()
    log(f"Device utilisé : {device}")

    entries = build_index(img_dir, labels_csv)
    if not entries:
        log("Aucune image valide trouvée, arrêt.")
        return
    class_names = VALID_LABELS

    # Nom de dossier repris dans le nom du run, pour distinguer sans ambiguïté
    # brut et traité dans MLflow.
    suffixe = img_dir.rstrip("/").split("/")[-1]

    resultats = {}
    for position in ["B", "T"]:
        pos_entries = [e for e in entries if e["position"] == position]
        if not pos_entries:
            continue
        train_e, val_e, test_e = split_stratified(pos_entries)
        log(f"\n--- Caméra {position} : train={len(train_e)} val={len(val_e)} test={len(test_e)} ---")

        run_name = f"efficientnet_b0_gele_{suffixe}_cam{position}"
        res = run_experiment_safe(
            train_pretrained, train_e, val_e, test_e, class_names,
            run_name=run_name, experiment_name="cnn_pretrained_finetune",
            device=device, model_name="efficientnet_b0", flip_classes=FLIP_CLASSES,
        )
        resultats[run_name] = res

    log(f"\n=== Récapitulatif (backbone gelé, dossier={img_dir}) ===")
    for name, res in resultats.items():
        if res is None:
            log(f"  {name:<40s} : ÉCHEC")
        else:
            log(f"  {name:<40s} : test_acc={res['test_acc']:.3f}")

    log("\nÀ comparer avec :")
    log("  - backbone gelé x BRUT (déjà obtenu)   : efficientnet_b0 camB=0.693 camT=0.712")
    log("  - backbone partiel x TRAITÉ (déjà obtenu) : efficientnet_b0 camB=0.788 camT=0.789")


if __name__ == "__main__":
    main()
