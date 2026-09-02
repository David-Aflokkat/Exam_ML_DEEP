"""
CastagNet - Finetuning PARTIEL d'EfficientNet-B0 (test de l'hypothèse de
décorrélation de domaine)
=================================================================================

Le finetuning à backbone entièrement gelé (finetune_pretrained.py) donne de
moins bons résultats que le CNN "from scratch", malgré le pré-entraînement
ImageNet. Hypothèse formulée : les caractéristiques de haut niveau apprises
sur des images naturelles (objets, scènes) sont décorrélées de la texture
fine propre à cette tâche (gros plan macro sur une seule texture, cadrage
circulaire artificiel introduit par le crop du §4.1) -- le simple entraînement
d'une tête linéaire sur ces caractéristiques gelées ne peut pas corriger ce
décalage.

Ce script teste cette hypothèse en dégelant les 3 DERNIERS blocs
d'EfficientNet-B0 (blocs 6, 7, 8 -- ~3,15M de paramètres sur 4,0M au total),
qui portent les caractéristiques de plus haut niveau, tout en gardant gelés
les blocs 0-5 (bas niveau : contours, dégradés, motifs simples -- plus
susceptibles de rester valables indépendamment du domaine).

Taux d'apprentissage différenciés : le backbone dégelé (déjà pré-entraîné,
ne doit pas être perturbé trop brutalement) reçoit un taux plus faible que
la tête de classification (entraînée de zéro) -- pratique standard de
finetuning.

Seul EfficientNet-B0 est concerné (meilleur des trois architectures
testées en gel complet, cf. rapport §4.2) ; MobileNetV2/V3 ne sont pas
repris ici, pour limiter le temps de calcul à la question posée.

Usage
-----
    python3 finetune_efficientnet_partial.py
    (demande le dossier des images et labels_principal.csv)
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision.models as tvm

from sklearn.metrics import accuracy_score

import mlflow
import mlflow.pytorch

sys.path.insert(0, ".")
from train_all_models import (
    build_index, split_stratified, get_device, VALID_LABELS, FLIP_CLASSES,
    log, LOG_PATH, save_confusion_matrix, log_classification_metrics,
    run_experiment_safe, BATCH_SIZE, MAX_EPOCHS, PATIENCE,
)
from finetune_pretrained import (
    PretrainedChestnutDataset, count_parameters, model_size_mb, measure_latency,
)

# Blocs d'EfficientNet-B0 à dégeler (indices dans model.features, 0 a 8).
# 6,7,8 = les 3 derniers, portant les caracteristiques de plus haut niveau.
BLOCS_A_DEGELER = [6, 7, 8]

LR_TETE = 1e-3        # tete de classification, entrainee de zero
LR_BACKBONE = 1e-4    # blocs degeles, deja pre-entraines -- taux plus faible


def build_efficientnet_partiel(n_classes):
    weights = tvm.EfficientNet_B0_Weights.DEFAULT
    model = tvm.efficientnet_b0(weights=weights)

    # Tout geler par defaut...
    for p in model.parameters():
        p.requires_grad = False

    # ... puis degeler explicitement les blocs cibles
    for i in BLOCS_A_DEGELER:
        for p in model.features[i].parameters():
            p.requires_grad = True

    # Nouvelle tete de classification, entrainable par definition
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, n_classes)

    preprocess = weights.transforms()
    return model, preprocess


def train_efficientnet_partiel(train_entries, val_entries, test_entries, class_names,
                                 run_name, experiment_name, device, flip_classes=None):
    label_to_idx = {c: i for i, c in enumerate(class_names)}

    model, preprocess = build_efficientnet_partiel(len(class_names))
    model = model.to(device)
    input_size = preprocess.crop_size[0] if hasattr(preprocess, "crop_size") else 224

    train_ds = PretrainedChestnutDataset(train_entries, label_to_idx, preprocess, flip_classes=flip_classes)
    val_ds = PretrainedChestnutDataset(val_entries, label_to_idx, preprocess)
    test_ds = PretrainedChestnutDataset(test_entries, label_to_idx, preprocess)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    total_params, trainable_params = count_parameters(model)

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("architecture", "efficientnet_b0")
        mlflow.log_param("strategie", f"degel partiel (blocs {BLOCS_A_DEGELER}) + tete")
        mlflow.log_param("lr_tete", LR_TETE)
        mlflow.log_param("lr_backbone", LR_BACKBONE)
        mlflow.log_param("input_size", input_size)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("max_epochs", MAX_EPOCHS)
        mlflow.log_param("patience", PATIENCE)
        mlflow.log_param("flip_classes", ",".join(sorted(flip_classes)) if flip_classes else "aucune")
        mlflow.log_param("n_train_images_effectif", len(train_ds))
        mlflow.log_metric("nb_parametres_total", total_params)
        mlflow.log_metric("nb_parametres_entraines", trainable_params)

        # Taux d'apprentissage differencies : backbone degele (deja
        # pre-entraine) plus prudent, tete (entrainee de zero) plus rapide.
        params_backbone = [p for i in BLOCS_A_DEGELER for p in model.features[i].parameters()]
        params_tete = list(model.classifier.parameters())
        optimizer = torch.optim.Adam([
            {"params": params_backbone, "lr": LR_BACKBONE},
            {"params": params_tete, "lr": LR_TETE},
        ])
        criterion = nn.CrossEntropyLoss()

        best_val_acc = -1.0
        best_state = None
        epochs_no_improve = 0

        for epoch in range(MAX_EPOCHS):
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * x.size(0)
                train_correct += (out.argmax(1) == y).sum().item()
                train_total += x.size(0)
            train_loss /= max(train_total, 1)
            train_acc = train_correct / max(train_total, 1)

            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    out = model(x)
                    loss = criterion(out, y)
                    val_loss += loss.item() * x.size(0)
                    val_correct += (out.argmax(1) == y).sum().item()
                    val_total += x.size(0)
            val_loss /= max(val_total, 1)
            val_acc = val_correct / max(val_total, 1)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_acc", train_acc, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)

            log(f"[{run_name}] époque {epoch+1}/{MAX_EPOCHS} "
                f"train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= PATIENCE:
                    log(f"[{run_name}] early stopping à l'époque {epoch+1}")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        mlflow.log_metric("best_val_acc", best_val_acc)

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                out = model(x)
                y_pred.extend(out.argmax(1).cpu().tolist())
                y_true.extend(y.tolist())

        test_acc = accuracy_score(y_true, y_pred)
        mlflow.log_metric("test_acc", test_acc)
        log_classification_metrics(y_true, y_pred, class_names, prefix="test_")

        cm_path = f"confusion_{run_name}.png"
        save_confusion_matrix(y_true, y_pred, class_names, cm_path,
                               f"Matrice de confusion - {run_name}")
        mlflow.log_artifact(cm_path)
        os.remove(cm_path)

        taille_mo = model_size_mb(model)
        latence_ms = measure_latency(model, device, input_size)
        mlflow.log_metric("taille_modele_mo", taille_mo)
        mlflow.log_metric("latence_ms_par_image", latence_ms)

        log(f"[{run_name}] TERMINÉ -- test_acc={test_acc:.3f}  "
            f"params_entraines={trainable_params:,}  taille={taille_mo:.2f} Mo  "
            f"latence={latence_ms:.2f} ms/image")

        model_cpu = model.to("cpu")
        mlflow.pytorch.log_model(
            model_cpu, "model",
            input_example=np.zeros((1, 3, input_size, input_size), dtype=np.float32),
            serialization_format="pickle",
        )

        return test_acc


def main():
    img_dir = input("Dossier des images d'entraînement (img_dataset_exam) : ").strip()
    labels_csv = input("Chemin de labels_principal.csv : ").strip()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    open(LOG_PATH, "a").close()
    log("=== Démarrage du finetuning partiel d'EfficientNet-B0 ===")

    device = get_device()
    log(f"Device utilisé : {device}")

    entries = build_index(img_dir, labels_csv)
    if not entries:
        log("Aucune image valide trouvée, arrêt.")
        return
    class_names = VALID_LABELS

    resultats = {}
    for position in ["B", "T"]:
        pos_entries = [e for e in entries if e["position"] == position]
        if not pos_entries:
            continue
        train_e, val_e, test_e = split_stratified(pos_entries)
        log(f"\n--- Caméra {position} : train={len(train_e)} val={len(val_e)} test={len(test_e)} ---")

        run_name = f"efficientnet_b0_partiel_cam{position}"
        acc = run_experiment_safe(
            train_efficientnet_partiel, train_e, val_e, test_e, class_names,
            run_name=run_name, experiment_name="cnn_pretrained_finetune_partiel",
            device=device, flip_classes=FLIP_CLASSES,
        )
        resultats[run_name] = acc

    log("\n=== Récapitulatif (à comparer avec efficientnet_b0 gelé, cf. rapport §4.2) ===")
    for name, acc in resultats.items():
        log(f"  {name:<30s} : {acc:.3f}" if acc is not None else f"  {name:<30s} : ÉCHEC")


if __name__ == "__main__":
    main()
