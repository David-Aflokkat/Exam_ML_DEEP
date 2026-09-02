"""
CastagNet - Finetuning d'architectures pré-entraînées + mesure des ressources
=================================================================================

Complète la comparaison du §4.2 (CNN "from scratch" + ML classique) avec
trois architectures pré-entraînées standard (torchvision, poids ImageNet),
pour trancher entre concevoir son propre modèle ou en finetuner un existant.

Le dossier `training/` mentionné dans l'énoncé n'étant pas accessible
(erreur du sujet, confirmée en cours de session), ces architectures
publiques et largement utilisées en contexte embarqué/contraint en
ressources en tiennent lieu -- choix justifié par la même contrainte
matérielle que celle qui a guidé la conception du CNN léger.

Architectures comparées (éventail volontairement large de tailles) :
  - MobileNetV3-Small : la plus légère, conçue pour le mobile/embarqué
  - MobileNetV2        : intermédiaire, standard de référence en embarqué
  - EfficientNet-B0     : plus lourde, meilleure exactitude attendue sur
                          ImageNet mais coût mémoire/calcul plus élevé

Stratégie de finetuning : le extracteur de caractéristiques (backbone) est
GELÉ (poids ImageNet conservés tels quels), seule la tête de classification
est réentraînée. Choix justifié par le volume de données disponible par
caméra (~10 000 images, modeste pour un finetuning complet d'un réseau à
plusieurs millions de paramètres) et par la contrainte de temps -- un
finetuning complet (déblocage progressif des dernières couches) est une
piste d'amélioration possible mais non retenue ici.

Mesures de ressources collectées pour chaque modèle, en plus des métriques
de classification habituelles :
  - nombre total de paramètres et nombre de paramètres entraînés (tête seule)
  - taille du modèle sérialisé (Mo)
  - latence d'inférence moyenne par image (ms), mesurée sur le device
    d'entraînement local -- indicative seulement, PAS une mesure sur le
    matériel de production cible (GTX 1060 3 Go), qui reste à faire au §4.3.

Usage
-----
    python3 finetune_pretrained.py
    (demande interactivement le dossier des images et labels_principal.csv ;
     nécessite une connexion internet au premier lancement pour télécharger
     les poids ImageNet de torchvision)
"""

import io
import os
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import torchvision.models as tvm

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

import mlflow
import mlflow.pytorch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from train_all_models import (
    build_index, split_stratified, get_device, VALID_LABELS, FLIP_CLASSES,
    log, LOG_PATH, save_confusion_matrix, log_classification_metrics,
    run_experiment_safe, BATCH_SIZE, MAX_EPOCHS, PATIENCE, LEARNING_RATE,
)


# ---------------------------------------------------------------------------
# Déclaration des architectures comparées
# ---------------------------------------------------------------------------

def get_model_specs():
    """Retourne le dict {nom: (constructeur, poids_par_defaut)}. Évalué à
    l'exécution (pas au chargement du module) pour ne planter que si on
    demande réellement à utiliser torchvision.models."""
    return {
        "mobilenet_v3_small": (tvm.mobilenet_v3_small, tvm.MobileNet_V3_Small_Weights.DEFAULT),
        "mobilenet_v2": (tvm.mobilenet_v2, tvm.MobileNet_V2_Weights.DEFAULT),
        "efficientnet_b0": (tvm.efficientnet_b0, tvm.EfficientNet_B0_Weights.DEFAULT),
    }


def build_pretrained_model(name, n_classes):
    ctor, weights = get_model_specs()[name]
    model = ctor(weights=weights)

    # Gel du backbone : seule la tête de classification sera entraînée.
    for p in model.parameters():
        p.requires_grad = False

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, n_classes)  # nouvelle tête, entraînable par défaut

    preprocess = weights.transforms()
    return model, preprocess


# ---------------------------------------------------------------------------
# Dataset adapté aux prétraitements spécifiques de chaque architecture
# ---------------------------------------------------------------------------

class PretrainedChestnutDataset(Dataset):
    def __init__(self, entries, label_to_idx, preprocess, flip_classes=None):
        self.label_to_idx = label_to_idx
        self.preprocess = preprocess
        self.items = []
        for e in entries:
            self.items.append((e, False))
            if flip_classes and e["label"] in flip_classes:
                self.items.append((e, True))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        entry, flip = self.items[idx]
        try:
            img = Image.open(entry["path"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224))
        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img = self.preprocess(img)
        label = self.label_to_idx[entry["label"]]
        return img, label


# ---------------------------------------------------------------------------
# Mesures de ressources
# ---------------------------------------------------------------------------

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    entraines = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, entraines


def model_size_mb(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell() / (1024 ** 2)


def measure_latency(model, device, input_size, n_warmup=20, n_repeats=100):
    """Latence moyenne (ms) d'une inférence à image unique (batch=1).
    Mesure indicative sur le device d'entraînement local -- PAS le
    matériel de production cible."""
    model.eval()
    dummy = torch.zeros((1, 3, input_size, input_size), device=device)
    with torch.no_grad():
        for _ in range(n_warmup):
            model(dummy)
        if device.type in ("cuda", "mps"):
            torch.cuda.synchronize() if device.type == "cuda" else None
        start = time.perf_counter()
        for _ in range(n_repeats):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    return (elapsed / n_repeats) * 1000  # ms/image


# ---------------------------------------------------------------------------
# Entraînement (tête uniquement)
# ---------------------------------------------------------------------------

def train_pretrained(train_entries, val_entries, test_entries, class_names, run_name,
                      experiment_name, device, model_name, flip_classes=None):
    label_to_idx = {c: i for i, c in enumerate(class_names)}

    model, preprocess = build_pretrained_model(model_name, len(class_names))
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
        mlflow.log_param("architecture", model_name)
        mlflow.log_param("strategie", "backbone gele, tete reentrainee")
        mlflow.log_param("input_size", input_size)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("max_epochs", MAX_EPOCHS)
        mlflow.log_param("patience", PATIENCE)
        mlflow.log_param("flip_classes", ",".join(sorted(flip_classes)) if flip_classes else "aucune")
        mlflow.log_param("n_train_images_effectif", len(train_ds))
        mlflow.log_metric("nb_parametres_total", total_params)
        mlflow.log_metric("nb_parametres_entraines", trainable_params)

        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE
        )
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

        # --- Mesures de ressources ---
        taille_mo = model_size_mb(model)
        latence_ms = measure_latency(model, device, input_size)
        mlflow.log_metric("taille_modele_mo", taille_mo)
        mlflow.log_metric("latence_ms_par_image", latence_ms)

        log(f"[{run_name}] TERMINÉ -- test_acc={test_acc:.3f}  "
            f"params_total={total_params:,}  params_entraines={trainable_params:,}  "
            f"taille={taille_mo:.2f} Mo  latence={latence_ms:.2f} ms/image")

        model_cpu = model.to("cpu")
        mlflow.pytorch.log_model(
            model_cpu, "model",
            input_example=np.zeros((1, 3, input_size, input_size), dtype=np.float32),
            serialization_format="pickle",
        )

        return {
            "test_acc": test_acc, "params_total": total_params,
            "params_entraines": trainable_params, "taille_mo": taille_mo,
            "latence_ms": latence_ms,
        }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main():
    img_dir = input("Dossier des images d'entraînement (img_dataset_exam) : ").strip()
    labels_csv = input("Chemin de labels_principal.csv : ").strip()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    open(LOG_PATH, "a").close()
    log("=== Démarrage du finetuning des architectures pré-entraînées ===")

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

        for model_name in ["mobilenet_v3_small", "mobilenet_v2", "efficientnet_b0"]:
            run_name = f"{model_name}_cam{position}"
            res = run_experiment_safe(
                train_pretrained, train_e, val_e, test_e, class_names,
                run_name=run_name, experiment_name="cnn_pretrained_finetune",
                device=device, model_name=model_name, flip_classes=FLIP_CLASSES,
            )
            resultats[run_name] = res

    log("\n=== Récapitulatif ===")
    log(f"{'Run':<28} {'test_acc':>9} {'taille (Mo)':>12} {'latence (ms)':>13} {'params entraînés':>18}")
    for name, res in resultats.items():
        if res is None:
            log(f"{name:<28} ÉCHEC")
            continue
        log(f"{name:<28} {res['test_acc']:>9.3f} {res['taille_mo']:>12.2f} "
            f"{res['latence_ms']:>13.2f} {res['params_entraines']:>18,}")

    log("\nPour consulter en détail : mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()
