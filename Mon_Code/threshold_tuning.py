"""
CastagNet - Ajustement du seuil de décision sur la classe Conforme
=====================================================================

Les modèles entraînés (train_all_models.py) utilisent l'argmax standard
pour décider de la classe prédite, ce qui ne tient pas compte de
l'asymétrie de coût du cahier des charges (précision Conforme >= 95%,
rappel >= 85%). Ce script recharge un modèle déjà entraîné, reconstruit
EXACTEMENT le même split de test (même seed, même logique de split que
train_all_models.py), et balaie un seuil de décision sur la probabilité
de la classe Conforme : on ne prédit "Conforme" que si sa probabilité
dépasse le seuil, sinon on bascule sur la seconde classe la plus probable.

Pour chaque seuil testé, précision et rappel sur Conforme sont recalculés,
afin de trouver s'il existe un point respectant les deux contraintes du
cahier des charges simultanément.

Usage
-----
    python3 threshold_tuning.py
    (demande le tracking URI MLflow, le run_id du modèle, le dossier
     d'images, le CSV de vérité terrain, et la caméra concernée)
"""

import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

import mlflow
import mlflow.pytorch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from train_all_models import (
    build_index, split_stratified, ChestnutDataset, get_device, VALID_LABELS,
)

SEUIL_PRECISION_CIBLE = 0.95
SEUIL_RAPPEL_CIBLE = 0.85


def get_test_probs(model, test_entries, class_names, device):
    """Reconstruit le jeu de test et retourne (y_true, probs) où probs est
    un tableau (n, n_classes) de probabilités softmax."""
    label_to_idx = {c: i for i, c in enumerate(class_names)}
    test_ds = ChestnutDataset(test_entries, label_to_idx)
    loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    model.eval()
    y_true, probs = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            p = torch.softmax(out, dim=1).cpu().numpy()
            probs.append(p)
            y_true.extend(y.tolist())
    return np.array(y_true), np.concatenate(probs, axis=0)


def evaluate_threshold(y_true, probs, conforme_idx, threshold):
    """Prédiction avec seuil sur la classe Conforme : Conforme seulement si
    sa probabilité dépasse `threshold`, sinon la 2e classe la plus probable
    l'emporte."""
    n = len(y_true)
    y_pred = np.zeros(n, dtype=int)
    for i in range(n):
        p = probs[i]
        if p[conforme_idx] >= threshold:
            y_pred[i] = conforme_idx
        else:
            # meilleure classe parmi les autres (Conforme exclue)
            alt = p.copy()
            alt[conforme_idx] = -1
            y_pred[i] = alt.argmax()

    y_true_bin = (y_true == conforme_idx)
    y_pred_bin = (y_pred == conforme_idx)

    tp = np.sum(y_true_bin & y_pred_bin)
    fp = np.sum(~y_true_bin & y_pred_bin)
    fn = np.sum(y_true_bin & ~y_pred_bin)

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rappel = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    return precision, rappel


def main():
    tracking_uri = input("Tracking URI MLflow (ex: sqlite:///mlflow.db) : ").strip()
    if not tracking_uri.startswith("sqlite:") and not tracking_uri.startswith("file:"):
        tracking_uri = f"sqlite:///{tracking_uri}"
    mlflow.set_tracking_uri(tracking_uri)

    run_id = input("run_id du modèle à évaluer (visible dans mlflow ui) : ").strip()
    img_dir = input("Dossier des images (img_dataset_exam) : ").strip()
    labels_csv = input("Chemin de labels_principal.csv : ").strip()
    position = input("Caméra concernée par ce run (T ou B) : ").strip().upper()

    device = get_device()
    print("Chargement du modèle...")
    model = mlflow.pytorch.load_model(f"runs:/{run_id}/model").to(device)

    entries = build_index(img_dir, labels_csv)
    pos_entries = [e for e in entries if e["position"] == position]
    # Important : reprendre EXACTEMENT le même ordre de classes qu'à
    # l'entraînement (VALID_LABELS, fixe), pas un recalcul à partir des
    # seules données de cette caméra -- un ordre différent décalerait les
    # indices de sortie du modèle chargé et fausserait silencieusement
    # tous les résultats.
    class_names = VALID_LABELS
    _, _, test_entries = split_stratified(pos_entries)  # même seed que l'entraînement

    print(f"Jeu de test reconstruit : {len(test_entries)} image(s).")
    y_true, probs = get_test_probs(model, test_entries, class_names, device)

    conforme_idx = class_names.index("Conforme")

    thresholds = np.arange(0.30, 1.00, 0.02)
    precisions, rappels = [], []
    meilleurs = []

    print(f"\n{'Seuil':>6} {'Précision Conforme':>20} {'Rappel Conforme':>17} {'Conforme cahier des charges ?':>15}")
    for t in thresholds:
        p, r = evaluate_threshold(y_true, probs, conforme_idx, t)
        precisions.append(p)
        rappels.append(r)
        ok = (p >= SEUIL_PRECISION_CIBLE) and (r >= SEUIL_RAPPEL_CIBLE)
        if ok:
            meilleurs.append((t, p, r))
        marque = "  <-- OK" if ok else ""
        print(f"{t:>6.2f} {p:>19.1%} {r:>16.1%}{marque}")

    if meilleurs:
        print(f"\n{len(meilleurs)} seuil(s) satisfont les deux critères simultanément :")
        for t, p, r in meilleurs:
            print(f"  seuil={t:.2f} -> précision={p:.1%} rappel={r:.1%}")
    else:
        print("\nAucun seuil ne satisfait simultanément précision >= 95% ET rappel >= 85% "
              "sur ce modèle. Le compromis precision/rappel disponible est insuffisant : "
              "il faudra probablement agir sur l'entraînement lui-même (pondération de "
              "classe dans la loss, plus de données, architecture différente) plutôt que "
              "sur le seul seuil de décision.")

    # Graphique du compromis precision/rappel
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(thresholds, precisions, label="Précision Conforme", marker="o", ms=3)
    ax.plot(thresholds, rappels, label="Rappel Conforme", marker="o", ms=3)
    ax.axhline(SEUIL_PRECISION_CIBLE, color="green", linestyle="--", label="Cible précision (95%)")
    ax.axhline(SEUIL_RAPPEL_CIBLE, color="orange", linestyle="--", label="Cible rappel (85%)")
    ax.set_xlabel("Seuil de décision sur P(Conforme)")
    ax.set_ylabel("Valeur")
    ax.set_title(f"Compromis précision/rappel Conforme selon le seuil ({position})")
    ax.legend()
    fig.tight_layout()
    out_path = f"seuil_conforme_{position}.png"
    fig.savefig(out_path, dpi=120)
    print(f"\nGraphique -> {out_path}")


if __name__ == "__main__":
    main()
