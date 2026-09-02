"""
CastagNet - Décomposition des erreurs "prédit Conforme" par vraie classe
============================================================================

Teste une hypothèse structurelle avant d'investir dans un réentraînement :
si PIETRA n'est pas un critère visuel (c'est une destination commerciale,
cf. rapport §4.1) mais un critère purement administratif, il est possible
qu'aucun modèle basé sur l'image ne puisse le distinguer fiablement de
Conforme -- auquel cas le plafond de précision observé serait une limite
structurelle de l'information disponible, pas un défaut d'entraînement
corrigible par plus de données ou une meilleure architecture.

Ce script recharge un modèle déjà entraîné, l'évalue sur son jeu de test
(argmax standard), et décompose les faux positifs "Conforme" (prédiction
Conforme, vraie classe différente) selon leur vraie classe réelle.

Usage
-----
    python3 analyse_erreurs_conforme.py
"""

import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

import mlflow
import mlflow.pytorch

sys.path.insert(0, ".")
from train_all_models import build_index, split_stratified, ChestnutDataset, get_device, VALID_LABELS


def main():
    tracking_uri = input("Tracking URI MLflow (ex: sqlite:///mlflow.db) : ").strip()
    if not tracking_uri.startswith("sqlite:") and not tracking_uri.startswith("file:"):
        tracking_uri = f"sqlite:///{tracking_uri}"
    mlflow.set_tracking_uri(tracking_uri)

    run_id = input("run_id du modèle à évaluer : ").strip()
    img_dir = input("Dossier des images (img_dataset_exam) : ").strip()
    labels_csv = input("Chemin de labels_principal.csv : ").strip()
    position = input("Caméra concernée par ce run (T ou B) : ").strip().upper()

    device = get_device()
    print("Chargement du modèle...")
    model = mlflow.pytorch.load_model(f"runs:/{run_id}/model").to(device)
    model.eval()

    entries = build_index(img_dir, labels_csv)
    pos_entries = [e for e in entries if e["position"] == position]
    class_names = VALID_LABELS  # même ordre qu'à l'entraînement, cf. threshold_tuning.py
    _, _, test_entries = split_stratified(pos_entries)

    label_to_idx = {c: i for i, c in enumerate(class_names)}
    test_ds = ChestnutDataset(test_entries, label_to_idx)
    loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            y_pred.extend(out.argmax(1).cpu().tolist())
            y_true.extend(y.tolist())

    conforme_idx = class_names.index("Conforme")
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Faux positifs Conforme : prédit Conforme, vraie classe différente
    faux_positifs_mask = (y_pred == conforme_idx) & (y_true != conforme_idx)
    n_fp = faux_positifs_mask.sum()

    print(f"\nNombre total de faux positifs Conforme (test) : {n_fp}")
    print("\nRépartition de ces faux positifs par vraie classe :")
    for i, cname in enumerate(class_names):
        if i == conforme_idx:
            continue
        n = np.sum(faux_positifs_mask & (y_true == i))
        n_total_classe = np.sum(y_true == i)
        pct_parmi_fp = 100 * n / n_fp if n_fp else 0
        pct_de_la_classe = 100 * n / n_total_classe if n_total_classe else 0
        print(f"  {cname:<15} : {n:4d} cas ({pct_parmi_fp:.1f}% des faux positifs, "
              f"soit {pct_de_la_classe:.1f}% de tous les {cname} du jeu de test mal classés Conforme)")

    print("\nInterprétation :")
    print("- Si les faux positifs sont concentrés sur PIETRA : cohérent avec l'hypothèse")
    print("  d'une limite structurelle (PIETRA visuellement indiscernable de Conforme).")
    print("- Si NON Conforme est fortement représenté aussi : le problème est plus")
    print("  probablement corrigible par l'entraînement (pondération, données, archi).")


if __name__ == "__main__":
    main()
