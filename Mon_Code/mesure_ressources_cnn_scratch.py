"""
CastagNet - Mesure des ressources (taille, latence) des CNN from scratch
============================================================================

Les runs cnn_baseline/cnn_flip ont été entraînés avant l'ajout des mesures
de ressources (introduites pour la comparaison avec les architectures
pré-entraînées, cf. finetune_pretrained.py). Ce script recharge les
modèles déjà entraînés et sauvegardés dans MLflow, et calcule les mêmes
mesures (taille du modèle sérialisé, latence d'inférence par image) pour
compléter la comparaison à effort constant.

Usage
-----
    python3 mesure_ressources_cnn_scratch.py
    (demande le tracking URI MLflow et les run_id des modèles à mesurer)
"""

import sys

import mlflow
import mlflow.pytorch

sys.path.insert(0, ".")
from train_all_models import get_device, RESOLUTION, log
from finetune_pretrained import model_size_mb, measure_latency


def main():
    tracking_uri = input("Tracking URI MLflow (ex: sqlite:///mlflow.db) : ").strip()
    if not tracking_uri.startswith("sqlite:") and not tracking_uri.startswith("file:"):
        tracking_uri = f"sqlite:///{tracking_uri}"
    mlflow.set_tracking_uri(tracking_uri)

    device = get_device()
    print(f"Device : {device}")

    run_ids_str = input(
        "Liste des run_id à mesurer, format 'nom=run_id', séparés par des virgules "
        "(ex: cnn_baseline_camB=xxx,cnn_flip_camB=yyy,...) : "
    ).strip()

    pairs = [p.split("=") for p in run_ids_str.split(",")]

    resultats = []
    for name, run_id in pairs:
        name, run_id = name.strip(), run_id.strip()
        print(f"\nChargement de {name} ({run_id})...")
        model = mlflow.pytorch.load_model(f"runs:/{run_id}/model").to(device)

        taille_mo = model_size_mb(model)
        latence_ms = measure_latency(model, device, RESOLUTION)
        resultats.append((name, taille_mo, latence_ms))
        print(f"  taille={taille_mo:.2f} Mo   latence={latence_ms:.2f} ms/image (résolution {RESOLUTION})")

        # Reversement dans le run existant, pour que mlflow ui affiche tout au même endroit
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric("taille_modele_mo", taille_mo)
            mlflow.log_metric("latence_ms_par_image", latence_ms)

    print(f"\n{'Run':<25} {'Taille (Mo)':>12} {'Latence (ms)':>13}")
    for name, taille, latence in resultats:
        print(f"{name:<25} {taille:>12.2f} {latence:>13.2f}")


if __name__ == "__main__":
    main()
