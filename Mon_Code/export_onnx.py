"""
CastagNet - §4.3 : Export ONNX, vérification et mesure de latence
======================================================================

Exporte au format ONNX les modèles candidats retenus à l'issue du §4.2
(CNN from scratch et EfficientNet-B0 en dégel partiel, les deux options
laissées en balance à la fin du rapport §4.2), vérifie numériquement que
l'export ONNX produit les mêmes prédictions que le modèle PyTorch d'origine
(pas seulement que l'export "ne plante pas"), puis mesure la latence
d'inférence via ONNX Runtime -- plus représentatif d'un déploiement réel
qu'une mesure PyTorch brute.

Usage
-----
    python3 export_onnx.py
    (demande le tracking URI MLflow, le run_id du modèle à exporter, sa
     résolution d'entrée, et le chemin de sortie du fichier .onnx)
"""

import sys
import time

import numpy as np
import torch

import onnx
import onnxruntime as ort

import mlflow
import mlflow.pytorch

sys.path.insert(0, ".")
from train_all_models import get_device


def exporter_onnx(model, resolution, chemin_sortie, opset=17):
    """Exporte le modèle PyTorch (déjà en mode eval, sur CPU) au format ONNX."""
    model.eval()
    dummy_input = torch.zeros((1, 3, resolution, resolution), dtype=torch.float32)

    torch.onnx.export(
        model, dummy_input, chemin_sortie,
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
    )

    # Vérification structurelle du graphe ONNX (schéma valide, pas de noeud orphelin)
    modele_onnx = onnx.load(chemin_sortie)
    onnx.checker.check_model(modele_onnx)
    print(f"Export ONNX -> {chemin_sortie} (structure vérifiée par onnx.checker)")


def verifier_equivalence(model_pytorch, chemin_onnx, resolution, n_echantillons=20, tol=1e-4):
    """Compare les prédictions PyTorch et ONNX Runtime sur des entrées
    aléatoires identiques : l'export n'est validé que si les deux
    concordent numériquement, pas seulement si l'export s'est déroulé
    sans erreur."""
    model_pytorch.eval()
    session = ort.InferenceSession(chemin_onnx, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    ecarts_max = []
    desaccords_classe = 0

    with torch.no_grad():
        for _ in range(n_echantillons):
            x = torch.rand((1, 3, resolution, resolution), dtype=torch.float32)

            out_pytorch = model_pytorch(x).numpy()
            out_onnx = session.run(None, {input_name: x.numpy()})[0]

            ecart = np.abs(out_pytorch - out_onnx).max()
            ecarts_max.append(ecart)

            if out_pytorch.argmax(axis=1)[0] != out_onnx.argmax(axis=1)[0]:
                desaccords_classe += 1

    ecart_max_observe = max(ecarts_max)
    ok = ecart_max_observe < tol and desaccords_classe == 0

    print(f"\nVérification d'équivalence PyTorch <-> ONNX sur {n_echantillons} entrées aléatoires :")
    print(f"  Écart numérique maximal (logits)   : {ecart_max_observe:.2e}  (tolérance : {tol:.0e})")
    print(f"  Désaccords de classe prédite       : {desaccords_classe}/{n_echantillons}")
    print(f"  Résultat : {'OK -- export validé' if ok else 'ÉCHEC -- écart trop important, ne pas déployer tel quel'}")

    return ok


def mesurer_latence_onnx(chemin_onnx, resolution, n_warmup=20, n_repeats=200):
    """Latence moyenne (ms) d'une inférence à image unique via ONNX Runtime
    CPU -- volontairement CPU (pas MPS/CUDA) car le matériel de production
    cible (GTX 1060 3 Go) n'est pas disponible pour test direct ; le CPU
    donne un ordre de grandeur conservateur, à affiner sur le matériel réel."""
    session = ort.InferenceSession(chemin_onnx, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    dummy = np.zeros((1, 3, resolution, resolution), dtype=np.float32)

    for _ in range(n_warmup):
        session.run(None, {input_name: dummy})

    start = time.perf_counter()
    for _ in range(n_repeats):
        session.run(None, {input_name: dummy})
    elapsed = time.perf_counter() - start

    return (elapsed / n_repeats) * 1000  # ms/image


def main():
    tracking_uri = input("Tracking URI MLflow (ex: sqlite:///mlflow.db) : ").strip()
    if not tracking_uri.startswith("sqlite:") and not tracking_uri.startswith("file:"):
        tracking_uri = f"sqlite:///{tracking_uri}"
    mlflow.set_tracking_uri(tracking_uri)

    run_id = input("run_id du modèle à exporter : ").strip()
    resolution = int(input("Résolution d'entrée du modèle (128 pour CNN maison, 224 pour EfficientNet) : ").strip())
    chemin_sortie = input("Chemin de sortie du fichier .onnx (ex: model_camB.onnx) : ").strip()

    print("Chargement du modèle PyTorch depuis MLflow...")
    model = mlflow.pytorch.load_model(f"runs:/{run_id}/model").to("cpu")

    exporter_onnx(model, resolution, chemin_sortie)
    ok = verifier_equivalence(model, chemin_sortie, resolution)

    if not ok:
        print("\n[ATTENTION] L'export ne passe pas la vérification numérique -- "
              "ne pas utiliser ce fichier .onnx en l'état.")
        return

    latence_ms = mesurer_latence_onnx(chemin_sortie, resolution)
    print(f"\nLatence ONNX Runtime (CPU) : {latence_ms:.2f} ms/image")
    print("Rappel : mesure sur CPU, pas sur le GPU de production (GTX 1060) -- "
          "ordre de grandeur conservateur, à affiner si le matériel cible devient accessible.")


if __name__ == "__main__":
    main()
