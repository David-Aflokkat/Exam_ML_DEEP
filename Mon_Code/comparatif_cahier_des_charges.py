"""
CastagNet - Comparatif des 6 runs au regard du cahier des charges GRPTMC
============================================================================

L'exactitude globale ne suffit pas à juger un modèle ici : le cahier des
charges impose des seuils spécifiques sur la classe Conforme (précision
>= 95%, rappel >= 85%). Ce script interroge MLflow et affiche, pour
chaque run déjà entraîné, l'exactitude globale ET les métriques Conforme,
avec un statut de conformité explicite.

Usage
-----
    python3 comparatif_cahier_des_charges.py
    (demande le chemin du mlflow.db)
"""

import mlflow

SEUIL_PRECISION_CONFORME = 0.95
SEUIL_RAPPEL_CONFORME = 0.85


def main():
    db_path = input("Chemin du fichier mlflow.db (ex: sqlite:///mlflow.db) : ").strip()
    if not db_path.startswith("sqlite:"):
        db_path = f"sqlite:///{db_path}"
    mlflow.set_tracking_uri(db_path)

    client = mlflow.tracking.MlflowClient()

    print(f"{'Run':<22} {'Exactitude':>11} {'Précision Conforme':>20} {'Rappel Conforme':>17} {'Conforme cahier des charges ?':>30}")
    print("-" * 105)

    rows = []
    for exp in client.search_experiments():
        for run in client.search_runs(exp.experiment_id):
            metrics = run.data.metrics
            test_acc = metrics.get("test_acc")
            precision_conf = metrics.get("test_precision_Conforme")
            rappel_conf = metrics.get("test_recall_Conforme")
            if test_acc is None:
                continue
            rows.append((run.info.run_name, test_acc, precision_conf, rappel_conf))

    for name, acc, prec, rap in sorted(rows, key=lambda r: (-r[1] if r[1] else 0)):
        if prec is not None and rap is not None:
            ok_prec = prec >= SEUIL_PRECISION_CONFORME
            ok_rap = rap >= SEUIL_RAPPEL_CONFORME
            statut = "OUI" if (ok_prec and ok_rap) else "NON"
            if not ok_prec:
                statut += f" (précision {prec:.1%} < 95%)"
            if not ok_rap:
                statut += f" (rappel {rap:.1%} < 85%)"
            print(f"{name:<22} {acc:>10.1%} {prec:>19.1%} {rap:>16.1%} {statut:>30}")
        else:
            print(f"{name:<22} {acc:>10.1%} {'N/A':>19} {'N/A':>16} {'métrique absente (ML classique ?)':>30}")

    print("\nRappel : seuils cahier des charges GRPTMC -- précision Conforme >= 95%, rappel Conforme >= 85%.")


if __name__ == "__main__":
    main()
