"""
CastagNet - Fusion des prédictions T/B et estimation théorique du gain
==========================================================================

Règle de fusion retenue (sur décision explicite, suite au diagnostic
montrant que 78,3 % des faux positifs Conforme du modèle B seul sont en
réalité des PIETRA -- cohérent avec le fait que le tan peut n'être visible
que d'un seul côté du fruit) :

    - Les deux modèles prédisent Conforme       -> Conforme
    - Au moins un modèle prédit NON Conforme    -> NON Conforme (priorité
      absolue : aucun défaut grave ne doit passer, même détecté par une
      seule caméra)
    - Tout le reste (désaccord sans NON Conforme) -> PIETRA (filet de
      sécurité : capture notamment le cas où une caméra voit le tan et
      l'autre non)

Ce module fournit :
  1. `fusionner(pred_t, pred_b)` : la règle elle-même, déployable telle
     quelle une fois une vraie correspondance T/B disponible en production
     (flux synchronisés en temps réel, contrairement au dataset labellisé).
  2. Une SIMULATION du gain de précision/rappel attendu sur Conforme après
     fusion, à partir des matrices de confusion réelles des deux modèles
     déjà entraînés (cnn_baseline ou cnn_flip, au choix), sous l'hypothèse
     que les erreurs de classification de T et B sont indépendantes
     conditionnellement à la vraie classe du fruit.

    ATTENTION -- cette simulation est une ESTIMATION, pas une mesure : on
    ne dispose d'aucune paire T/B réelle sur ce dataset (cf. rapport §4.1,
    décision d'abandon de la correspondance). L'hypothèse d'indépendance
    est plausible (le côté du fruit où le tan est visible n'a pas de
    raison de dépendre de la caméra) mais non vérifiée. Un vrai chiffre ne
    pourra être obtenu qu'avec des données appariées en production.

Usage
-----
    python3 fusion_tb.py
    (demande les run_id des modèles T et B à comparer, dans MLflow)
"""

import sys
from itertools import product

import numpy as np
import torch
from torch.utils.data import DataLoader

import mlflow
import mlflow.pytorch

sys.path.insert(0, ".")
from train_all_models import build_index, split_stratified, ChestnutDataset, get_device, VALID_LABELS


# ---------------------------------------------------------------------------
# 1. La règle de fusion (déployable en production)
# ---------------------------------------------------------------------------

def fusionner(pred_t, pred_b):
    """Applique la règle de fusion retenue à deux prédictions (chaînes de
    caractères parmi 'Conforme', 'NON Conforme', 'PIETRA')."""
    if pred_t == "Conforme" and pred_b == "Conforme":
        return "Conforme"
    if pred_t == "NON Conforme" or pred_b == "NON Conforme":
        return "NON Conforme"
    return "PIETRA"


def fusionner_pondere(proba_t, proba_b, class_names,
                       seuil_conforme=0.5, seuil_veto_non_conforme=None):
    """Variante pondérée par la confiance des deux modèles.

    Le veto NON Conforme reste un vote dur et prioritaire (aucune raison
    de l'assouplir : c'est l'erreur grave du cahier des charges, à garder
    stricte même si un seul modèle n'est que modérément confiant) --
    optionnellement lui-même soumis à un seuil de confiance minimal
    (`seuil_veto_non_conforme`) si on veut éviter qu'un NON Conforme
    proposé à faible confiance déclenche le veto.

    Pour Conforme, au lieu d'exiger que les deux votes soient
    individuellement "Conforme" (ce qui traite un 98%/98% et un 51%/51%
    de façon identique, et surtout envoie en PIETRA un cas 98% Conforme /
    51% PIETRA qui mérite sans doute d'être reconsidéré), on combine les
    deux probabilités de Conforme par moyenne géométrique -- une façon
    simple de faire converger les deux avis sans qu'un score très faible
    d'un côté soit totalement écrasé par un score très fort de l'autre,
    ni l'inverse. Conforme est retenu si ce score combiné dépasse
    `seuil_conforme`.

    proba_t, proba_b : vecteurs de probabilités (softmax) dans l'ordre de
    class_names, pour une même châtaigne vue par T et par B.
    """
    conforme_idx = class_names.index("Conforme")
    non_conforme_idx = class_names.index("NON Conforme")

    p_nc_t, p_nc_b = proba_t[non_conforme_idx], proba_b[non_conforme_idx]
    veto_seuil = seuil_veto_non_conforme if seuil_veto_non_conforme is not None else 0.0
    if p_nc_t > veto_seuil and np.argmax(proba_t) == non_conforme_idx:
        return "NON Conforme"
    if p_nc_b > veto_seuil and np.argmax(proba_b) == non_conforme_idx:
        return "NON Conforme"

    score_conforme = (proba_t[conforme_idx] * proba_b[conforme_idx]) ** 0.5  # moyenne geometrique
    if score_conforme >= seuil_conforme:
        return "Conforme"
    return "PIETRA"


# ---------------------------------------------------------------------------
# 2. Simulation du gain attendu (indépendance T/B supposée)
# ---------------------------------------------------------------------------

def get_confusion_probabilities(model, entries, class_names, device, threshold=None):
    """Retourne P(prédit=x | vrai=y) pour chaque paire (y, x), à partir du
    jeu de test réel du modèle. Si `threshold` est fourni, la décision
    n'est plus l'argmax standard : Conforme n'est retenu que si sa
    probabilité dépasse ce seuil (sinon la 2e classe la plus probable
    l'emporte) -- même logique que threshold_tuning.py, pour explorer le
    compromis précision/rappel de CHAQUE modèle avant fusion."""
    label_to_idx = {c: i for i, c in enumerate(class_names)}
    ds = ChestnutDataset(entries, label_to_idx)
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    conforme_idx = class_names.index("Conforme")

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            if threshold is None:
                pred = out.argmax(1).cpu().numpy()
            else:
                p = torch.softmax(out, dim=1).cpu().numpy()
                pred = np.zeros(len(p), dtype=int)
                for i in range(len(p)):
                    if p[i, conforme_idx] >= threshold:
                        pred[i] = conforme_idx
                    else:
                        alt = p[i].copy()
                        alt[conforme_idx] = -1
                        pred[i] = alt.argmax()
            y_pred.extend(pred.tolist())
            y_true.extend(y.tolist())
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    n = len(class_names)
    probs = np.zeros((n, n))  # probs[y, x] = P(predit=x | vrai=y)
    for y in range(n):
        mask = y_true == y
        total = mask.sum()
        if total == 0:
            continue
        for x in range(n):
            probs[y, x] = np.sum((y_pred == x) & mask) / total
    return probs, y_true


def simulate_fusion(probs_t, probs_b, class_names, priors):
    """Simule la fusion sous hypothèse d'indépendance conditionnelle des
    prédictions T et B sachant la vraie classe. Retourne précision et
    rappel simulés pour Conforme, ainsi que la matrice de confusion
    simulée complète (vrai x fusionné)."""
    n = len(class_names)
    conforme_idx = class_names.index("Conforme")

    # matrice de confusion simulee : confusion_sim[y, z] = P(fusionne=z | vrai=y)
    confusion_sim = np.zeros((n, n))
    for y in range(n):
        for xt, xb in product(range(n), range(n)):
            p_joint = probs_t[y, xt] * probs_b[y, xb]  # independance supposee
            z = class_names.index(fusionner(class_names[xt], class_names[xb]))
            confusion_sim[y, z] += p_joint

    # Precision/rappel Conforme simules, ponderes par les priors (proportions reelles)
    vrai_positifs = priors[conforme_idx] * confusion_sim[conforme_idx, conforme_idx]
    faux_positifs = sum(priors[y] * confusion_sim[y, conforme_idx] for y in range(n) if y != conforme_idx)
    faux_negatifs = priors[conforme_idx] * (1 - confusion_sim[conforme_idx, conforme_idx])

    precision = vrai_positifs / (vrai_positifs + faux_positifs) if (vrai_positifs + faux_positifs) > 0 else float("nan")
    rappel = vrai_positifs / (vrai_positifs + faux_negatifs) if (vrai_positifs + faux_negatifs) > 0 else float("nan")

    return precision, rappel, confusion_sim


def main():
    tracking_uri = input("Tracking URI MLflow (ex: sqlite:///mlflow.db) : ").strip()
    if not tracking_uri.startswith("sqlite:") and not tracking_uri.startswith("file:"):
        tracking_uri = f"sqlite:///{tracking_uri}"
    mlflow.set_tracking_uri(tracking_uri)

    run_id_t = input("run_id du modèle caméra T : ").strip()
    run_id_b = input("run_id du modèle caméra B : ").strip()
    img_dir = input("Dossier des images (img_dataset_exam) : ").strip()
    labels_csv = input("Chemin de labels_principal.csv : ").strip()

    device = get_device()
    class_names = VALID_LABELS

    print("Chargement des modèles...")
    model_t = mlflow.pytorch.load_model(f"runs:/{run_id_t}/model").to(device)
    model_b = mlflow.pytorch.load_model(f"runs:/{run_id_b}/model").to(device)

    entries = build_index(img_dir, labels_csv)

    entries_t = [e for e in entries if e["position"] == "T"]
    entries_b = [e for e in entries if e["position"] == "B"]
    _, _, test_t = split_stratified(entries_t)
    _, _, test_b = split_stratified(entries_b)

    print("Calcul des matrices de confusion individuelles (jeux de test réels, argmax standard)...")
    probs_t, y_true_t = get_confusion_probabilities(model_t, test_t, class_names, device)
    probs_b, y_true_b = get_confusion_probabilities(model_b, test_b, class_names, device)

    print("\nMatrice P(prédit=x | vrai=y) -- caméra T :")
    print("           " + "  ".join(f"{c:>13s}" for c in class_names))
    for i, cname in enumerate(class_names):
        print(f"{cname:<11s}" + "  ".join(f"{probs_t[i, j]:>13.1%}" for j in range(len(class_names))))

    print("\nMatrice P(prédit=x | vrai=y) -- caméra B :")
    print("           " + "  ".join(f"{c:>13s}" for c in class_names))
    for i, cname in enumerate(class_names):
        print(f"{cname:<11s}" + "  ".join(f"{probs_b[i, j]:>13.1%}" for j in range(len(class_names))))

    # Priors : proportions reelles des classes (calculees sur l'ensemble du dataset indexe)
    counts = {c: 0 for c in class_names}
    for e in entries:
        counts[e["label"]] += 1
    total = sum(counts.values())
    priors = [counts[c] / total for c in class_names]
    print(f"\nPriors utilisés (proportions réelles du dataset) : "
          + ", ".join(f"{c}={p:.1%}" for c, p in zip(class_names, priors)))

    precision_sim, rappel_sim, confusion_sim = simulate_fusion(probs_t, probs_b, class_names, priors)

    print(f"\n=== ESTIMATION (argmax standard, sous hypothèse d'indépendance T/B) après fusion ===")
    print(f"Précision Conforme estimée : {precision_sim:.1%}")
    print(f"Rappel Conforme estimée    : {rappel_sim:.1%}")

    # --- Balayage des seuils individuels avant fusion ---
    # La règle de fusion multiplie mécaniquement les rappels individuels
    # (unanimité requise pour Conforme) : un modèle trop conservateur sur
    # Conforme plombe le rappel global. On peut se permettre d'abaisser le
    # seuil de CHAQUE modèle (plus généreux sur Conforme) car la fusion
    # filtre déjà une partie des faux positifs supplémentaires via le "ET".
    print("\n=== Balayage des seuils (T, B) avant fusion ===")
    print(f"{'seuil_T':>8} {'seuil_B':>8} {'précision':>11} {'rappel':>9} {'cahier des charges ?':>10}")

    thresholds = np.arange(0.30, 1.00, 0.05)

    # Precalcul des matrices de confusion pour chaque seuil, une seule fois
    # par modele (pas une passe d'inference par combinaison (T,B) -- inutile
    # et couteux, l'inference ne depend que du seuil de SON PROPRE modele).
    probs_t_par_seuil = {th: get_confusion_probabilities(model_t, test_t, class_names, device, threshold=th)[0]
                          for th in thresholds}
    probs_b_par_seuil = {th: get_confusion_probabilities(model_b, test_b, class_names, device, threshold=th)[0]
                          for th in thresholds}

    resultats = []
    for th_t in thresholds:
        for th_b in thresholds:
            p, r, _ = simulate_fusion(probs_t_par_seuil[th_t], probs_b_par_seuil[th_b], class_names, priors)
            resultats.append((th_t, th_b, p, r))

    # Tri : d'abord les combinaisons qui satisfont les deux criteres, par rappel decroissant
    conformes = [r for r in resultats if r[2] >= 0.95 and r[3] >= 0.85]
    conformes.sort(key=lambda r: -r[3])

    if conformes:
        print(f"\n{len(conformes)} combinaison(s) de seuils satisfont précision >= 95% ET rappel >= 85% :")
        for th_t, th_b, p, r in conformes[:10]:
            print(f"  seuil_T={th_t:.2f}  seuil_B={th_b:.2f}  ->  précision={p:.1%}  rappel={r:.1%}")
    else:
        print("\nAucune combinaison de seuils ne satisfait les deux critères simultanément.")
        meilleur = max(resultats, key=lambda r: min(r[2] - 0.95, r[3] - 0.85))
        print(f"Meilleur compromis trouvé : seuil_T={meilleur[0]:.2f} seuil_B={meilleur[1]:.2f} "
              f"-> précision={meilleur[2]:.1%} rappel={meilleur[3]:.1%}")

    print("\nRappel : ceci reste une SIMULATION théorique (hypothèse d'indépendance T/B), "
          "pas une mesure -- aucune paire T/B réelle n'est disponible sur ce dataset.")


if __name__ == "__main__":
    main()
