"""
CastagNet - Classification vide/non-vide/AMBIGU sur tout le dataset
=======================================================================

Parcourt un dossier contenant l'ensemble des images du dataset (~35 254
attendues), applique à chacune la classification vide/non-vide/AMBIGU déjà
établie (fraction de pixels "brun châtaigne" à l'intérieur du cercle vert
de repérage, avec les deux seuils calibrés sur les vidéos de démo et le
vrai exemple "vide" fourni), et produit :

  1. Un CSV complet : une ligne par image, avec ses métadonnées extraites
     du nom de fichier (année, label, position caméra, numéro de caméra,
     numéro d'échantillon) et son statut vide/non-vide/AMBIGU.
  2. Un camembert de la répartition Vide / Non-vide / AMBIGU (en %).
  3. Un histogramme de la fraction de brun mesurée sur toutes les images,
     avec les deux seuils de décision matérialisés par des lignes verticales
     (utile pour juger visuellement si les seuils sont bien placés par
     rapport à la vraie distribution du dataset).

Usage
-----
    python3 classify_dataset.py
    (demande interactivement le dossier source et les chemins de sortie)

Remarque de performance : le traitement de ~35 000 images peut prendre
plusieurs minutes. Une progression est affichée toutes les 1000 images.
"""

import csv
import glob
import os
import re

import cv2
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ---------------------------------------------------------------------------
# Regex de parsing du nom de fichier (identique au reste du pipeline)
# ---------------------------------------------------------------------------
FILENAME_PATTERN = re.compile(
    r"^(?P<annee>\d{4})_(?P<label>.+?)_Cam_(?P<pos>[TB])_(?P<numcam>\d+)_(?P<numech>\d+)\.jpe?g$",
    re.IGNORECASE,
)


def parse_filename(filename):
    m = FILENAME_PATTERN.match(filename)
    if not m:
        return None
    d = m.groupdict()
    d["numcam"] = int(d["numcam"])
    d["numech"] = int(d["numech"])
    return d


# ---------------------------------------------------------------------------
# Détection du cercle vert et classification vide / non-vide / AMBIGU
# (identique à dataset_correspondence.py)
# ---------------------------------------------------------------------------
HSV_GREEN_LOWER = np.array([40, 80, 80])
HSV_GREEN_UPPER = np.array([85, 255, 255])
HSV_BROWN_LOWER = np.array([5, 30, 60])
HSV_BROWN_UPPER = np.array([30, 200, 230])
INTERIOR_MARGIN_RATIO = 0.85

PRESENCE_THRESHOLD_LOW = 0.02
PRESENCE_THRESHOLD_HIGH = 0.06


def detect_green_circle(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_GREEN_LOWER, HSV_GREEN_UPPER)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    pts = np.column_stack([xs, ys]).astype(np.float32)
    (cx, cy), r = cv2.minEnclosingCircle(pts)
    return cx, cy, r


def classify_vide(img):
    """Retourne (statut, fraction) où statut in {'True','False','AMBIGU','ERREUR_DETECTION'}."""
    res = detect_green_circle(img)
    if res is None:
        return "ERREUR_DETECTION", 0.0

    cx, cy, r = res
    r_in = r * INTERIOR_MARGIN_RATIO
    h, w = img.shape[:2]
    Y, X = np.ogrid[:h, :w]
    circle_mask = (X - cx) ** 2 + (Y - cy) ** 2 <= r_in ** 2

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    brown_mask = cv2.inRange(hsv, HSV_BROWN_LOWER, HSV_BROWN_UPPER).astype(bool)
    fraction = (brown_mask & circle_mask).sum() / circle_mask.sum()

    if fraction < PRESENCE_THRESHOLD_LOW:
        return "True", fraction
    if fraction > PRESENCE_THRESHOLD_HIGH:
        return "False", fraction
    return "AMBIGU", fraction


# ---------------------------------------------------------------------------
# Programme principal
# ---------------------------------------------------------------------------

def main():
    src_dir = input("Dossier contenant l'ensemble des images du dataset : ").strip()
    csv_out = input("Chemin du CSV de sortie (ex: classification_dataset.csv) : ").strip()
    graph_dir = input("Dossier de sortie pour les graphiques (camembert + histogramme) : ").strip()
    os.makedirs(graph_dir, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(src_dir, "*.jpg")) + glob.glob(os.path.join(src_dir, "*.jpeg")))
    n_total = len(paths)
    print(f"{n_total} image(s) trouvée(s) dans {src_dir}")
    if n_total == 0:
        print("Aucune image trouvée, arrêt.")
        return

    n_sans_nom = 0
    n_erreur_lecture = 0
    counts = {"True": 0, "False": 0, "AMBIGU": 0, "ERREUR_DETECTION": 0}
    fractions_by_status = {"True": [], "False": [], "AMBIGU": []}

    with open(csv_out, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["nom_img", "annee", "label", "position", "numcam", "numech", "vide", "fraction_brun"])

        for i, path in enumerate(paths):
            filename = os.path.basename(path)
            info = parse_filename(filename)
            if info is None:
                n_sans_nom += 1
                continue

            img = cv2.imread(path)
            if img is None:
                n_erreur_lecture += 1
                continue

            statut, fraction = classify_vide(img)
            counts[statut] += 1
            if statut in fractions_by_status:
                fractions_by_status[statut].append(fraction)

            writer.writerow([
                filename, info["annee"], info["label"], info["pos"],
                info["numcam"], info["numech"], statut, f"{fraction:.4f}",
            ])

            if (i + 1) % 1000 == 0:
                print(f"  ... {i + 1}/{n_total} images traitées")

    print(f"\nTerminé. CSV -> {csv_out}")
    if n_sans_nom:
        print(f"[ATTENTION] {n_sans_nom} fichier(s) ignoré(s) (nom non conforme).")
    if n_erreur_lecture:
        print(f"[ATTENTION] {n_erreur_lecture} fichier(s) illisible(s).")

    n_classees = sum(counts.values())
    print(f"\n--- Répartition sur {n_classees} image(s) classée(s) ---")
    for statut in ["True", "False", "AMBIGU", "ERREUR_DETECTION"]:
        n = counts[statut]
        pct = 100 * n / n_classees if n_classees else 0
        print(f"  {statut:18s} : {n:6d}  ({pct:5.1f}%)")

    if not HAS_MATPLOTLIB:
        print("\nmatplotlib non installé : graphiques non générés "
              "(pip install matplotlib --break-system-packages).")
        return

    # --- Camembert Vide / Non-vide / AMBIGU ---
    labels_camembert = []
    valeurs_camembert = []
    for statut, nom in [("True", "Vide"), ("False", "Non-vide"), ("AMBIGU", "AMBIGU")]:
        if counts[statut] > 0:
            labels_camembert.append(f"{nom}\n{counts[statut]} ({100*counts[statut]/n_classees:.1f}%)")
            valeurs_camembert.append(counts[statut])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(valeurs_camembert, labels=labels_camembert, autopct="%1.1f%%",
           startangle=90, colors=["#4C72B0", "#DD8452", "#55A868"])
    ax.set_title(f"Répartition Vide / Non-vide / AMBIGU (n={n_classees})")
    camembert_path = os.path.join(graph_dir, "camembert_vide_nonvide_ambigu.png")
    fig.savefig(camembert_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nCamembert -> {camembert_path}")

    # --- Histogramme de la fraction de brun ---
    fig, ax = plt.subplots(figsize=(10, 5))
    all_fractions = fractions_by_status["True"] + fractions_by_status["False"] + fractions_by_status["AMBIGU"]
    ax.hist(all_fractions, bins=60, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.axvline(PRESENCE_THRESHOLD_LOW, color="green", linestyle="--",
               label=f"seuil bas (vide) = {PRESENCE_THRESHOLD_LOW}")
    ax.axvline(PRESENCE_THRESHOLD_HIGH, color="red", linestyle="--",
               label=f"seuil haut (non-vide) = {PRESENCE_THRESHOLD_HIGH}")
    ax.set_xlabel("Fraction de pixels bruns dans le cercle")
    ax.set_ylabel("Nombre d'images")
    ax.set_title(f"Distribution de la fraction de brun (n={len(all_fractions)})")
    ax.legend()
    hist_path = os.path.join(graph_dir, "histogramme_fraction_brun.png")
    fig.savefig(hist_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Histogramme -> {hist_path}")

    # Statistiques complémentaires utiles pour le choix d'architecture
    if all_fractions:
        arr = np.array(all_fractions)
        print(f"\nStatistiques fraction_brun (toutes images confondues) :")
        print(f"  moyenne={arr.mean():.4f}  médiane={np.median(arr):.4f}  "
              f"min={arr.min():.4f}  max={arr.max():.4f}  écart-type={arr.std():.4f}")


if __name__ == "__main__":
    main()
