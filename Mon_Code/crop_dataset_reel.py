"""
CastagNet - Crop tangent au cercle des images non-vides (dataset réel)
=========================================================================

Les images du dataset réel sont déjà cadrées autour du hublot, matérialisé
par un cercle vert complet : tout ce qui est en dehors du cercle est du
noir sans information. Il suffit donc de découper le carré tangent au
cercle sur ses 4 côtés (2r x 2r, centré sur le cercle) pour capturer
l'intégralité du contenu utile, sans marge inutile et sans distinction
T/B nécessaire (le carré englobe naturellement le haut ET le bas du
cercle, quelle que soit la caméra).

Seules les images classées NON-VIDE ou AMBIGU (statuts 'False' et 'AMBIGU'
dans le CSV produit par classify_dataset.py, typiquement nommé
metrics_dataset.csv) sont traitées. Décision prise après analyse contre la
vérité terrain (labels_principal.csv) : 97,2 % des images AMBIGU sont en
réalité de vraies châtaignes (fragments limites pour le détecteur colorimétrique
de vide, mais pas nécessairement problématiques pour un modèle de
classification) -- elles sont donc envoyées au classifieur plutôt qu'écartées.
Seul le vide franc (statut 'True') est exclu, destiné à un lot "repasse"
séparé. Les crops sont enregistrés dans un dossier fixe : img_dataset_exam/

Usage
-----
    python3 crop_dataset_reel.py
    (demande interactivement : dossier des images, chemin du CSV de
     classification -- ex. metrics_dataset.csv produit par classify_dataset.py)
"""

import csv
import os

import cv2
import numpy as np

HSV_GREEN_LOWER = np.array([40, 80, 80])
HSV_GREEN_UPPER = np.array([85, 255, 255])

OUTPUT_DIR = "img_dataset_exam"


def detect_green_circle(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_GREEN_LOWER, HSV_GREEN_UPPER)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    pts = np.column_stack([xs, ys]).astype(np.float32)
    (cx, cy), r = cv2.minEnclosingCircle(pts)
    return cx, cy, r


def crop_tangent_cercle(img, cx, cy, r):
    """Découpe le carré tangent au cercle de chaque côté (2r x 2r, centré
    sur le cercle). Taille de sortie variable d'une image à l'autre (2r) --
    sans incidence sur le calcul de fraction de brun (déjà normalisé par
    la surface du cercle) ni sur un éventuel redimensionnement final avant
    entraînement du modèle."""
    h, w = img.shape[:2]
    size = int(round(2 * r))
    x0 = int(round(cx - r))
    y0 = int(round(cy - r))
    x1 = x0 + size
    y1 = y0 + size

    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - w)
    pad_bottom = max(0, y1 - h)

    if pad_left or pad_top or pad_right or pad_bottom:
        img = cv2.copyMakeBorder(
            img, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(0, 0, 0),
        )
        x0 += pad_left
        y0 += pad_top
        x1 += pad_left
        y1 += pad_top

    return img[y0:y1, x0:x1]


def main():
    images_dir = input("Dossier contenant les images du dataset : ").strip()
    classif_csv = input(
        "Chemin du CSV de classification (ex: metrics_dataset.csv, "
        "produit par classify_dataset.py) : "
    ).strip()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    n_traite = 0
    n_echec_detection = 0
    n_fichier_manquant = 0

    with open(classif_csv, newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        # Décision (cf. brainstorm §4.2) : les images AMBIGU sont envoyées au
        # classifieur au même titre que les non-vides -- l'analyse contre la
        # vérité terrain a montré que 97,2 % des AMBIGU sont en réalité de
        # vraies châtaignes (fragments limites pour le détecteur de vide,
        # mais pas nécessairement pour le modèle de classification). Seul
        # le "vide franc" (statut 'True') est écarté avant prédiction.
        rows = [r for r in reader if r.get("vide") in ("False", "AMBIGU")]

    n_non_vide = sum(1 for r in rows if r["vide"] == "False")
    n_ambigu = sum(1 for r in rows if r["vide"] == "AMBIGU")
    print(f"{len(rows)} image(s) à traiter : {n_non_vide} non-vide(s) + {n_ambigu} AMBIGU "
          f"(le vide franc est exclu -> lot \"repasse\").")

    for i, row in enumerate(rows):
        filename = row["nom_img"]
        img_path = os.path.join(images_dir, filename)

        img = cv2.imread(img_path)
        if img is None:
            n_fichier_manquant += 1
            continue

        circle = detect_green_circle(img)
        if circle is None:
            n_echec_detection += 1
            continue

        cx, cy, r = circle
        crop = crop_tangent_cercle(img, cx, cy, r)

        out_path = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(out_path, crop)
        n_traite += 1

        if (i + 1) % 1000 == 0:
            print(f"  ... {i + 1}/{len(rows)} images traitées")

    print(f"\nTerminé. {n_traite} crop(s) écrit(s) -> {OUTPUT_DIR}/")
    if n_echec_detection:
        print(f"[ATTENTION] {n_echec_detection} échec(s) de détection du cercle vert.")
    if n_fichier_manquant:
        print(f"[ATTENTION] {n_fichier_manquant} fichier(s) introuvable(s) dans {images_dir}.")


if __name__ == "__main__":
    main()
