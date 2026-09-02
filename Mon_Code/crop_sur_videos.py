"""
CastagNet - Crop fixe 350x350 asymétrique (pastille verte comme seule référence)
===================================================================================

Règle de crop :
  - Carré de taille fixe 350x350, centré horizontalement sur le milieu de
    l'image (pas sur la pastille).
  - Caméra T : le bord HAUT du carré est à 100 px au-dessus de la ligne
    horizontale passant par la pastille -> le carré descend donc jusqu'à
    250 px en dessous de cette ligne (marge basse généreuse, pour la partie
    de châtaigne qui pend sous le hublot vue depuis le dessus).
  - Caméra B : le bord BAS du carré est à 100 px en dessous de la ligne de
    la pastille -> le carré remonte donc jusqu'à 250 px au-dessus de cette
    ligne (inverse de T).

La caméra (T ou B) n'est plus un argument : elle est lue directement dans le
nom de chaque fichier, d'après la convention du projet
    annee_label_Cam_{T|B}_numCamera_numEchantillon.jpg
Exemple : 2025_Conforme_1_Cam_B_1_1.jpg -> caméra B.
Chaque image du dossier peut donc être T ou B, traitée individuellement.

Un filtre achromatique est ensuite appliqué (pixels blanc/gris/noir mis au
noir pur, seuil de saturation HSV réglable, demandé à l'exécution).

Si la zone de crop déborde de l'image, elle est complétée par du noir
(taille de sortie toujours strictement 350x350).

Usage
-----
    python3 crop_350.py
    (le programme demande interactivement le dossier source et le seuil)

Le dossier de résultats est créé automatiquement : <dossier_source>_crop
"""

import glob
import os
import re

import cv2
import numpy as np

HSV_GREEN_LOWER = np.array([40, 80, 80])
HSV_GREEN_UPPER = np.array([85, 255, 255])
MIN_DOT_AREA = 30

CROP_SIZE = 350
OFFSET_T_TOP = 100     # T : bord haut = pastille_y - 100
OFFSET_B_BOTTOM = 100  # B : bord bas  = pastille_y + 100

# Convention de nommage du projet : annee_label_Cam_{T|B}_numCamera_numEchantillon.jpg
# Exemple : 2025_Conforme_1_Cam_B_1_1.jpg
CAMERA_PATTERN = re.compile(r"_Cam_([TB])_", re.IGNORECASE)


def extract_camera_from_filename(filename):
    """Extrait la caméra (T ou B) depuis le nom de fichier, d'après le motif
    '_Cam_T_' ou '_Cam_B_'. Retourne None si le motif n'est pas trouvé."""
    match = CAMERA_PATTERN.search(filename)
    if match:
        return match.group(1).upper()
    return None


def detect_green_dot(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_GREEN_LOWER, HSV_GREEN_UPPER)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in contours:
        area = cv2.contourArea(c)
        if area > MIN_DOT_AREA and (best is None or area > best[0]):
            x, y, w, h = cv2.boundingRect(c)
            best = (area, x + w / 2, y + h / 2)
    if best is None:
        return None
    return best[1], best[2]


def crop_350(img, camera, dot_x, dot_y):
    """Calcule et découpe le carré 350x350 selon la règle T/B décrite plus
    haut. Centré horizontalement sur le MILIEU DE L'IMAGE (pas sur la
    pastille) ; seule la position verticale dépend de la pastille.
    Complète par du noir si besoin pour garantir la taille exacte."""
    h, w = img.shape[:2]
    img_center_x = w / 2

    x0 = int(round(img_center_x - CROP_SIZE / 2))
    x1 = x0 + CROP_SIZE

    if camera.upper() == "T":
        y0 = int(round(dot_y - OFFSET_T_TOP))
    else:  # B
        y1_base = int(round(dot_y + OFFSET_B_BOTTOM))
        y0 = y1_base - CROP_SIZE
    y1 = y0 + CROP_SIZE

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

    crop = img[y0:y1, x0:x1]
    if crop.shape[:2] != (CROP_SIZE, CROP_SIZE):
        crop = cv2.resize(crop, (CROP_SIZE, CROP_SIZE))
    return crop


def remove_achromatic(img, sat_threshold):
    """Remplace par du noir pur tout pixel dont la couleur s'approche du
    blanc, du gris ou du noir (faible saturation HSV, peu importe la
    luminosité). Plus sat_threshold est élevé, plus le filtre est
    restrictif (plus de pixels sont considérés comme achromatiques et
    donc mis au noir)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    mask_achromatic = saturation < sat_threshold
    out = img.copy()
    out[mask_achromatic] = (0, 0, 0)
    return out


def process_image(path, sat_threshold, out_path=None):
    camera = extract_camera_from_filename(os.path.basename(path))
    if camera is None:
        return None, "caméra non identifiable dans le nom de fichier (motif '_Cam_T_' ou '_Cam_B_' absent)"

    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)

    dot = detect_green_dot(img)
    if dot is None:
        return None, "pastille non détectée"

    dot_x, dot_y = dot
    crop = crop_350(img, camera, dot_x, dot_y)
    crop = remove_achromatic(crop, sat_threshold)

    if out_path:
        cv2.imwrite(out_path, crop)

    return crop, (camera, dot_x, dot_y)


def main():
    src_dir = input("Chemin du dossier contenant les images à cropper : ").strip()

    seuil_str = input(
        "Seuil de saturation pour le filtre achromatique "
        "(plus la valeur est élevée, plus le filtre est restrictif, "
        "c'est-à-dire qu'il élimine davantage de pixels) : "
    ).strip()
    sat_threshold = int(seuil_str)

    src_dir_norm = os.path.normpath(src_dir)
    outdir = src_dir_norm + "_crop"
    os.makedirs(outdir, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(src_dir, "*.jpg")))
    n_echec = 0
    for path in paths:
        out_path = os.path.join(outdir, os.path.basename(path))
        crop, info = process_image(path, sat_threshold, out_path)
        if crop is None:
            n_echec += 1
            print(f"  [ATTENTION] échec : {path} ({info})")
    print(f"{len(paths)} image(s) traitée(s) -> {outdir}  ({n_echec} échec(s))")


if __name__ == "__main__":
    main()
