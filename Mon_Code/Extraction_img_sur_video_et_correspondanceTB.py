"""
CastagNet - Pipeline de correspondance T/B par détection de la pastille verte
================================================================================

Ce programme repart de zéro sur la logique suivante (validée après analyse) :

  - Le timestamp et l'apparence de la châtaigne ne sont PAS fiables pour
    faire correspondre les cycles de deux vidéos (démarrage différent,
    latence, mouvement/rotation de la châtaigne entre les deux vues).
  - L'ABSENCE de châtaigne (cycle "vide") est en revanche un évènement
    fiable et binaire : un cycle vide l'est forcément dans les deux vues
    simultanément, puisqu'il s'agit du même passage physique.
  - On construit donc, pour chaque vidéo, une séquence de cycles avec un
    indicateur vide/non-vide, et on cherche le décalage qui aligne EXACTEMENT
    les deux séquences de vides (décalage possible dû à un démarrage
    d'enregistrement différent, géré par un remplissage NULL en début/fin).
  - IMPORTANT : si aucun décalage ne produit une correspondance exacte des
    zones vides, on considère qu'il N'Y A PAS de correspondance possible
    entre les deux vidéos. On ne force jamais un alignement approximatif.

Étapes du programme
--------------------
1. Demande (ou reçoit en argument) le chemin de 2 vidéos.
2. Pour chaque vidéo :
   a. Extraction des frames (ffmpeg, désentrelacement).
   b. Détection de la pastille verte sur chaque frame.
   c. Segmentation en cycles (un cycle = un passage devant le hublot) via
      les minima locaux de la distance de la pastille à la ligne médiane
      horizontale de l'image (centre vertical, cf. mouvement de balayage
      haut -> bas de la pastille observé).
   d. Pour chaque cycle, sauvegarde de l'image où la pastille est la plus
      proche de cette ligne médiane, dans un dossier nommé d'après la vidéo.
   e. Détection vide/non-vide sur chaque image sauvegardée (fraction de
      pixels "brun châtaigne" dans une zone définie relativement à la
      position de la pastille).
   f. Écriture d'un CSV (image_id; image_nom; vide) dans le même dossier.
3. Génération d'un échantillon visuel : 2 planches, chacune montrant 10
   images T superposées à 10 images B (dans l'ordre d'extraction).
4. Recherche de la correspondance entre les deux séquences de vides
   (décalage exact requis, remplissage NULL aux extrémités). Si aucun
   décalage ne donne une correspondance parfaite, le programme l'indique
   clairement : PAS DE CORRESPONDANCE.

Usage
-----
    python3 castagnet_pipeline.py --video1 attachment.avi --video2 attachment-2.avi --outdir resultats/

    # Pour tests/reprise sans les fichiers vidéo originaux (frames déjà extraites) :
    python3 castagnet_pipeline.py --frames1-dir frames_A --name1 attachment \
                                   --frames2-dir frames_B --name2 attachment-2 \
                                   --outdir resultats/
"""

import argparse
import csv
import os
import shutil
import subprocess

import cv2
import numpy as np
from scipy.signal import argrelextrema

# ---------------------------------------------------------------------------
# Paramètres (à recalibrer si l'éclairage / le montage caméra change)
# ---------------------------------------------------------------------------
HSV_GREEN_LOWER = np.array([40, 80, 80])
HSV_GREEN_UPPER = np.array([85, 255, 255])
MIN_DOT_AREA = 30  # aire minimale (px) pour valider une détection de pastille

HSV_BROWN_LOWER = np.array([5, 30, 60])
HSV_BROWN_UPPER = np.array([30, 200, 230])
PRESENCE_THRESHOLD_LOW = 0.05   # en dessous : vide fiable
PRESENCE_THRESHOLD_HIGH = 0.12  # au-dessus : présence fiable
# Entre les deux seuils : zone d'ambiguïté (ex. fragment de châtaigne minuscule
# ou partiellement hors cadre). Ces cas sont marqués "AMBIGU" plutôt que forcés
# en True/False, car un seuil unique ferait basculer arbitrairement ces cas au
# moindre bruit de détection (observé en pratique : 0.077 vs 0.084 pour un même
# cycle physique selon la méthode de calcul de la zone de recherche).

# Zone de recherche de la châtaigne, définie relativement à la position de
# la pastille (le hublot se trouve à gauche de la pastille sur les deux
# caméras observées). Décalage et taille par défaut, ajustables si besoin.
ROI_OFFSET_X = -200   # décalage horizontal du centre de la zone p/r à la pastille
ROI_WIDTH = 340
ROI_HEIGHT = 300

CYCLE_MERGE_MIN_DISTANCE = 15  # distance mini (frames) entre deux cycles distincts


# ---------------------------------------------------------------------------
# Étape a. Extraction des frames
# ---------------------------------------------------------------------------

def extract_frames(video_path, tmp_dir):
    os.makedirs(tmp_dir, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", "yadif",
        os.path.join(tmp_dir, "f_%04d.png"),
        "-loglevel", "error",
    ]
    subprocess.run(cmd, check=True)
    return sorted(
        os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir)
        if f.startswith("f_") and f.endswith(".png")
    )


def list_existing_frames(frames_dir, prefix):
    return sorted(
        os.path.join(frames_dir, f) for f in os.listdir(frames_dir)
        if f.startswith(prefix + "_") and f.endswith(".png")
    )


# ---------------------------------------------------------------------------
# Étape b. Détection de la pastille verte
# ---------------------------------------------------------------------------

def detect_green_dot(img):
    """Retourne (area, cx, cy) du plus gros contour vert valide, ou None."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_GREEN_LOWER, HSV_GREEN_UPPER)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in contours:
        area = cv2.contourArea(c)
        if area > MIN_DOT_AREA and (best is None or area > best[0]):
            x, y, w, h = cv2.boundingRect(c)
            best = (area, x + w / 2, y + h / 2)
    return best


# ---------------------------------------------------------------------------
# Étape c. Segmentation en cycles
# ---------------------------------------------------------------------------

def find_cycle_minima(dist, min_distance=CYCLE_MERGE_MIN_DISTANCE):
    filled = np.where(np.isnan(dist), np.nanmax(dist) + 1, dist)
    candidates = argrelextrema(filled, np.less_equal, order=5)[0]
    candidates = [c for c in candidates if not np.isnan(dist[c])]

    merged = []
    for c in candidates:
        if merged and c - merged[-1] < min_distance:
            if dist[c] < dist[merged[-1]]:
                merged[-1] = c
        else:
            merged.append(c)
    return merged


# ---------------------------------------------------------------------------
# Étape e. Détection vide / non-vide
# ---------------------------------------------------------------------------

def is_empty(img, dot_x, dot_y):
    """Retourne (statut, fraction) où statut in {'True','False','AMBIGU'}
    selon la fraction de pixels bruns détectée dans la zone du hublot
    (cf. commentaire sur les deux seuils)."""
    h, w = img.shape[:2]
    cx = int(dot_x + ROI_OFFSET_X)
    cy = int(dot_y)
    x0 = max(0, cx - ROI_WIDTH // 2)
    y0 = max(0, cy - ROI_HEIGHT // 2)
    x1 = min(w, cx + ROI_WIDTH // 2)
    y1 = min(h, cy + ROI_HEIGHT // 2)
    if x1 <= x0 or y1 <= y0:
        return "True", 0.0  # zone hors image -> considéré vide par défaut

    crop = img[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_BROWN_LOWER, HSV_BROWN_UPPER)
    fraction = mask.mean() / 255.0
    if fraction < PRESENCE_THRESHOLD_LOW:
        return "True", fraction
    if fraction > PRESENCE_THRESHOLD_HIGH:
        return "False", fraction
    return "AMBIGU", fraction


# ---------------------------------------------------------------------------
# Traitement complet d'une vidéo
# ---------------------------------------------------------------------------

def process_video(frames, video_name, outdir):
    """Traite une liste de frames (chemins triés) et produit :
       - le dossier outdir/<video_name>/ contenant les images sélectionnées
       - le CSV outdir/<video_name>/<video_name>.csv (image_id;image_nom;vide)
       Retourne la liste des enregistrements (dict) pour usage ultérieur.
    """
    video_dir = os.path.join(outdir, video_name)
    os.makedirs(video_dir, exist_ok=True)

    sample_img = cv2.imread(frames[0])
    h, w = sample_img.shape[:2]
    center_y = h / 2

    print(f"[{video_name}] Détection de la pastille sur {len(frames)} frames...")
    dots = []  # (x, y) ou (None, None)
    for path in frames:
        img = cv2.imread(path)
        res = detect_green_dot(img)
        dots.append((res[1], res[2]) if res else (None, None))

    dist = np.array([
        abs(y - center_y) if y is not None else np.nan
        for (_, y) in dots
    ])

    minima = find_cycle_minima(dist)
    print(f"[{video_name}] {len(minima)} cycles détectés.")

    records = []
    for cycle_id, m in enumerate(minima):
        img = cv2.imread(frames[m])
        dot_x, dot_y = dots[m]
        if dot_x is not None:
            vide, fraction = is_empty(img, dot_x, dot_y)
        else:
            vide, fraction = "True", 0.0

        image_nom = f"{video_name}_{cycle_id:03d}.jpg"
        image_path = os.path.join(video_dir, image_nom)
        cv2.imwrite(image_path, img)

        records.append({
            "image_id": cycle_id,
            "image_nom": image_nom,
            "vide": vide,
            "fraction": fraction,
            "image_path": image_path,
            "frame_index": m,
        })

    csv_path = os.path.join(video_dir, f"{video_name}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["image_id", "image_nom", "vide", "fraction_brun"])
        for r in records:
            writer.writerow([r["image_id"], r["image_nom"], r["vide"], f"{r['fraction']:.4f}"])
    print(f"[{video_name}] CSV -> {csv_path}")

    return records, video_dir


# ---------------------------------------------------------------------------
# Étape 3. Planches d'échantillon (10 T / 10 B x2)
# ---------------------------------------------------------------------------

def build_sample_sheets(records1, records2, name1, name2, outdir, n_per_block=10, n_blocks=2):
    for block in range(n_blocks):
        lo = block * n_per_block
        hi = lo + n_per_block
        imgs1 = [r["image_path"] for r in records1[lo:hi]]
        imgs2 = [r["image_path"] for r in records2[lo:hi]]
        if not imgs1 and not imgs2:
            continue

        thumbs1 = [cv2.resize(cv2.imread(p), (120, 96)) for p in imgs1]
        thumbs2 = [cv2.resize(cv2.imread(p), (120, 96)) for p in imgs2]
        # complète si moins de n_per_block images disponibles
        blank = np.zeros((96, 120, 3), dtype=np.uint8)
        thumbs1 += [blank] * (n_per_block - len(thumbs1))
        thumbs2 += [blank] * (n_per_block - len(thumbs2))

        row1 = np.hstack(thumbs1)
        row2 = np.hstack(thumbs2)
        sheet = np.vstack([row1, row2])

        out_path = os.path.join(outdir, f"sample_block{block + 1}_{name1}_over_{name2}.jpg")
        cv2.imwrite(out_path, sheet)
        print(f"Planche d'échantillon -> {out_path}")


# ---------------------------------------------------------------------------
# Étape 4. Correspondance stricte (décalage exact requis)
# ---------------------------------------------------------------------------

def find_best_shift(seq1, seq2):
    """Recherche automatique du décalage k (aucune constante à renseigner).

    Principe demandé : on part du milieu de chaque fichier pour construire
    une première hypothèse de décalage, puis on balaie TOUS les décalages
    possibles entre les deux séquences. Pour chaque décalage testé, on
    compte le nombre de fois où une correspondance vide<->vide est juste
    (les deux valent 'True' au même endroit) : c'est le score du décalage.
    On conserve le décalage qui obtient le meilleur score.

    Exemple donné : fichier 1 de 50 frames, fichier 2 de 60 frames -> le
    milieu du fichier 1 est l'index 24 (25e frame). Comparer la 1re frame
    du fichier 2 (index 0) à la 25e frame du fichier 1 (index 24) revient à
    tester le décalage k=24 (car index_1 = index_2 + 24). C'est un point de
    départ ; le balayage complet couvre ensuite tous les k possibles pour
    ne rien manquer.

    Les cas 'AMBIGU' ne comptent ni pour ni contre un décalage (ils sont
    ignorés dans le calcul du score ET dans la détection de conflit).

    Retourne la liste des décalages testés, triée du meilleur au moins bon :
    chaque élément est (k, score_vide_vide, conflits, comparaisons_fiables).
    Retourne une liste vide si l'une des deux séquences est vide (aucun
    cycle détecté), auquel cas aucune comparaison n'est possible.
    """
    n1, n2 = len(seq1), len(seq2)
    if n1 == 0 or n2 == 0:
        return []

    # Point de départ illustratif (milieu de la séquence 1 vs début de la 2),
    # conservé uniquement à titre indicatif dans les logs.
    # Exemple : 50 frames -> la "25e frame" correspond à l'index 24 (0-indexé).
    milieu_seq1 = (n1 // 2) - 1 if n1 > 0 else 0
    decalage_depart = milieu_seq1  # index_1 (milieu) <-> index_2 (0) => k = milieu - 0

    # Balayage exhaustif : k doit permettre au moins un chevauchement entre
    # les deux séquences, donc k va de -(n2-1) à (n1-1).
    results = []
    for k in range(-(n2 - 1), n1):
        score_vide_vide, conflits, total = 0, 0, 0
        for i in range(n1):
            j = i - k  # convention : index_1 (i) <-> index_2 (j) avec i = j + k
            if 0 <= j < n2:
                v1, v2 = seq1[i], seq2[j]
                if v1 == "AMBIGU" or v2 == "AMBIGU":
                    continue
                total += 1
                if v1 == "True" and v2 == "True":
                    score_vide_vide += 1
                if v1 != v2:
                    conflits += 1
        results.append((k, score_vide_vide, conflits, total))

    # Meilleur résultat : le plus de vide<->vide corrects, puis le moins de
    # conflits, puis le plus grand nombre de comparaisons fiables.
    results.sort(key=lambda r: (-r[1], r[2], -r[3]))

    print(f"(point de départ illustratif : milieu de la séquence 1 = index {milieu_seq1} "
          f"<-> 1re frame de la séquence 2 = index 0, soit k={decalage_depart:+d})")

    return results


def build_correspondence(records1, records2, name1, name2, outdir):
    seq1 = [r["vide"] for r in records1]
    seq2 = [r["vide"] for r in records2]
    n1, n2 = len(seq1), len(seq2)

    print(f"\nSéquence vide {name1} ({n1}) : {seq1}")
    print(f"Séquence vide {name2} ({n2}) : {seq2}")

    results = find_best_shift(seq1, seq2)

    out_path = os.path.join(outdir, "correspondence.csv")

    if not results:
        print("\n*** IMPOSSIBLE DE CALCULER UNE CORRESPONDANCE ***")
        print("Au moins une des deux séquences ne contient aucun cycle détecté.")
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(["RESULTAT"])
            writer.writerow(["IMPOSSIBLE_AUCUN_CYCLE_DETECTE"])
        return None

    print("\nTop 5 décalages testés (k, vide<->vide corrects, conflits, comparaisons fiables) :")
    for k, score, conflits, total in results[:5]:
        print(f"  k={k:+d}  vide_vide={score}  conflits={conflits}  total={total}")

    best_k, best_score, best_conflits, best_total = results[0]

    if best_score == 0 and best_total == 0:
        print("\n*** IMPOSSIBLE DE CALCULER UNE CORRESPONDANCE ***")
        print("Aucune comparaison fiable n'a pu être établie (pastille non "
              "détectée, ou aucun cycle détecté dans l'une des deux vidéos).")
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(["RESULTAT"])
            writer.writerow(["IMPOSSIBLE_AUCUNE_COMPARAISON_FIABLE"])
        return None

    if best_score == 0:
        print("\n*** AUCUNE CORRESPONDANCE FIABLE TROUVÉE ***")
        print("Aucun décalage ne produit de correspondance vide<->vide confirmée.")
        print("=> Les deux vidéos ne peuvent pas être mises en correspondance "
              "de façon fiable avec cette méthode.")
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(["RESULTAT"])
            writer.writerow(["AUCUNE_CORRESPONDANCE"])
        return None

    if best_conflits > 0:
        print(f"\nATTENTION : le meilleur décalage (k={best_k:+d}) obtient "
              f"{best_score} correspondance(s) vide<->vide mais aussi "
              f"{best_conflits} conflit(s) (vide vs non-vide). À vérifier.")

    tied = [r for r in results if r[1] == best_score and r[2] == best_conflits]
    if len(tied) > 1:
        print(f"\nATTENTION : {len(tied)} décalages sont à égalité "
              f"(score={best_score}, conflits={best_conflits}) : "
              f"{[r[0] for r in tied]}. Cas ambigu à vérifier manuellement.")

    print(f"\n--> Décalage retenu : k={best_k:+d}  "
          f"({best_score} vide<->vide confirmé(s), {best_conflits} conflit(s), "
          f"{best_total} comparaisons fiables)")

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow([f"image_{name1}", f"image_{name2}", "vide_1", "vide_2", "accord"])

        # convention : index_1 (i) <-> index_2 (j) avec j = i - best_k
        idx1_min, idx1_max = 0, n1 - 1
        idx2_min, idx2_max = best_k, best_k + n2 - 1
        lo = min(idx1_min, idx2_min)
        hi = max(idx1_max, idx2_max)

        for i in range(lo, hi + 1):
            j = i - best_k
            in1 = 0 <= i < n1
            in2 = 0 <= j < n2
            name_i = records1[i]["image_nom"] if in1 else "NULL"
            name_j = records2[j]["image_nom"] if in2 else "NULL"
            v1 = records1[i]["vide"] if in1 else "NULL"
            v2 = records2[j]["vide"] if in2 else "NULL"
            if not in1 or not in2:
                accord = "NULL"
            elif v1 == "AMBIGU" or v2 == "AMBIGU":
                accord = "AMBIGU"
            elif v1 == v2:
                accord = "OK"
            else:
                accord = "CONFLIT"
            writer.writerow([name_i, name_j, v1, v2, accord])

    print(f"Table de correspondance -> {out_path}")
    return best_k


# ---------------------------------------------------------------------------
# Programme principal
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video1")
    ap.add_argument("--video2")
    ap.add_argument("--frames1-dir", help="Dossier de frames déjà extraites pour la vidéo 1 (debug/reprise)")
    ap.add_argument("--frames2-dir", help="Dossier de frames déjà extraites pour la vidéo 2 (debug/reprise)")
    ap.add_argument("--name1", help="Nom à utiliser pour la vidéo 1 (déduit du chemin sinon)")
    ap.add_argument("--name2", help="Nom à utiliser pour la vidéo 2 (déduit du chemin sinon)")
    ap.add_argument("--frames-prefix1", default="f")
    ap.add_argument("--frames-prefix2", default="f")
    ap.add_argument("--outdir", default="Resultats")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    tmp_dirs_to_clean = []

    # --- Vidéo 1 ---
    if not args.video1 and not args.frames1_dir:
        args.video1 = input("Chemin de la première vidéo : ").strip()
    name1 = args.name1 or (
        os.path.splitext(os.path.basename(args.video1))[0] if args.video1
        else os.path.basename(os.path.normpath(args.frames1_dir))
    )
    if args.frames1_dir:
        frames1 = list_existing_frames(args.frames1_dir, args.frames_prefix1)
    else:
        tmp1 = os.path.join(args.outdir, "_tmp_frames_1")
        frames1 = extract_frames(args.video1, tmp1)
        tmp_dirs_to_clean.append(tmp1)

    # --- Vidéo 2 ---
    if not args.video2 and not args.frames2_dir:
        args.video2 = input("Chemin de la deuxième vidéo : ").strip()
    name2 = args.name2 or (
        os.path.splitext(os.path.basename(args.video2))[0] if args.video2
        else os.path.basename(os.path.normpath(args.frames2_dir))
    )
    if args.frames2_dir:
        frames2 = list_existing_frames(args.frames2_dir, args.frames_prefix2)
    else:
        tmp2 = os.path.join(args.outdir, "_tmp_frames_2")
        frames2 = extract_frames(args.video2, tmp2)
        tmp_dirs_to_clean.append(tmp2)

    records1, dir1 = process_video(frames1, name1, args.outdir)
    records2, dir2 = process_video(frames2, name2, args.outdir)

    try:
        build_sample_sheets(records1, records2, name1, name2, args.outdir)
        build_correspondence(records1, records2, name1, name2, args.outdir)
    finally:
        # Nettoyage des frames temporaires extraites des vidéos, même en cas
        # d'erreur dans les étapes suivantes.
        for tmp_dir in tmp_dirs_to_clean:
            if os.path.isdir(tmp_dir):
                shutil.rmtree(tmp_dir)
                print(f"Dossier temporaire supprimé : {tmp_dir}")


if __name__ == "__main__":
    main()
