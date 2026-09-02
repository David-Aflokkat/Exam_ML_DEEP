"""
CastagNet - Dataset réel pré-cropé : CSV vide/non-vide + correspondance T/B
===============================================================================

Contexte : contrairement aux vidéos brutes traitées précédemment, les images
du dataset réel sont fournies déjà cropées (fond noir, cercle vert marquant
le hublot, châtaigne visible ou non à l'intérieur). Convention de nommage :

    annee_label_Cam_{T|B}_{numCamera}_{numEchantillon}.jpg
    ex: 2025_Conforme_2_Cam_B_3_338.jpg
        -> année=2025, label=Conforme_2, position=B, numCamera=3, échantillon=338

Le label peut contenir un suffixe (ex. 'Conforme_2') ou non (ex. 'PIETRA') :
le nom est extrait comme tout ce qui se trouve entre l'année et '_Cam_'.

Étapes
------
1. Parcourt le dossier, extrait (année, label, position, numCamera, numEchantillon)
   pour chaque image via une regex.
2. Détecte le cercle vert (contour complet, pas juste une pastille ici) par
   cercle englobant minimal sur les pixels verts détectés.
3. Calcule la fraction de pixels "brun châtaigne" à l'intérieur du cercle
   (zone légèrement réduite pour exclure l'anneau vert lui-même) ->
   classification vide / non-vide / AMBIGU (mêmes deux seuils que pour les
   vidéos, à recalibrer si nécessaire une fois de vrais exemples "vide"
   disponibles : aucun n'était présent dans l'échantillon utilisé ici).
4. Regroupe les images par (numCamera, position), triées par numEchantillon
   -> constitue un "cycle" par caméra, un CSV par cycle
   (id_img; nom_img; vide) dans le dossier de sortie.
5. Pour chaque numCamera présent en double (T et B), compare les deux CSV
   avec la même recherche automatique de décalage que pour les vidéos
   (nombre de vide<->vide confirmés à maximiser, conflits à minimiser),
   et produit une table de correspondance T/B par paire de caméra.

Usage
-----
    python3 dataset_correspondence.py
    (demande interactivement le dossier source et les 2 seuils)
"""

import csv
import glob
import os
import re
from collections import defaultdict

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Regex de parsing du nom de fichier
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
# ---------------------------------------------------------------------------
HSV_GREEN_LOWER = np.array([40, 80, 80])
HSV_GREEN_UPPER = np.array([85, 255, 255])

HSV_BROWN_LOWER = np.array([5, 30, 60])
HSV_BROWN_UPPER = np.array([30, 200, 230])

INTERIOR_MARGIN_RATIO = 0.85  # rayon réduit pour exclure l'anneau vert

# Seuils à deux niveaux (identiques à la démarche vidéo). À RECALIBRER une
# fois de vrais exemples "vide" disponibles dans le dataset réel.
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
    """Retourne (statut, fraction) où statut in {'True','False','AMBIGU'}."""
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
# Recherche automatique du décalage — basée sur la VALEUR de numEchantillon
# (et non sur la position dans la liste triée, qui n'a de sens que si les
# numéros d'échantillon sont parfaitement consécutifs et sans trou -- ce qui
# n'est pas garanti sur des images déjà sélectionnées/labellisées).
# ---------------------------------------------------------------------------

SHIFT_SEARCH_WINDOW = 10  # +/- fenêtre autour du décalage d'ancrage


def find_first_non_vide(records_sorted_by_numech):
    """Retourne le numEchantillon du premier enregistrement franchement
    non-vide ('False') dans la liste triée, ou None si aucun."""
    for r in records_sorted_by_numech:
        if r["vide"] == "False":
            return r["numech"]
    return None


def find_best_shift_by_numech(records_b, records_t, window=SHIFT_SEARCH_WINDOW):
    """Recherche du décalage k = numech_B - numech_T, ANCRÉE sur la première
    image non-vide de chaque liste, avec une fenêtre de recherche restreinte
    (+/- `window`) autour de ce point d'ancrage -- plutôt qu'une recherche
    exhaustive sur toutes les différences possibles.

    Pourquoi ce changement : une recherche exhaustive génère des milliers de
    décalages candidats qui, par construction, ont TOUJOURS au moins une
    paire "gagnante" -- y compris des décalages absurdes portés par une
    unique coïncidence (observé en pratique : des décalages de plusieurs
    centaines de frames retenus sur la foi d'une seule paire alignée). En
    ancrant la recherche sur la première apparition non-vide de chaque
    caméra (repère physique commun et fiable), puis en limitant l'exploration
    à une fenêtre de +/-100 échantillons autour de cette estimation, on ne
    considère que des décalages plausibles a priori.

    Critère de sélection dans la fenêtre :
      1. Zéro conflit EXIGÉ (un vide ne peut jamais faire face à un non-vide :
         filtre dur, pas une simple priorité de tri).
      2. Parmi les décalages sans conflit, on maximise l'accord total
         (vide<->vide + non-vide<->non-vide confondus, hors AMBIGU) : c'est
         le "meilleur compromis" demandé, pas seulement les vide<->vide.

    Retourne (résultats_triés, ancrage_k0, décalage_sans_conflit_trouvé).
    """
    dict_b = {r["numech"]: r["vide"] for r in records_b}
    dict_t = {r["numech"]: r["vide"] for r in records_t}
    if not dict_b or not dict_t:
        return [], None, False

    first_b = find_first_non_vide(records_b)
    first_t = find_first_non_vide(records_t)
    if first_b is None or first_t is None:
        # Repli : pas de non-vide franc dans l'une des deux listes -> on
        # ancre sur le tout premier échantillon disponible de chaque liste.
        first_b = records_b[0]["numech"]
        first_t = records_t[0]["numech"]

    k0 = first_b - first_t

    results = []
    for k in range(k0 - window, k0 + window + 1):
        accord, conflits, total_fiable = 0, 0, 0
        for nb, vb in dict_b.items():
            nt = nb - k
            if nt in dict_t:
                vt = dict_t[nt]
                if vb == "AMBIGU" or vt == "AMBIGU":
                    continue
                total_fiable += 1
                if vb == vt:
                    accord += 1
                else:
                    conflits += 1
        results.append((k, accord, conflits, total_fiable))

    sans_conflit = [r for r in results if r[2] == 0 and r[3] > 0]
    pool = sans_conflit if sans_conflit else results

    # accord DESCENDANT en premier (meilleur compromis vide<->vide +
    # non-vide<->non-vide), puis total_fiable descendant en cas d'égalité.
    pool.sort(key=lambda r: (-r[1], -r[3]))
    return pool, k0, bool(sans_conflit)


# ---------------------------------------------------------------------------
# Programme principal
# ---------------------------------------------------------------------------

def main():
    src_dir = input("Chemin du dossier contenant les images du dataset : ").strip()
    outdir = os.path.normpath(src_dir) + "_cycles_csv"
    os.makedirs(outdir, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(src_dir, "*.jpg")) + glob.glob(os.path.join(src_dir, "*.jpeg")))
    print(f"{len(paths)} image(s) trouvée(s) dans {src_dir}")

    # Regroupement par (année, label, numCamera, position) : la comparaison
    # T/B n'a de sens qu'à l'intérieur d'un même triplet (année, label,
    # numCamera) -- comparer des images d'années ou de labels différents
    # n'aurait aucune signification physique.
    groups = defaultdict(list)  # (annee, label, numcam, pos) -> [ (numech, filename) ]
    n_non_conformes_nom = 0
    for path in paths:
        filename = os.path.basename(path)
        info = parse_filename(filename)
        if info is None:
            n_non_conformes_nom += 1
            continue
        key = (info["annee"], info["label"], info["numcam"], info["pos"])
        groups[key].append((info["numech"], filename))

    if n_non_conformes_nom:
        print(f"[ATTENTION] {n_non_conformes_nom} fichier(s) ignoré(s) "
              f"(nom non conforme à la convention attendue).")

    # Tri par numEchantillon dans chaque groupe (ordre du "cycle")
    for key in groups:
        groups[key].sort(key=lambda t: t[0])

    # Classification vide/non-vide/AMBIGU + écriture d'un CSV par cycle
    cycle_records = {}  # (annee, label, numcam, pos) -> liste de dicts
    for (annee, label, numcam, pos), items in sorted(groups.items()):
        records = []
        for idx, (numech, filename) in enumerate(items):
            img = cv2.imread(os.path.join(src_dir, filename))
            statut, fraction = classify_vide(img)
            records.append({
                "id_img": idx,
                "nom_img": filename,
                "vide": statut,
                "fraction_brun": fraction,
                "numech": numech,
            })
        cycle_records[(annee, label, numcam, pos)] = records

        csv_path = os.path.join(outdir, f"cycle_{annee}_{label}_Cam_{pos}_{numcam}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(["id_img", "nom_img", "vide", "fraction_brun"])
            for r in records:
                writer.writerow([r["id_img"], r["nom_img"], r["vide"], f"{r['fraction_brun']:.4f}"])
        n_ambigu = sum(1 for r in records if r["vide"] == "AMBIGU")
        n_erreur = sum(1 for r in records if r["vide"] == "ERREUR_DETECTION")
        print(f"{annee}_{label}_Cam_{pos}_{numcam} : {len(records)} image(s) -> {csv_path}  "
              f"({n_ambigu} AMBIGU, {n_erreur} échec(s) de détection du cercle)")

    # Comparaison T/B : uniquement à l'intérieur d'un même triplet
    # (année, label, numCamera), positions opposées (T vs B).
    print("\n=== Correspondance T/B par (année, label, caméra) ===")
    triplets = sorted({(annee, label, numcam) for (annee, label, numcam, pos) in cycle_records})
    for annee, label, numcam in triplets:
        key_t = (annee, label, numcam, "T")
        key_b = (annee, label, numcam, "B")
        if key_t not in cycle_records or key_b not in cycle_records:
            print(f"{annee}_{label}_Cam_{numcam} : pas de paire T/B complète, comparaison impossible.")
            continue

        records_t = cycle_records[key_t]
        records_b = cycle_records[key_b]

        results, k0, has_zero_conflict = find_best_shift_by_numech(records_b, records_t)
        corr_path = os.path.join(outdir, f"correspondence_{annee}_{label}_Cam_{numcam}.csv")

        if not results:
            print(f"{annee}_{label}_Cam_{numcam} : impossible de calculer une correspondance "
                  f"(au moins une séquence vide).")
            with open(corr_path, "w", newline="") as f:
                writer = csv.writer(f, delimiter=";")
                writer.writerow(["RESULTAT"])
                writer.writerow(["IMPOSSIBLE_AUCUN_CYCLE_DETECTE"])
            continue

        best_k, best_accord, best_conflits, best_total_fiable = results[0]
        print(f"{annee}_{label}_Cam_{numcam} : ancrage k0={k0:+d} (1re image non-vide de chaque "
              f"liste), décalage retenu k={best_k:+d}  "
              f"({best_accord} accord(s) [vide<->vide + non-vide<->non-vide], "
              f"{best_conflits} conflit(s), {best_total_fiable} comparaisons fiables)")
        if not has_zero_conflict:
            print(f"  [ATTENTION] AUCUN décalage sans conflit trouvé dans la fenêtre "
                  f"[{k0 - SHIFT_SEARCH_WINDOW:+d}, {k0 + SHIFT_SEARCH_WINDOW:+d}] pour "
                  f"{annee}_{label}_Cam_{numcam} : résultat NON FIABLE, à vérifier manuellement.")
        elif best_total_fiable <= 1:
            print(f"  [ATTENTION] le meilleur décalage ne s'appuie que sur {best_total_fiable} "
                  f"comparaison(s) fiable(s) : support statistique très faible.")

        # Construction de la table de correspondance directement à partir
        # des numEchantillon (numech_T = numech_B - best_k), pas de position
        # de liste -- robuste aux trous de numérotation.
        dict_b_by_numech = {r["numech"]: r for r in records_b}
        dict_t_by_numech = {r["numech"]: r for r in records_t}
        all_numech_b = set(dict_b_by_numech)
        all_numech_t_shifted = {nt + best_k for nt in dict_t_by_numech}  # -> repère numech_B équivalent
        all_positions = sorted(all_numech_b | all_numech_t_shifted)

        with open(corr_path, "w", newline="") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(["numech_B_repere", "nom_img_B", "nom_img_T", "vide_B", "vide_T", "accord"])
            for numech_b_repere in all_positions:
                numech_t_reel = numech_b_repere - best_k
                r_b = dict_b_by_numech.get(numech_b_repere)
                r_t = dict_t_by_numech.get(numech_t_reel)
                nom_b = r_b["nom_img"] if r_b else "NULL"
                nom_t = r_t["nom_img"] if r_t else "NULL"
                v_b = r_b["vide"] if r_b else "NULL"
                v_t = r_t["vide"] if r_t else "NULL"
                if r_b is None or r_t is None:
                    accord = "NULL"
                elif v_b == "AMBIGU" or v_t == "AMBIGU":
                    accord = "AMBIGU"
                elif v_b == v_t:
                    accord = "OK"
                else:
                    accord = "CONFLIT"
                writer.writerow([numech_b_repere, nom_b, nom_t, v_b, v_t, accord])
        print(f"  -> {corr_path}")

    print(f"\nTerminé. Résultats dans : {outdir}")


if __name__ == "__main__":
    main()
