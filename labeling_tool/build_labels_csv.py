"""Initialise ou met à jour labels_principal.csv à partir des images du dossier images/.

Usage:
    python labeling_tool/build_labels_csv.py

- Si labels_principal.csv n'existe pas, il est créé avec une ligne par image,
  label_principal pré-rempli avec le label déduit du nom de fichier.
- S'il existe déjà, seules les nouvelles images (pas encore dans le CSV)
  sont ajoutées. Les lignes existantes ne sont jamais modifiées.
"""

import os
import re

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(ROOT, "images")
OUTPUT_CSV = os.path.join(ROOT, "labels_principal.csv")

PATTERN = re.compile(
    r"^(?P<year>\d{4})_(?P<label>NON_Conforme|Conforme|PIETRA)"
    r"(?:_(?P<batch>\d+))?_Cam_(?P<pos>[TB])_(?P<cam>\d+)_(?P<sample>\d+)\.jpg$"
)

NEW_COLUMNS = {
    "multiple": False,
    "chunk": False,
    "mixed_quality": False,
    "reviewed": False,
    "labeled_by": "",
}


def parse_filename(filename):
    m = PATTERN.match(filename)
    if not m:
        raise ValueError(f"Nom de fichier inattendu : {filename}")
    return {
        "filename": filename,
        "year": int(m.group("year")),
        "cam_position": m.group("pos"),
        "cam_num": int(m.group("cam")),
        "sample_num": int(m.group("sample")),
        "label_filename": m.group("label").replace("_", " "),
    }


def main():
    files = sorted(f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(".jpg"))
    new_df = pd.DataFrame(parse_filename(f) for f in files)
    new_df["label_principal"] = new_df["label_filename"]
    for col, default in NEW_COLUMNS.items():
        new_df[col] = default

    if os.path.exists(OUTPUT_CSV):
        existing = pd.read_csv(OUTPUT_CSV, dtype={"labeled_by": str})
        existing["labeled_by"] = existing["labeled_by"].fillna("")
        known = set(existing["filename"])
        added = new_df[~new_df["filename"].isin(known)]
        if added.empty:
            print("Aucune nouvelle image, fichier inchangé.")
            return
        result = pd.concat([existing, added], ignore_index=True)
    else:
        result = new_df

    result = result.sort_values("filename").reset_index(drop=True)
    result.to_csv(OUTPUT_CSV, index=False)
    print(f"{OUTPUT_CSV}: {len(result)} images au total ({len(new_df)} dans images/).")


if __name__ == "__main__":
    main()
