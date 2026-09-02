"""Outil de labelisation pour le dataset de châtaignes.

Lancement :
    streamlit run labeling_tool/app.py

Modifie labels_principal.csv (à la racine du dépôt) en place.
"""

import os

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(ROOT, "images")
LABELS_CSV = os.path.join(ROOT, "labels_principal.csv")

LABELS = ["Conforme", "NON Conforme", "PIETRA", "Vide"]

st.set_page_config(page_title="Labelisation Châtaignes", layout="wide")


def load_labels():
    df = pd.read_csv(LABELS_CSV, dtype={"labeled_by": str})
    df["labeled_by"] = df["labeled_by"].fillna("")
    return df


def save_labels(df):
    df.to_csv(LABELS_CSV, index=False)


@st.cache_data(show_spinner=False)
def load_thumbnail(path, max_size=220):
    img = Image.open(path)
    img.thumbnail((max_size, max_size))
    return img


if "df" not in st.session_state:
    st.session_state.df = load_labels()
if "pos_idx" not in st.session_state:
    st.session_state.pos_idx = 0

df = st.session_state.df

# --- Sidebar : labeliseur + filtres -----------------------------------------
st.sidebar.title("Filtres")

labeler = st.sidebar.text_input("Ton nom / pseudo", value=st.session_state.get("labeler", ""))
st.session_state.labeler = labeler

mode = st.sidebar.radio(
    "Mode d'affichage", ["Une image à la fois", "Galerie (batch)"], index=0
)

years = sorted(df["year"].unique())
year_sel = st.sidebar.multiselect("Année", years, default=years)

cats = sorted(df["label_filename"].unique())
cat_sel = st.sidebar.multiselect("Catégorie (nom de fichier)", cats, default=cats)

positions = sorted(df["cam_position"].unique())
pos_sel = st.sidebar.multiselect("Position caméra (T/B)", positions, default=positions)

cams = sorted(df["cam_num"].unique())
cam_sel = st.sidebar.multiselect("Numéro de caméra", cams, default=cams)

only_unreviewed = st.sidebar.checkbox("Afficher seulement les images non revues", value=True)

# --- Calcul du sous-ensemble filtré ------------------------------------------
base_mask = (
    df["year"].isin(year_sel)
    & df["label_filename"].isin(cat_sel)
    & df["cam_position"].isin(pos_sel)
    & df["cam_num"].isin(cam_sel)
)

total_in_filter = int(base_mask.sum())
reviewed_in_filter = int((base_mask & df["reviewed"]).sum())

progress_ratio = reviewed_in_filter / total_in_filter if total_in_filter else 0

st.sidebar.markdown("---")
st.sidebar.write(f"**{reviewed_in_filter} / {total_in_filter}** images revues dans ce filtre ({progress_ratio:.0%})")
st.sidebar.progress(progress_ratio)

if only_unreviewed:
    display_mask = base_mask & ~df["reviewed"]
else:
    display_mask = base_mask

filtered_indices = df.index[display_mask].tolist()

st.title("Labelisation des images de châtaignes")

st.metric(
    "Avancement sur le filtre actuel",
    f"{reviewed_in_filter} / {total_in_filter}",
    f"{progress_ratio:.0%}",
)
st.progress(progress_ratio)

if not filtered_indices:
    st.success("Toutes les images de ce filtre sont revues ! 🎉")
    st.stop()

if mode == "Une image à la fois":
    st.session_state.pos_idx = max(0, min(st.session_state.pos_idx, len(filtered_indices) - 1))
    current_idx = filtered_indices[st.session_state.pos_idx]
    row = df.loc[current_idx]

    # --- Affichage -----------------------------------------------------------------
    col_img, col_controls = st.columns([2, 1])

    with col_img:
        img_path = os.path.join(IMAGES_DIR, row["filename"])
        st.image(Image.open(img_path), caption=row["filename"], use_container_width=True)
        st.caption(f"Image {st.session_state.pos_idx + 1} / {len(filtered_indices)} dans le filtre actuel")

    with col_controls:
        st.subheader("Tags")
        multiple = st.checkbox(
            "Plusieurs châtaignes", value=bool(row["multiple"]), key=f"multiple_{current_idx}"
        )
        chunk = st.checkbox(
            "Morceau / débris", value=bool(row["chunk"]), key=f"chunk_{current_idx}"
        )
        mixed_quality = st.checkbox(
            "Qualité mixte (bon + mauvais)", value=bool(row["mixed_quality"]), key=f"mixed_{current_idx}"
        )

        components.html(
            """
            <script>
            (function() {
                const doc = window.parent.document;

                function clickCheckbox(ariaLabel) {
                    const input = doc.querySelector('input[aria-label="' + ariaLabel + '"]');
                    if (input) {
                        input.click();
                    }
                }

                function handleKeydown(e) {
                    if (e.metaKey || e.ctrlKey || e.altKey) return;
                    const active = doc.activeElement;
                    const tag = active ? active.tagName.toLowerCase() : "";
                    if (tag === "input" || tag === "textarea") {
                        if (!active.type || active.type === "text" || active.type === "search") return;
                    }
                    if (e.key === "d" || e.key === "D") {
                        clickCheckbox("Morceau / débris");
                        e.preventDefault();
                    } else if (e.key === "s" || e.key === "S") {
                        clickCheckbox("Plusieurs châtaignes");
                        e.preventDefault();
                    }
                }

                if (!doc._castaShortcutsAttached) {
                    doc.addEventListener("keydown", handleKeydown);
                    doc._castaShortcutsAttached = true;
                }
            })();
            </script>
            """,
            height=0,
            width=0,
        )

        st.subheader("Label principal")
        st.write(f"Pré-rempli (nom de fichier) : **{row['label_filename']}**")
        if row["reviewed"]:
            st.caption(f"Déjà revu — label actuel : **{row['label_principal']}** (par {row['labeled_by'] or '?'})")

        def confirm_and_advance(label):
            df.at[current_idx, "label_principal"] = label
            df.at[current_idx, "multiple"] = multiple
            df.at[current_idx, "chunk"] = chunk
            df.at[current_idx, "mixed_quality"] = mixed_quality
            df.at[current_idx, "reviewed"] = True
            df.at[current_idx, "labeled_by"] = labeler
            save_labels(df)
            if not only_unreviewed:
                st.session_state.pos_idx += 1

        btn_cols = st.columns(2)
        for i, label in enumerate(LABELS):
            is_current = row["reviewed"] and row["label_principal"] == label
            button_label = ("✅ " if is_current else "") + label
            if btn_cols[i % 2].button(button_label, key=f"btn_{label}_{current_idx}", use_container_width=True):
                confirm_and_advance(label)
                st.rerun()

        st.markdown("---")
        nav_cols = st.columns(2)
        if nav_cols[0].button("⬅️ Précédent", disabled=st.session_state.pos_idx == 0, use_container_width=True):
            st.session_state.pos_idx -= 1
            st.rerun()
        if nav_cols[1].button("Suivant — confirmer le label affiché ➡️", use_container_width=True):
            confirm_and_advance(row["label_principal"])
            st.rerun()

else:
    # --- Mode galerie : revue par batch ------------------------------------------
    st.subheader("Galerie — revue par batch")

    batch_size = st.sidebar.selectbox("Taille de batch", [100, 250, 500], index=2)
    n_cols = 5

    n_batches = max(1, -(-len(filtered_indices) // batch_size))  # ceil division
    if "gallery_batch" not in st.session_state:
        st.session_state.gallery_batch = 0
    st.session_state.gallery_batch = max(0, min(st.session_state.gallery_batch, n_batches - 1))

    nav_cols = st.columns([1, 2, 1])
    if nav_cols[0].button(
        "⬅️ Batch précédent", disabled=st.session_state.gallery_batch == 0, use_container_width=True
    ):
        st.session_state.gallery_batch -= 1
        st.rerun()
    nav_cols[1].markdown(
        f"<div style='text-align:center; padding-top: 0.5em;'>"
        f"Batch {st.session_state.gallery_batch + 1} / {n_batches}</div>",
        unsafe_allow_html=True,
    )
    if nav_cols[2].button(
        "Batch suivant ➡️", disabled=st.session_state.gallery_batch >= n_batches - 1, use_container_width=True
    ):
        st.session_state.gallery_batch += 1
        st.rerun()

    start = st.session_state.gallery_batch * batch_size
    end = min(start + batch_size, len(filtered_indices))
    batch_indices = filtered_indices[start:end]

    st.caption(f"Images {start + 1} à {end} sur {len(filtered_indices)} dans le filtre actuel")

    def validate_batch():
        for idx in batch_indices:
            row = df.loc[idx]
            is_empty = st.session_state.get(f"g_empty_{idx}", False)
            df.at[idx, "multiple"] = st.session_state.get(f"g_multiple_{idx}", False)
            df.at[idx, "chunk"] = st.session_state.get(f"g_chunk_{idx}", False)
            if is_empty:
                df.at[idx, "label_principal"] = "Vide"
            elif row["label_principal"] == "Vide":
                # la case Empty a été décochée : on revient au label pré-rempli
                df.at[idx, "label_principal"] = row["label_filename"]
            df.at[idx, "reviewed"] = True
            df.at[idx, "labeled_by"] = labeler
        save_labels(df)
        if only_unreviewed:
            st.session_state.gallery_batch = 0
        else:
            st.session_state.gallery_batch = min(st.session_state.gallery_batch + 1, n_batches - 1)

    if st.button(
        f"✅ Valider ce batch ({len(batch_indices)} images)",
        key="validate_top",
        type="primary",
        use_container_width=True,
    ):
        validate_batch()
        st.rerun()

    st.markdown("---")

    grid_cols = st.columns(n_cols)
    for i, idx in enumerate(batch_indices):
        row = df.loc[idx]
        with grid_cols[i % n_cols]:
            img_path = os.path.join(IMAGES_DIR, row["filename"])
            st.image(load_thumbnail(img_path), use_container_width=True)
            st.caption(row["filename"])
            st.checkbox("Multiple", value=bool(row["multiple"]), key=f"g_multiple_{idx}")
            st.checkbox("Chunks", value=bool(row["chunk"]), key=f"g_chunk_{idx}")
            st.checkbox("Empty", value=bool(row["reviewed"]) and row["label_principal"] == "Vide", key=f"g_empty_{idx}")

    st.markdown("---")
    if st.button(
        f"✅ Valider ce batch ({len(batch_indices)} images)",
        key="validate_bottom",
        type="primary",
        use_container_width=True,
    ):
        validate_batch()
        st.rerun()
