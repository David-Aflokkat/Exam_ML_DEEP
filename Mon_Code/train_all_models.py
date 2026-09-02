"""
CastagNet - Entraînement des modèles de classification (§4.2)
=================================================================

Exécute une série d'entraînements en séquence, conçue pour tourner sans
supervision sur plusieurs heures, en s'appuyant sur les décisions prises :

  - Deux modèles indépendants (caméra T et caméra B), pas de correspondance
    T/B sur ce dataset (cf. rapport §4.1).
  - Les images classées "vide franc" sont déjà exclues de img_dataset_exam ;
    les AMBIGU y sont en revanche inclus (97,2 % sont de vraies châtaignes
    d'après la vérité terrain).
  - Trois classes cibles : Conforme / NON Conforme / PIETRA (label_principal
    de labels_principal.csv, PAS le label du nom de fichier).
  - Trois approches comparées, pour chaque caméra :
      1. CNN "from scratch" léger, sur le dataset déséquilibré tel quel.
      2. Le même CNN, avec les classes minoritaires (NON Conforme, PIETRA)
         doublées par flip horizontal pour rééquilibrer.
      3. Un classifieur ML classique (Random Forest) sur des caractéristiques
         géométriques/colorimétriques simples, avec pondération de classe
         équivalente (class_weight='balanced') plutôt qu'un flip (qui n'a
         pas de sens sur des caractéristiques déjà quasi invariantes par
         symétrie horizontale).
  - CNN volontairement frugal (peu de couches, faible résolution, global
    average pooling plutôt qu'un Dense énorme) en vue d'un déploiement sur
    matériel très contraint en mémoire (cf. cahier des charges GRPTMC).

Toutes les métriques, hyperparamètres et matrices de confusion sont loggés
dans MLflow (un experiment par approche : cnn_baseline, cnn_flip_augmente,
ml_classique). Chaque expérience est isolée dans un bloc try/except : un
échec sur l'une n'interrompt pas les suivantes -- important pour un
déroulement sans supervision.

Prérequis
---------
    pip install torch torchvision scikit-learn mlflow opencv-python-headless

Usage
-----
    python3 train_all_models.py
    (demande interactivement le dossier des images et le CSV de vérité terrain)

Les résultats sont consultables ensuite avec :
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import csv
import os
import re
import time
import traceback
from collections import Counter, defaultdict

import cv2
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

import mlflow
import mlflow.pytorch
import mlflow.sklearn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Configuration générale (ajustable)
# ---------------------------------------------------------------------------
VALID_LABELS = ["Conforme", "NON Conforme", "PIETRA"]
FLIP_CLASSES = {"NON Conforme", "PIETRA"}  # classes doublées par flip horizontal

RESOLUTION = 128        # taille d'entrée du CNN (carré), volontairement modeste
BATCH_SIZE = 32
MAX_EPOCHS = 30
PATIENCE = 6             # early stopping : arrêt si pas d'amélioration en N époques
LEARNING_RATE = 1e-3
SEED = 42

FILENAME_PATTERN = re.compile(
    r"^(?P<annee>\d{4})_(?P<label>.+?)_Cam_(?P<pos>[TB])_(?P<numcam>\d+)_(?P<numech>\d+)\.jpe?g$",
    re.IGNORECASE,
)

LOG_PATH = "train_all_models.log"


def log(msg):
    """Affiche et journalise un message (utile pour relecture au réveil)."""
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Construction de l'index du dataset
# ---------------------------------------------------------------------------

def build_index(img_dir, labels_csv):
    """Associe chaque image de img_dir à son label_principal (vérité terrain)
    et sa position caméra (T/B), en ignorant les images sans label valide
    (Vide résiduel, ou absentes de labels_principal.csv)."""
    truth = {}
    with open(labels_csv, newline="") as f:
        for row in csv.DictReader(f):
            truth[row["filename"]] = row["label_principal"]

    entries = []
    n_ignorees = 0
    for fname in sorted(os.listdir(img_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg")):
            continue
        label = truth.get(fname)
        if label not in VALID_LABELS:
            n_ignorees += 1
            continue
        m = FILENAME_PATTERN.match(fname)
        if m is None:
            n_ignorees += 1
            continue
        entries.append({
            "path": os.path.join(img_dir, fname),
            "label": label,
            "position": m.group("pos").upper(),
        })

    log(f"Index construit : {len(entries)} image(s) valide(s), {n_ignorees} ignorée(s) "
        f"(label hors {VALID_LABELS} ou nom non conforme).")
    return entries


def split_stratified(entries, seed=SEED):
    """Découpe 70/15/15 stratifié par label."""
    labels = [e["label"] for e in entries]
    train, temp = train_test_split(entries, test_size=0.30, stratify=labels, random_state=seed)
    temp_labels = [e["label"] for e in temp]
    val, test = train_test_split(temp, test_size=0.50, stratify=temp_labels, random_state=seed)
    return train, val, test


# ---------------------------------------------------------------------------
# Dataset PyTorch (CNN)
# ---------------------------------------------------------------------------

class ChestnutDataset(Dataset):
    """Charge les images en RGB, redimensionne à RESOLUTION x RESOLUTION.
    Si flip_classes est fourni, chaque image dont le label y figure est
    dupliquée (originale + version miroir horizontal) dans l'index -- c'est
    le mécanisme de rééquilibrage par flip décidé pour l'expérience 2."""

    def __init__(self, entries, label_to_idx, flip_classes=None):
        self.label_to_idx = label_to_idx
        self.items = []
        for e in entries:
            self.items.append((e, False))
            if flip_classes and e["label"] in flip_classes:
                self.items.append((e, True))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        entry, flip = self.items[idx]
        img = cv2.imread(entry["path"])
        if img is None:
            # Repli robuste : image illisible -> image noire (rare, ne doit
            # pas faire planter tout un entraînement de plusieurs heures)
            img = np.zeros((RESOLUTION, RESOLUTION, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if flip:
                img = cv2.flip(img, 1)
            img = cv2.resize(img, (RESOLUTION, RESOLUTION))

        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)  # HWC -> CHW
        label = self.label_to_idx[entry["label"]]
        return img, label


# ---------------------------------------------------------------------------
# Architecture CNN (volontairement légère)
# ---------------------------------------------------------------------------

class SmallCNN(nn.Module):
    """CNN "from scratch" à 3 blocs conv + global average pooling, pensé
    pour un déploiement sur matériel très contraint en mémoire : peu de
    canaux, pas de couche Dense géante en sortie des features (le GAP
    ramène directement à un vecteur de taille = nb de canaux du dernier
    bloc, indépendamment de la résolution d'entrée)."""

    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Utilitaires communs (matrice de confusion, métriques)
# ---------------------------------------------------------------------------

def save_confusion_matrix(y_true, y_pred, class_names, out_path, title):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=20)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    ax.set_title(title)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return cm


def log_classification_metrics(y_true, y_pred, class_names, prefix=""):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0
    )
    for i, cname in enumerate(class_names):
        safe_name = cname.replace(" ", "_")
        mlflow.log_metric(f"{prefix}precision_{safe_name}", precision[i])
        mlflow.log_metric(f"{prefix}recall_{safe_name}", recall[i])
        mlflow.log_metric(f"{prefix}f1_{safe_name}", f1[i])


# ---------------------------------------------------------------------------
# Entraînement CNN
# ---------------------------------------------------------------------------

def train_cnn(train_entries, val_entries, test_entries, class_names, run_name,
              experiment_name, device, flip_classes=None):
    label_to_idx = {c: i for i, c in enumerate(class_names)}

    train_ds = ChestnutDataset(train_entries, label_to_idx, flip_classes=flip_classes)
    val_ds = ChestnutDataset(val_entries, label_to_idx)
    test_ds = ChestnutDataset(test_entries, label_to_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("resolution", RESOLUTION)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("architecture", "conv16-32-64_gap")
        mlflow.log_param("max_epochs", MAX_EPOCHS)
        mlflow.log_param("patience", PATIENCE)
        mlflow.log_param("flip_classes", ",".join(sorted(flip_classes)) if flip_classes else "aucune")
        mlflow.log_param("n_train_images_effectif", len(train_ds))
        mlflow.log_param("n_val", len(val_ds))
        mlflow.log_param("n_test", len(test_ds))

        model = SmallCNN(n_classes=len(class_names)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()

        best_val_acc = -1.0
        best_state = None
        epochs_no_improve = 0

        for epoch in range(MAX_EPOCHS):
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * x.size(0)
                train_correct += (out.argmax(1) == y).sum().item()
                train_total += x.size(0)
            train_loss /= max(train_total, 1)
            train_acc = train_correct / max(train_total, 1)

            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    out = model(x)
                    loss = criterion(out, y)
                    val_loss += loss.item() * x.size(0)
                    val_correct += (out.argmax(1) == y).sum().item()
                    val_total += x.size(0)
            val_loss /= max(val_total, 1)
            val_acc = val_correct / max(val_total, 1)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_acc", train_acc, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)

            log(f"[{run_name}] époque {epoch+1}/{MAX_EPOCHS} "
                f"train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= PATIENCE:
                    log(f"[{run_name}] early stopping à l'époque {epoch+1}")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        mlflow.log_metric("best_val_acc", best_val_acc)

        # Évaluation finale sur le jeu de test
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                out = model(x)
                y_pred.extend(out.argmax(1).cpu().tolist())
                y_true.extend(y.tolist())

        test_acc = accuracy_score(y_true, y_pred)
        mlflow.log_metric("test_acc", test_acc)
        log_classification_metrics(y_true, y_pred, class_names, prefix="test_")

        cm_path = f"confusion_{run_name}.png"
        save_confusion_matrix(y_true, y_pred, class_names, cm_path,
                               f"Matrice de confusion - {run_name}")
        mlflow.log_artifact(cm_path)
        os.remove(cm_path)

        # Sauvegarde du modèle : on repasse explicitement sur CPU avant de
        # logger (le format de sérialisation par défaut de MLflow récent,
        # 'pt2', trace le modèle avec l'exemple fourni -- un mélange
        # d'appareils CPU/MPS fait échouer cette trace. Le format 'pickle'
        # est plus simple et robuste, et le modèle sur CPU reste rechargeable
        # sur n'importe quelle machine ensuite, y compris sans MPS).
        model_cpu = model.to("cpu")
        mlflow.pytorch.log_model(
            model_cpu, "model",
            input_example=np.zeros((1, 3, RESOLUTION, RESOLUTION), dtype=np.float32),
            serialization_format="pickle",
        )

        log(f"[{run_name}] TERMINÉ -- test_acc={test_acc:.3f} (best_val_acc={best_val_acc:.3f})")
        return test_acc


# ---------------------------------------------------------------------------
# Classifieur ML classique (features géométriques/colorimétriques)
# ---------------------------------------------------------------------------

HSV_GREEN_LOWER = np.array([40, 80, 80])
HSV_GREEN_UPPER = np.array([85, 255, 255])
HSV_BROWN_LOWER = np.array([5, 30, 60])
HSV_BROWN_UPPER = np.array([30, 200, 230])
INTERIOR_MARGIN_RATIO = 0.85

FEATURE_KEYS = ["fraction_brun", "aire_norm", "largeur_norm", "hauteur_norm",
                "centre_x_norm", "centre_y_norm"]


def extract_ml_features(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, HSV_GREEN_LOWER, HSV_GREEN_UPPER)
    ys, xs = np.where(mask_green > 0)
    if len(xs) == 0:
        return None
    pts = np.column_stack([xs, ys]).astype(np.float32)
    (cx, cy), r = cv2.minEnclosingCircle(pts)
    r_in = r * INTERIOR_MARGIN_RATIO

    h, w = img.shape[:2]
    Y, X = np.ogrid[:h, :w]
    circle_mask = (X - cx) ** 2 + (Y - cy) ** 2 <= r_in ** 2

    brown_mask = cv2.inRange(hsv, HSV_BROWN_LOWER, HSV_BROWN_UPPER)
    brown_in_circle = (brown_mask.astype(bool) & circle_mask).astype(np.uint8) * 255
    fraction = brown_in_circle.sum() / 255 / circle_mask.sum()

    contours, _ = cv2.findContours(brown_in_circle, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"fraction_brun": fraction, "aire_norm": 0.0, "largeur_norm": 0.0,
                "hauteur_norm": 0.0, "centre_x_norm": 0.0, "centre_y_norm": 0.0}

    biggest = max(contours, key=cv2.contourArea)
    aire = cv2.contourArea(biggest)
    bx, by, bw, bh = cv2.boundingRect(biggest)
    mx, my = bx + bw / 2, by + bh / 2

    return {
        "fraction_brun": fraction,
        "aire_norm": aire / (r_in ** 2),
        "largeur_norm": bw / (2 * r_in),
        "hauteur_norm": bh / (2 * r_in),
        "centre_x_norm": (mx - cx) / r_in,
        "centre_y_norm": (my - cy) / r_in,
    }


def build_feature_matrix(entries):
    X, y, kept = [], [], []
    for e in entries:
        feats = extract_ml_features(e["path"])
        if feats is None:
            continue
        X.append([feats[k] for k in FEATURE_KEYS])
        y.append(e["label"])
        kept.append(e)
    return np.array(X), np.array(y), kept


def train_ml_classifier(train_entries, val_entries, test_entries, class_names, run_name, experiment_name):
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        log(f"[{run_name}] extraction des caractéristiques (train/val/test)...")
        X_train, y_train, _ = build_feature_matrix(train_entries)
        X_val, y_val, _ = build_feature_matrix(val_entries)
        X_test, y_test, _ = build_feature_matrix(test_entries)

        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 300)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("features", ",".join(FEATURE_KEYS))
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_val", len(X_val))
        mlflow.log_param("n_test", len(X_test))

        clf = RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1
        )
        clf.fit(X_train, y_train)

        val_acc = clf.score(X_val, y_val)
        mlflow.log_metric("val_acc", val_acc)

        y_pred = clf.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric("test_acc", test_acc)

        # feature importances (utile pour l'interprétation)
        for feat_name, importance in zip(FEATURE_KEYS, clf.feature_importances_):
            mlflow.log_metric(f"importance_{feat_name}", importance)

        y_true_idx = [class_names.index(v) for v in y_test]
        y_pred_idx = [class_names.index(v) for v in y_pred]
        log_classification_metrics(y_true_idx, y_pred_idx, class_names, prefix="test_")

        cm_path = f"confusion_{run_name}.png"
        save_confusion_matrix(y_true_idx, y_pred_idx, class_names, cm_path,
                               f"Matrice de confusion - {run_name}")
        mlflow.log_artifact(cm_path)
        os.remove(cm_path)

        mlflow.sklearn.log_model(clf, "model")

        log(f"[{run_name}] TERMINÉ -- test_acc={test_acc:.3f} (val_acc={val_acc:.3f})")
        return test_acc


# ---------------------------------------------------------------------------
# Orchestration principale
# ---------------------------------------------------------------------------

def run_experiment_safe(func, *args, **kwargs):
    """Exécute une expérience en isolant les erreurs -- un échec n'interrompt
    pas la suite de la nuit d'entraînement."""
    name = kwargs.get("run_name", func.__name__)
    try:
        return func(*args, **kwargs)
    except Exception:
        log(f"[{name}] ÉCHEC : \n{traceback.format_exc()}")
        return None


def main():
    img_dir = input("Dossier des images d'entraînement (img_dataset_exam) : ").strip()
    labels_csv = input("Chemin de labels_principal.csv : ").strip()

    # Backend SQLite local (le simple dossier mlruns/ est déprécié dans les
    # versions récentes de MLflow) -- aucun service à installer, un seul
    # fichier mlflow.db créé dans le dossier courant.
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    open(LOG_PATH, "w").close()  # reset du fichier de log
    log("=== Démarrage de la campagne d'entraînement ===")

    device = get_device()
    log(f"Device utilisé : {device}")

    entries = build_index(img_dir, labels_csv)
    if not entries:
        log("Aucune image valide trouvée, arrêt.")
        return

    class_names = sorted({e["label"] for e in entries})
    log(f"Classes détectées : {class_names}")

    results = {}

    for position in ["B", "T"]:
        pos_entries = [e for e in entries if e["position"] == position]
        if not pos_entries:
            log(f"Aucune image pour la caméra {position}, passage à la suivante.")
            continue

        counts = Counter(e["label"] for e in pos_entries)
        log(f"\n--- Caméra {position} : {len(pos_entries)} image(s) -- répartition {dict(counts)} ---")

        train_e, val_e, test_e = split_stratified(pos_entries)
        log(f"Split caméra {position} : train={len(train_e)} val={len(val_e)} test={len(test_e)}")

        # 1. CNN baseline (pas d'augmentation)
        acc = run_experiment_safe(
            train_cnn, train_e, val_e, test_e, class_names,
            run_name=f"cnn_baseline_cam{position}", experiment_name="cnn_baseline",
            device=device, flip_classes=None,
        )
        results[f"cnn_baseline_cam{position}"] = acc

        # 2. CNN avec rééquilibrage par flip horizontal des classes minoritaires
        acc = run_experiment_safe(
            train_cnn, train_e, val_e, test_e, class_names,
            run_name=f"cnn_flip_cam{position}", experiment_name="cnn_flip_augmente",
            device=device, flip_classes=FLIP_CLASSES,
        )
        results[f"cnn_flip_cam{position}"] = acc

        # 3. Classifieur ML classique (Random Forest, class_weight balanced)
        acc = run_experiment_safe(
            train_ml_classifier, train_e, val_e, test_e, class_names,
            run_name=f"ml_rf_cam{position}", experiment_name="ml_classique",
        )
        results[f"ml_rf_cam{position}"] = acc

    log("\n=== Campagne terminée -- résumé des test_acc ===")
    for name, acc in results.items():
        log(f"  {name:<25s} : {acc:.3f}" if acc is not None else f"  {name:<25s} : ÉCHEC")

    log("\nPour consulter les résultats en détail : lancer `mlflow ui` dans ce dossier.")


if __name__ == "__main__":
    main()
