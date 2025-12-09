"""
file to generate simple dataset from dataset 0, 1, and 4 only:
Dataset: dataset/0
      Images: 18000
      RG: 602
      NRG: 17398

   Dataset: dataset/1
      Images: 18000
      RG: 622
      NRG: 17378

   Dataset: dataset/4
      Images: 18000
      RG: 564
      NRG: 17436
Construct final dataset with all the RG images and random same number of NRG images
"""


# python
import os
import sys
import argparse
import shutil
import random
import glob
from pathlib import Path

import pandas as pd

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


def find_label_file(dataset_idx):
    # Cherche train_labels_{idx}.csv dans current dir puis dataset dir
    candidates = [
        Path(f"train_labels_{dataset_idx}.csv"),
        Path("dataset") / f"train_labels_{dataset_idx}.csv",
        Path("dataset") / str(dataset_idx) / "labels.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    # fallback: any file matching *train*{idx}*.csv
    for p in glob.glob(f"*{dataset_idx}*.csv") + glob.glob(f"dataset/*{dataset_idx}*.csv"):
        return Path(p)
    return None


def detect_columns(df):
    # Colonne image: cherche colonne contenant des chaînes avec extension d'image
    for col in df.columns:
        sample = df[col].astype(str).head(50).str.lower()
        if sample.str.contains("|".join([ext.strip(".") for ext in IMAGE_EXTS])).any():
            return col, detect_label_column(df, exclude=col)
    # fallback image column = first
    img_col = df.columns[0]
    lbl_col = detect_label_column(df, exclude=img_col)
    return img_col, lbl_col


def detect_label_column(df, exclude=None):
    preferred = ["glaucoma", "label", "is_glaucoma", "rg", "target"]
    lc = [c for c in df.columns if c not in (exclude,)]
    for name in preferred:
        for c in lc:
            if c.lower() == name:
                return c
    # cherche colonne binaire 0/1
    for c in lc:
        vals = pd.unique(df[c].dropna().astype(str))
        vals_set = set(v.strip() for v in vals)
        if vals_set.issubset({"0", "1", "0.0", "1.0", "nr", "rg", "nrg", "rg"}):
            return c
    # sinon la dernière colonne
    return lc[-1] if lc else None


def normalize_label(val):
    if pd.isna(val):
        return None
    s = str(val).strip().lower()
    if s in {"1", "1.0", "rg", "true", "yes"}:
        return 1
    if s in {"0", "0.0", "nrg", "nr", "false", "no"}:
        return 0
    try:
        iv = int(float(s))
        return 1 if iv == 1 else 0
    except Exception:
        return None


def main(selected=(0, 1, 4), out_dir="dataset/simple", seed=42, copy_images=True):
    random.seed(seed)
    out_dir = Path(out_dir)
    images_out = out_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    rows = []
    total_rg = 0
    total_nrg = 0

    for idx in selected:
        lbl_file = find_label_file(idx)
        if lbl_file is None:
            print(f"Warning: label file for dataset {idx} not found, skipping.", file=sys.stderr)
            continue
        df = pd.read_csv(lbl_file)
        img_col, lbl_col = detect_columns(df)
        if img_col is None or lbl_col is None:
            print(f"Warning: impossible de détecter colonnes dans {lbl_file}, skipping.", file=sys.stderr)
            continue

        # Construire chemin absolu/relatif vers images
        # Si les filenames n'ont pas de chemin, on suppose dataset/{idx}/<filename>
        def build_path(fn):
            fn = str(fn)
            if any(ext in fn.lower() for ext in IMAGE_EXTS) and (Path(fn).parent != Path(".")):
                return Path(fn)
            p1 = Path("dataset") / str(idx) / fn
            p2 = Path("dataset") / fn
            if p1.exists():
                return p1
            if p2.exists():
                return p2
            # sinon garder p1 (peut être absent, sera signalé plus tard)
            return p1

        df = df[[img_col, lbl_col]].rename(columns={img_col: "image", lbl_col: "label"})
        df["image_path"] = df["image"].apply(build_path)
        df["label_norm"] = df["label"].apply(normalize_label)
        # Filter rows with valid label and image extension
        df = df[df["label_norm"].isin([0, 1])]
        rg = df[df["label_norm"] == 1].copy()
        nrg = df[df["label_norm"] == 0].copy()

        print(f"Dataset {idx}: images={len(df)} RG={len(rg)} NRG={len(nrg)}")
        total_rg += len(rg)
        total_nrg += len(nrg)
        rows.append((rg, nrg, idx))

    # concat all RG and sample same number of NRG
    rg_all = pd.concat([r[0] for r in rows]) if rows else pd.DataFrame(columns=["image", "label", "image_path", "label_norm"])
    nrg_all = pd.concat([r[1] for r in rows]) if rows else pd.DataFrame(columns=["image", "label", "image_path", "label_norm"])

    n_needed = len(rg_all)
    if n_needed == 0:
        print("Aucun exemple RG trouvé. Abort.", file=sys.stderr)
        return 1

    if len(nrg_all) < n_needed:
        print(f"Attention: NRG disponibles ({len(nrg_all)}) < RG nécessaires ({n_needed}), on prendra tout le NRG disponible.", file=sys.stderr)
        sampled_nrg = nrg_all
    else:
        sampled_nrg = nrg_all.sample(n=n_needed, random_state=seed)

    final_df = pd.concat([rg_all, sampled_nrg]).reset_index(drop=True)
    # shuffle final
    final_df = final_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Copier les images et écrire le CSV
    out_records = []
    missing = 0
    for i, row in final_df.iterrows():
        src = Path(row["image_path"])
        if not src.exists():
            print(f"Missing image: {src}", file=sys.stderr)
            missing += 1
            continue
        # Préférer garder le nom original mais préfixer par dataset index si collision possible
        dst_name = src.name
        dst = images_out / dst_name
        if dst.exists():
            # éviter collision
            dst = images_out / f"{i}_{dst_name}"
        if copy_images:
            shutil.copy2(src, dst)
        rel_path = Path("images") / dst.name
        out_records.append({"image": str(rel_path), "label": int(row["label_norm"])})

    out_csv = out_dir / "train_labels_simple.csv"
    pd.DataFrame(out_records).to_csv(out_csv, index=False)
    print(f"\nRésultat: total_RG={len(rg_all)} sampled_NRG={len(sampled_nrg)} written={len(out_records)} missing_images={missing}")
    print(f"Fichiers écrits: {out_csv}  et {len(list(images_out.glob('*')))} images dans {images_out}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Génère dataset simple à partir de dataset 0/1/4")
    parser.add_argument("--selected", nargs="+", type=int, default=[0, 1, 4], help="indices des datasets à utiliser")
    parser.add_argument("--out", type=str, default="dataset/simple", help="dossier de sortie")
    parser.add_argument("--seed", type=int, default=42, help="seed pour l'échantillonnage")
    parser.add_argument("--no-copy", action="store_true", help="ne pas copier les images (utile pour debug)")
    args = parser.parse_args()
    rc = main(selected=args.selected, out_dir=args.out, seed=args.seed, copy_images=not args.no_copy)
    sys.exit(rc)
