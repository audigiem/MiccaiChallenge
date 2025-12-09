# python
import os
import sys
import argparse
import shutil
import random
from pathlib import Path

import pandas as pd

def find_label_file(dataset_idx):
    candidates = [
        Path(f"train_labels_{dataset_idx}.csv"),
        Path("dataset") / f"train_labels_{dataset_idx}.csv",
        Path("dataset") / f"{dataset_idx}" / "train_labels.csv",
        Path("dataset") / "train_labels.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

def normalize_label_simple(s):
    s = str(s).strip().upper()
    return 1 if s == "RG" else 0  # assume only RG/NRG

def main(selected=(0, 1, 4), out_dir="dataset/simple", seed=42, copy_images=True):
    random.seed(seed)
    out_dir = Path(out_dir)
    images_out = out_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    rg_list = []
    nrg_list = []

    for idx in selected:
        lbl_file = find_label_file(idx)
        if lbl_file is None:
            print(f"Warning: label file for dataset {idx} not found, skipping.", file=sys.stderr)
            continue

        df = pd.read_csv(lbl_file)
        # format minimal: columns are challenge_id,class
        df = df.rename(columns={df.columns[0]: "challenge_id", df.columns[1]: "class"})[[ "challenge_id", "class" ]]
        df["label"] = df["class"].apply(normalize_label_simple)
        # build image path assuming .jpg
        df["image_path"] = df["challenge_id"].astype(str) + ".jpg"
        df["image_path"] = df["image_path"].apply(lambda fn: Path("dataset") / str(idx) / fn)

        rg = df[df["label"] == 1].copy()
        nrg = df[df["label"] == 0].copy()

        print(f"Dataset {idx}: images={len(df)} RG={len(rg)} NRG={len(nrg)}")
        rg_list.append(rg)
        nrg_list.append(nrg)

    if not rg_list:
        print("Aucun dataset valide trouvé. Abort.", file=sys.stderr)
        return 1

    rg_all = pd.concat(rg_list, ignore_index=True)
    nrg_all = pd.concat(nrg_list, ignore_index=True)

    n_needed = len(rg_all)
    if len(nrg_all) <= n_needed:
        sampled_nrg = nrg_all
    else:
        sampled_nrg = nrg_all.sample(n=n_needed, random_state=seed)

    final_df = pd.concat([rg_all, sampled_nrg], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    out_records = []
    missing = 0
    for i, row in final_df.iterrows():
        src = Path(row["image_path"])
        if not src.exists():
            print(f"Missing image: {src}", file=sys.stderr)
            missing += 1
            continue
        dst_name = src.name
        dst = images_out / dst_name
        if dst.exists():
            dst = images_out / f"{i}_{dst_name}"
        if copy_images:
            shutil.copy2(src, dst)
        # write challenge_id (sans extension) and textual class RG/NRG
        challenge_id = Path(dst.name).stem
        class_text = "RG" if int(row["label"]) == 1 else "NRG"
        out_records.append({"challenge_id": challenge_id, "class": class_text})

    out_csv = out_dir / "train_labels_simple.csv"
    pd.DataFrame(out_records).to_csv(out_csv, index=False)
    print(f"\nRésultat: total_RG={len(rg_all)} sampled_NRG={len(sampled_nrg)} written={len(out_records)} missing_images={missing}")
    print(f"Fichiers écrits: {out_csv}  et {len(list(images_out.glob('*')))} images dans {images_out}")
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Génère dataset simple à partir de dataset 0/1/4 (format minimal)")
    parser.add_argument("--selected", nargs="+", type=int, default=[0, 1, 4])
    parser.add_argument("--out", type=str, default="dataset/simple")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-copy", action="store_true")
    args = parser.parse_args()
    rc = main(selected=args.selected, out_dir=args.out, seed=args.seed, copy_images=not args.no_copy)
    sys.exit(rc)
