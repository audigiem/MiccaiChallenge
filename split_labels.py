"""
Script pour diviser train_labels.csv en plusieurs fichiers selon les datasets
À exécuter sur le cluster où se trouvent les données
"""

import os
import pandas as pd
import argparse


def split_labels_by_dataset(labels_csv, dataset_base_dir, output_dir=None):
    """
    Divise train_labels.csv en plusieurs fichiers selon les images présentes dans chaque dataset

    Args:
        labels_csv: Chemin vers train_labels.csv
        dataset_base_dir: Répertoire de base contenant les sous-dossiers (ex: dataset/datasetPart1/)
        output_dir: Répertoire où sauvegarder les fichiers de labels divisés (par défaut: même que labels_csv)
    """

    # Charger le fichier labels complet
    print(f"📂 Chargement de {labels_csv}...")
    df = pd.read_csv(labels_csv)
    print(f"✅ {len(df)} labels chargés")
    print(f"   RG (Glaucoma): {(df['class'] == 'RG').sum()}")
    print(f"   NRG (No Glaucoma): {(df['class'] == 'NRG').sum()}")

    # Déterminer le répertoire de sortie
    if output_dir is None:
        output_dir = os.path.dirname(labels_csv)
    os.makedirs(output_dir, exist_ok=True)

    # Trouver tous les sous-dossiers de dataset
    dataset_dirs = []
    if os.path.exists(dataset_base_dir):
        for item in sorted(os.listdir(dataset_base_dir)):
            item_path = os.path.join(dataset_base_dir, item)
            if os.path.isdir(item_path):
                dataset_dirs.append((item, item_path))

    if not dataset_dirs:
        print(f"❌ Aucun sous-dossier trouvé dans {dataset_base_dir}")
        return

    print(f"\n📁 {len(dataset_dirs)} dossiers de dataset trouvés:")
    for name, path in dataset_dirs:
        print(f"   - {name}: {path}")

    # Pour chaque dossier de dataset, créer un fichier labels correspondant
    total_matched = 0
    for dataset_name, dataset_path in dataset_dirs:
        print(f"\n🔍 Traitement de {dataset_name}...")

        # Lister toutes les images dans ce dossier
        image_files = set()
        if os.path.exists(dataset_path):
            for fname in os.listdir(dataset_path):
                if fname.endswith('.jpg') or fname.endswith('.png'):
                    # Extraire l'ID (nom sans extension)
                    image_id = os.path.splitext(fname)[0]
                    image_files.add(image_id)

        print(f"   📷 {len(image_files)} images trouvées dans le dossier")

        # Filtrer le dataframe pour ne garder que les IDs présents
        df_subset = df[df['challenge_id'].isin(image_files)].copy()

        print(f"   ✅ {len(df_subset)} labels correspondants")
        print(f"      RG: {(df_subset['class'] == 'RG').sum()}")
        print(f"      NRG: {(df_subset['class'] == 'NRG').sum()}")

        # Sauvegarder le fichier labels pour ce dataset
        output_file = os.path.join(output_dir, f"train_labels_{dataset_name}.csv")
        df_subset.to_csv(output_file, index=False)
        print(f"   💾 Sauvegardé dans {output_file}")

        total_matched += len(df_subset)

    print(f"\n✅ Terminé ! {total_matched}/{len(df)} labels associés à des images")

    # Vérifier s'il reste des labels non associés
    unmatched = len(df) - total_matched
    if unmatched > 0:
        print(f"⚠️  {unmatched} labels n'ont pas été associés à des images")


def main():
    parser = argparse.ArgumentParser(
        description="Diviser train_labels.csv en plusieurs fichiers selon les datasets"
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="dataset/train_labels.csv",
        help="Chemin vers train_labels.csv (défaut: dataset/train_labels.csv)"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="dataset/datasetPart1",
        help="Répertoire de base contenant les sous-dossiers (défaut: dataset/datasetPart1)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Répertoire de sortie (défaut: même que --labels)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.labels):
        print(f"❌ Fichier labels introuvable : {args.labels}")
        return 1

    if not os.path.exists(args.dataset_dir):
        print(f"❌ Répertoire dataset introuvable : {args.dataset_dir}")
        return 1

    split_labels_by_dataset(args.labels, args.dataset_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    exit(main())

