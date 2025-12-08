"""
Evaluation metrics for AIROGS challenge
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    auc,
    precision_recall_curve,
    confusion_matrix,
    cohen_kappa_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
import config
import tensorflow as tf
import argparse
from dataset import AIROGSDataset
import os


def compute_partial_auc(y_true, y_pred, specificity_range=(0.9, 1.0)):
    """
    Compute partial AUC in specified specificity range

    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        specificity_range: Tuple of (min_spec, max_spec)

    Returns:
        Partial AUC value
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)

    # Convert to specificity (specificity = 1 - fpr)
    specificity = 1 - fpr

    # Find indices within specificity range
    min_spec, max_spec = specificity_range

    # Reverse arrays because specificity decreases as fpr increases
    specificity_rev = specificity[::-1]
    tpr_rev = tpr[::-1]

    # Find indices in range
    mask = (specificity_rev >= min_spec) & (specificity_rev <= max_spec)

    if mask.sum() < 2:
        return 0.0

    # Extract relevant portion
    spec_range = specificity_rev[mask]
    tpr_range = tpr_rev[mask]

    # Sort by specificity for proper integration
    sorted_indices = np.argsort(spec_range)
    spec_range = spec_range[sorted_indices]
    tpr_range = tpr_range[sorted_indices]

    # Compute partial AUC using trapezoidal rule
    partial_auc = auc(spec_range, tpr_range)

    # Normalize by range width
    range_width = max_spec - min_spec
    normalized_pauc = partial_auc / range_width if range_width > 0 else 0.0

    return normalized_pauc


def compute_sensitivity_at_specificity(y_true, y_pred, target_specificity=0.95):
    """
    Compute sensitivity at a target specificity level

    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        target_specificity: Target specificity (default 0.95)

    Returns:
        Sensitivity at target specificity
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    specificity = 1 - fpr

    # Find the threshold closest to target specificity
    idx = np.argmin(np.abs(specificity - target_specificity))

    sensitivity = tpr[idx]
    actual_specificity = specificity[idx]
    threshold = thresholds[idx]

    return sensitivity, actual_specificity, threshold


def evaluate_model(model, test_generator, test_df):
    """
    Comprehensive model evaluation

    Args:
        model: Trained Keras model
        test_generator: Test data generator
        test_df: Test dataframe with true labels

    Returns:
        Dictionary of evaluation metrics
    """
    # Get predictions
    print(f"\n🔍 Debug Info:")
    print(f"   Test generator samples: {test_generator.n}")
    print(f"   Test dataframe size: {len(test_df)}")
    print(f"   Batch size: {test_generator.batch_size}")
    print(f"   Number of batches: {len(test_generator)}")

    y_pred_proba = model.predict(test_generator, verbose=1).flatten()
    y_true = test_df["label"].values

    print(f"\n🔍 Prediction Debug:")
    print(f"   y_pred_proba shape: {y_pred_proba.shape}")
    print(f"   y_true shape: {y_true.shape}")
    print(f"   y_pred_proba contains NaN: {np.isnan(y_pred_proba).any()}")
    print(f"   Number of NaN in predictions: {np.isnan(y_pred_proba).sum()}")
    print(f"   y_pred_proba min: {np.nanmin(y_pred_proba) if not np.all(np.isnan(y_pred_proba)) else 'all NaN'}")
    print(f"   y_pred_proba max: {np.nanmax(y_pred_proba) if not np.all(np.isnan(y_pred_proba)) else 'all NaN'}")
    print(f"   y_pred_proba mean: {np.nanmean(y_pred_proba) if not np.all(np.isnan(y_pred_proba)) else 'all NaN'}")

    # Check for size mismatch
    if len(y_pred_proba) != len(y_true):
        raise ValueError(
            f"Size mismatch: predictions ({len(y_pred_proba)}) != labels ({len(y_true)}). "
            f"Generator has {test_generator.n} samples."
        )

    # Handle NaN values
    if np.isnan(y_pred_proba).any():
        nan_count = np.isnan(y_pred_proba).sum()
        print(f"\n⚠️  WARNING: Found {nan_count} NaN values in predictions ({100*nan_count/len(y_pred_proba):.2f}%)!")
        print(f"   This usually means:")
        print(f"   1. Model encountered invalid images")
        print(f"   2. Generator and DataFrame are out of sync")
        print(f"   3. Model output layer has numerical issues")

        # Try to identify which samples have NaN
        nan_indices = np.where(np.isnan(y_pred_proba))[0]
        print(f"\n   NaN at indices: {nan_indices[:10]}..." if len(nan_indices) > 10 else f"\n   NaN at indices: {nan_indices}")

        # Check if ALL non-NaN predictions are also 0.0
        non_nan_preds = y_pred_proba[~np.isnan(y_pred_proba)]
        if len(non_nan_preds) > 0:
            unique_values = np.unique(non_nan_preds)
            print(f"\n   📊 Non-NaN predictions:")
            print(f"      Count: {len(non_nan_preds)}")
            print(f"      Unique values: {len(unique_values)}")
            if len(unique_values) <= 10:
                print(f"      Values: {unique_values}")
            else:
                print(f"      Sample values: {unique_values[:10]}")

            if len(unique_values) == 1 and unique_values[0] == 0.0:
                print(f"\n   🔴 CRITIQUE: Toutes les prédictions non-NaN sont exactement 0.0!")
                print(f"   Cela indique que le modèle n'a pas été entraîné correctement.")
                print(f"\n   💡 ACTIONS RECOMMANDÉES:")
                print(f"      1. Vérifiez que le fichier de modèle est complet:")
                print(f"         ls -lh {test_df.iloc[0]['image_path'].rsplit('/', 2)[0]}/../outputs/models/")
                print(f"      2. Exécutez: python3 inspect_model.py <model_path>")
                print(f"      3. Vérifiez les logs d'entraînement")
                print(f"      4. Si nécessaire, ré-entraînez le modèle")

        raise ValueError(
            f"Predictions contain {nan_count} NaN values. "
            "Cannot compute metrics."
        )

    # Standard metrics
    auc_score = roc_auc_score(y_true, y_pred_proba)

    # Partial AUC (90-100% specificity)
    pauc_score = compute_partial_auc(
        y_true, y_pred_proba, specificity_range=config.PAUC_RANGE
    )

    # Sensitivity at 95% specificity
    sensitivity_95, actual_spec_95, threshold_95 = compute_sensitivity_at_specificity(
        y_true, y_pred_proba, target_specificity=config.SPECIFICITY_THRESHOLD
    )

    # Binary predictions at threshold
    y_pred_binary = (y_pred_proba >= threshold_95).astype(int)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()

    # Additional metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    results = {
        "auc": auc_score,
        "partial_auc": pauc_score,
        "sensitivity_at_95_specificity": sensitivity_95,
        "actual_specificity": actual_spec_95,
        "threshold_at_95_spec": threshold_95,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "specificity": specificity,
        "confusion_matrix": cm,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "y_true": y_true,
        "y_pred_proba": y_pred_proba,
        "y_pred_binary": y_pred_binary,
    }

    return results


def print_evaluation_results(results):
    """Print evaluation results in a formatted way"""
    print("\n" + "=" * 60)
    print("AIROGS CHALLENGE - EVALUATION RESULTS")
    print("=" * 60)

    print("\n📊 CHALLENGE METRICS:")
    print(f"  α - Partial AUC (90-100% spec): {results['partial_auc']:.4f}")
    print(
        f"  β - Sensitivity @ 95% spec:     {results['sensitivity_at_95_specificity']:.4f}"
    )
    print(f"      (actual specificity:         {results['actual_specificity']:.4f})")
    print(f"      (threshold used:             {results['threshold_at_95_spec']:.4f})")

    print("\n📈 STANDARD METRICS:")
    print(f"  AUC-ROC:      {results['auc']:.4f}")
    print(f"  Accuracy:     {results['accuracy']:.4f}")
    print(f"  Precision:    {results['precision']:.4f}")
    print(f"  Recall:       {results['recall']:.4f}")
    print(f"  F1-Score:     {results['f1_score']:.4f}")
    print(f"  Specificity:  {results['specificity']:.4f}")

    print("\n🎯 CONFUSION MATRIX:")
    print(f"  True Positives:  {results['true_positives']}")
    print(f"  False Positives: {results['false_positives']}")
    print(f"  True Negatives:  {results['true_negatives']}")
    print(f"  False Negatives: {results['false_negatives']}")

    print("=" * 60 + "\n")


def plot_roc_curve(results, save_path=None, show=False):
    """Plot ROC curve with partial AUC highlighted"""
    fpr, tpr, _ = roc_curve(results["y_true"], results["y_pred_proba"])
    specificity = 1 - fpr

    plt.figure(figsize=(10, 8))

    # Plot full ROC curve
    plt.plot(
        fpr, tpr, "b-", linewidth=2, label=f'ROC Curve (AUC = {results["auc"]:.4f})'
    )

    # Highlight partial AUC region (90-100% specificity = 0-10% FPR)
    mask = fpr <= 0.1
    plt.fill_between(
        fpr[mask],
        0,
        tpr[mask],
        alpha=0.3,
        color="green",
        label=f'Partial AUC (pAUC = {results["partial_auc"]:.4f})',
    )

    # Plot sensitivity at 95% specificity point
    sens_95 = results["sensitivity_at_95_specificity"]
    fpr_95 = 1 - results["actual_specificity"]
    plt.plot(
        fpr_95,
        sens_95,
        "ro",
        markersize=10,
        label=f"Sensitivity @ 95% Spec = {sens_95:.4f}",
    )

    # Diagonal reference line
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")

    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    plt.title("ROC Curve - AIROGS Glaucoma Detection", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"ROC curve saved to {save_path}")

    plt.tight_layout()

    if not show:
        plt.close()

    return plt.gcf() if show else None


def plot_confusion_matrix(results, save_path=None, show=False):
    """Plot confusion matrix"""
    cm = results["confusion_matrix"]

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        xticklabels=["NRG", "RG"],
        yticklabels=["NRG", "RG"],
    )
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title(
        "Confusion Matrix - AIROGS Glaucoma Detection", fontsize=14, fontweight="bold"
    )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")

    plt.tight_layout()

    if not show:
        plt.close()

    return plt.gcf() if show else None


def plot_prediction_distribution(results, save_path=None, show=False):
    """Plot distribution of predicted probabilities"""
    y_true = results["y_true"]
    y_pred_proba = results["y_pred_proba"]

    plt.figure(figsize=(10, 6))

    # Plot histograms for each class
    plt.hist(
        y_pred_proba[y_true == 0],
        bins=50,
        alpha=0.6,
        label="NRG (No Glaucoma)",
        color="blue",
    )
    plt.hist(
        y_pred_proba[y_true == 1],
        bins=50,
        alpha=0.6,
        label="RG (Glaucoma)",
        color="red",
    )

    # Add threshold line
    threshold = results["threshold_at_95_spec"]
    plt.axvline(
        threshold,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Threshold @ 95% Spec = {threshold:.4f}",
    )

    plt.xlabel("Predicted Probability", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title("Distribution of Predicted Probabilities", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Prediction distribution saved to {save_path}")

    plt.tight_layout()

    if not show:
        plt.close()

    return plt.gcf() if show else None

def main():
    parser = argparse.ArgumentParser(description="Évaluation du modèle AIROGS")
    parser.add_argument("--model-path", type=str, required=True, help="Chemin du modèle .h5")
    parser.add_argument("--data-dir", type=str, required=True, help="Répertoire des images")
    parser.add_argument("--labels-csv", type=str, required=True, help="CSV des labels")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--output-dir", type=str, default="evaluation_results", help="Répertoire de sortie pour les figures et résultats")
    parser.add_argument("--show-plots", action="store_true", help="Afficher les graphiques (pour exécution locale)")
    args = parser.parse_args()

    # Créer le répertoire de sortie
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n📁 Répertoire de sortie : {args.output_dir}")

    # Chargement du modèle
    print(f"\n🔄 Chargement du modèle depuis {args.model_path}...")
    model = tf.keras.models.load_model(args.model_path, compile=False)
    print("✅ Modèle chargé avec succès")

    # Préparation du dataset
    print(f"\n📂 Chargement des données depuis {args.data_dir}...")
    dataset = AIROGSDataset(labels_csv=args.labels_csv, images_dir=args.data_dir)

    dataset.load_data()

    # AJOUT : Filtrer uniquement les images qui existent réellement
    dataset.df = dataset.df[dataset.df['image_path'].apply(os.path.exists)]
    print(f"✅ Images valides trouvées : {len(dataset.df)}")

    if len(dataset.df) == 0:
        raise ValueError(f"❌ Aucune image trouvée dans {args.data_dir}. Vérifiez le chemin.")

    print(f"   RG (Glaucoma): {(dataset.df['label'] == 1).sum()}")
    print(f"   NRG (No Glaucoma): {(dataset.df['label'] == 0).sum()}")

    # Vérifier plus en détail les fichiers
    print(f"\n🔍 Vérification des fichiers image...")
    missing_files = []
    corrupted_files = []
    valid_files = []

    for idx, row in dataset.df.iterrows():
        img_path = row['image_path']
        if not os.path.exists(img_path):
            missing_files.append(img_path)
        elif os.path.getsize(img_path) == 0:
            corrupted_files.append(img_path)
        else:
            valid_files.append(img_path)

    print(f"   ✅ Fichiers valides: {len(valid_files)}")
    if missing_files:
        print(f"   ⚠️  Fichiers manquants: {len(missing_files)}")
        print(f"      Exemples: {missing_files[:3]}")
    if corrupted_files:
        print(f"   ⚠️  Fichiers corrompus (taille 0): {len(corrupted_files)}")
        print(f"      Exemples: {corrupted_files[:3]}")

    # Filtrer à nouveau pour être sûr (au cas où)
    if missing_files or corrupted_files:
        print(f"\n🔧 Nettoyage des fichiers invalides...")
        dataset.df = dataset.df[dataset.df['image_path'].isin(valid_files)]
        print(f"   Dataset final: {len(dataset.df)} images")

    # Utiliser toutes les données comme test (pas de split)
    _, _, test_df = dataset.split_data(
        train_split=0.0, val_split=0.0, test_split=1.0, random_seed=config.RANDOM_SEED
    )
    _, _, test_gen = dataset.create_generators(
        batch_size=args.batch_size, augment=False
    )

    # Vérifier la cohérence générateur / DataFrame
    print(f"\n🔍 Vérification cohérence générateur/DataFrame:")
    print(f"   Test DataFrame: {len(test_df)} échantillons")
    print(f"   Test Generator: {test_gen.n} échantillons")
    print(f"   Batch size: {test_gen.batch_size}")
    print(f"   Nombre de batches: {len(test_gen)}")

    if len(test_df) != test_gen.n:
        raise ValueError(
            f"❌ Incohérence détectée!\n"
            f"   DataFrame: {len(test_df)} échantillons\n"
            f"   Generator: {test_gen.n} échantillons\n"
            f"   Ces deux nombres doivent être identiques!"
        )

    print(f"   ✅ Cohérence OK: {len(test_df)} échantillons")


    # Évaluation
    print("\n🔎 Évaluation du modèle sur le jeu de test...")
    results = evaluate_model(model, test_gen, test_df)
    print_evaluation_results(results)

    # Sauvegarder les résultats textuels
    results_file = os.path.join(args.output_dir, "evaluation_results.txt")
    with open(results_file, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("AIROGS CHALLENGE - EVALUATION RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write("CHALLENGE METRICS:\n")
        f.write(f"  α - Partial AUC (90-100% spec): {results['partial_auc']:.4f}\n")
        f.write(f"  β - Sensitivity @ 95% spec:     {results['sensitivity_at_95_specificity']:.4f}\n")
        f.write(f"      (actual specificity:         {results['actual_specificity']:.4f})\n")
        f.write(f"      (threshold used:             {results['threshold_at_95_spec']:.4f})\n\n")
        f.write("STANDARD METRICS:\n")
        f.write(f"  AUC-ROC:      {results['auc']:.4f}\n")
        f.write(f"  Accuracy:     {results['accuracy']:.4f}\n")
        f.write(f"  Precision:    {results['precision']:.4f}\n")
        f.write(f"  Recall:       {results['recall']:.4f}\n")
        f.write(f"  F1-Score:     {results['f1_score']:.4f}\n")
        f.write(f"  Specificity:  {results['specificity']:.4f}\n\n")
        f.write("CONFUSION MATRIX:\n")
        f.write(f"  True Positives:  {results['true_positives']}\n")
        f.write(f"  False Positives: {results['false_positives']}\n")
        f.write(f"  True Negatives:  {results['true_negatives']}\n")
        f.write(f"  False Negatives: {results['false_negatives']}\n")
        f.write("=" * 60 + "\n")
    print(f"✅ Résultats sauvegardés dans {results_file}")

    # Sauvegarder les figures
    print("\n📊 Génération et sauvegarde des graphiques...")
    roc_path = os.path.join(args.output_dir, "roc_curve.png")
    plot_roc_curve(results, save_path=roc_path, show=args.show_plots)

    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plot_confusion_matrix(results, save_path=cm_path, show=args.show_plots)

    dist_path = os.path.join(args.output_dir, "prediction_distribution.png")
    plot_prediction_distribution(results, save_path=dist_path, show=args.show_plots)

    print("\n✅ Évaluation terminée avec succès!")

if __name__ == "__main__":
    # "python evaluation.py --model-path='firstModel.h5' --data-dir='5' --labels-csv='train_labels.csv' --batch-size=32"
    main()