"""
Utility functions for analyzing and comparing AIROGS challenge models
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10


def load_model_results(result_folder: str) -> Dict:
    """
    Load evaluation results and training history from a result folder.

    Args:
        result_folder: Path to folder containing evaluation_results.txt and history.json

    Returns:
        Dictionary with parsed results
    """
    result_path = Path(result_folder)
    results = {"folder": result_folder, "name": result_path.name}

    # Load evaluation results
    eval_file = result_path / "evaluation_results.txt"
    if eval_file.exists():
        with open(eval_file, "r") as f:
            content = f.read()

            # Parse metrics
            for line in content.split("\n"):
                if "Partial AUC" in line:
                    results["partial_auc"] = float(line.split(":")[1].strip())
                elif "Sensitivity @ 95% spec:" in line:
                    results["sensitivity_95"] = float(line.split(":")[1].strip())
                elif "actual specificity:" in line:
                    results["actual_specificity"] = float(
                        line.split(":")[1].strip().rstrip(")")
                    )
                elif "threshold used:" in line:
                    results["threshold"] = float(line.split(":")[1].strip().rstrip(")"))
                elif "AUC-ROC:" in line:
                    results["auc_roc"] = float(line.split(":")[1].strip())
                elif "Accuracy:" in line:
                    results["accuracy"] = float(line.split(":")[1].strip())
                elif "Precision:" in line:
                    results["precision"] = float(line.split(":")[1].strip())
                elif "Recall:" in line:
                    results["recall"] = float(line.split(":")[1].strip())
                elif "F1-Score:" in line:
                    results["f1_score"] = float(line.split(":")[1].strip())
                elif "Specificity:" in line and "actual" not in line:
                    results["specificity"] = float(line.split(":")[1].strip())
                elif "True Positives:" in line:
                    results["tp"] = int(line.split(":")[1].strip())
                elif "False Positives:" in line:
                    results["fp"] = int(line.split(":")[1].strip())
                elif "True Negatives:" in line:
                    results["tn"] = int(line.split(":")[1].strip())
                elif "False Negatives:" in line:
                    results["fn"] = int(line.split(":")[1].strip())

    # Load training history
    history_file = result_path / "history.json"
    if history_file.exists():
        with open(history_file, "r") as f:
            results["history"] = json.load(f)

    # Load training CSV if available
    training_csv = result_path / "training.csv"
    if training_csv.exists():
        results["training_df"] = pd.read_csv(training_csv)

    # Try to find model file
    for ext in [".keras", ".h5"]:
        model_files = list(result_path.glob(f"*{ext}"))
        if model_files:
            results["model_file"] = str(model_files[0])
            break

    return results


def compare_models_table(models: List[Dict]) -> pd.DataFrame:
    """
    Create a comparison table of multiple models.

    Args:
        models: List of model result dictionaries

    Returns:
        DataFrame with comparison metrics
    """
    metrics = []

    for model in models:
        metric_dict = {
            "Model": model["name"],
            "pAUC (90-100%)": model.get("partial_auc", np.nan),
            "Sens@95%spec": model.get("sensitivity_95", np.nan),
            "AUC-ROC": model.get("auc_roc", np.nan),
            "Accuracy": model.get("accuracy", np.nan),
            "Precision": model.get("precision", np.nan),
            "Recall": model.get("recall", np.nan),
            "F1-Score": model.get("f1_score", np.nan),
            "Specificity": model.get("specificity", np.nan),
        }
        metrics.append(metric_dict)

    df = pd.DataFrame(metrics)
    return df


def plot_training_curves(models: List[Dict], figsize=(15, 10)):
    """
    Plot training curves for multiple models.

    Args:
        models: List of model result dictionaries
        figsize: Figure size tuple
    """
    n_models = len(models)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    metrics_to_plot = [
        ("loss", "Loss"),
        ("auc", "AUC"),
        ("accuracy", "Accuracy"),
        ("pr_auc", "PR AUC"),
    ]

    for idx, (metric_key, metric_name) in enumerate(metrics_to_plot):
        ax = axes[idx]

        for model in models:
            if "history" not in model:
                continue

            history = model["history"]

            if metric_key in history:
                epochs = range(1, len(history[metric_key]) + 1)
                ax.plot(
                    epochs,
                    history[metric_key],
                    label=f"{model['name']} (train)",
                    linewidth=2,
                )

                val_key = f"val_{metric_key}"
                if val_key in history:
                    # Filter out NaN values
                    val_data = np.array(history[val_key])
                    valid_mask = ~np.isnan(val_data)
                    valid_epochs = np.array(epochs)[valid_mask]
                    valid_data = val_data[valid_mask]

                    ax.plot(
                        valid_epochs,
                        valid_data,
                        label=f"{model['name']} (val)",
                        linestyle="--",
                        linewidth=2,
                    )

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(f"Training {metric_name}", fontsize=14, fontweight="bold")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_confusion_matrices(models: List[Dict], figsize=(15, 5)):
    """
    Plot confusion matrices for multiple models side by side.

    Args:
        models: List of model result dictionaries
        figsize: Figure size tuple
    """
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=figsize)

    if n_models == 1:
        axes = [axes]

    for idx, model in enumerate(models):
        ax = axes[idx]

        if all(k in model for k in ["tp", "fp", "tn", "fn"]):
            cm = np.array([[model["tn"], model["fp"]], [model["fn"], model["tp"]]])

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["NRG", "RG"],
                yticklabels=["NRG", "RG"],
                ax=ax,
                cbar=True,
            )

            ax.set_xlabel("Predicted Label", fontsize=11)
            ax.set_ylabel("True Label", fontsize=11)
            ax.set_title(
                f"{model['name']}\nConfusion Matrix", fontsize=12, fontweight="bold"
            )

    plt.tight_layout()
    return fig


def plot_metrics_comparison(models: List[Dict], figsize=(12, 6)):
    """
    Create bar plots comparing key metrics across models.

    Args:
        models: List of model result dictionaries
        figsize: Figure size tuple
    """
    df = compare_models_table(models)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Challenge metrics
    challenge_metrics = ["pAUC (90-100%)", "Sens@95%spec"]
    df[challenge_metrics].plot(kind="bar", ax=axes[0], rot=45)
    axes[0].set_title("Challenge Metrics Comparison", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Score", fontsize=12)
    axes[0].set_xlabel("Model", fontsize=12)
    axes[0].set_xticklabels(df["Model"], rotation=45, ha="right")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])

    # Standard metrics
    standard_metrics = ["AUC-ROC", "Accuracy", "F1-Score"]
    df[standard_metrics].plot(kind="bar", ax=axes[1], rot=45)
    axes[1].set_title("Standard Metrics Comparison", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Score", fontsize=12)
    axes[1].set_xlabel("Model", fontsize=12)
    axes[1].set_xticklabels(df["Model"], rotation=45, ha="right")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])

    plt.tight_layout()
    return fig


def plot_learning_rate_schedule(models: List[Dict], figsize=(10, 6)):
    """
    Plot learning rate schedules for models.

    Args:
        models: List of model result dictionaries
        figsize: Figure size tuple
    """
    fig, ax = plt.subplots(figsize=figsize)

    for model in models:
        if "history" in model and "learning_rate" in model["history"]:
            epochs = range(1, len(model["history"]["learning_rate"]) + 1)
            ax.plot(
                epochs,
                model["history"]["learning_rate"],
                marker="o",
                label=model["name"],
                linewidth=2,
            )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Learning Rate", fontsize=12)
    ax.set_title("Learning Rate Schedule", fontsize=14, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    return fig


def generate_report_summary(models: List[Dict]) -> str:
    """
    Generate a text summary of model comparison.

    Args:
        models: List of model result dictionaries

    Returns:
        Formatted text summary
    """
    summary = "=" * 80 + "\n"
    summary += "AIROGS CHALLENGE - MODEL COMPARISON SUMMARY\n"
    summary += "=" * 80 + "\n\n"

    # Find best model for each metric
    metrics = {
        "pAUC (90-100%)": "partial_auc",
        "Sensitivity @ 95% spec": "sensitivity_95",
        "AUC-ROC": "auc_roc",
        "Accuracy": "accuracy",
        "F1-Score": "f1_score",
    }

    summary += "BEST MODELS BY METRIC:\n"
    summary += "-" * 80 + "\n"

    for metric_name, metric_key in metrics.items():
        values = [(m["name"], m.get(metric_key, 0)) for m in models if metric_key in m]
        if values:
            best_model, best_value = max(values, key=lambda x: x[1])
            summary += f"  {metric_name:30s}: {best_model:30s} ({best_value:.4f})\n"

    summary += "\n" + "=" * 80 + "\n\n"

    # Detailed comparison
    summary += "DETAILED COMPARISON:\n"
    summary += "-" * 80 + "\n"

    for model in models:
        summary += f"\n{model['name']}:\n"
        summary += f"  Challenge Metrics:\n"
        summary += (
            f"    Partial AUC (90-100% spec): {model.get('partial_auc', 'N/A')}\n"
        )
        summary += (
            f"    Sensitivity @ 95% spec:     {model.get('sensitivity_95', 'N/A')}\n"
        )
        summary += f"  Standard Metrics:\n"
        summary += f"    AUC-ROC:      {model.get('auc_roc', 'N/A')}\n"
        summary += f"    Accuracy:     {model.get('accuracy', 'N/A')}\n"
        summary += f"    Precision:    {model.get('precision', 'N/A')}\n"
        summary += f"    Recall:       {model.get('recall', 'N/A')}\n"
        summary += f"    F1-Score:     {model.get('f1_score', 'N/A')}\n"
        summary += f"    Specificity:  {model.get('specificity', 'N/A')}\n"

        if "history" in model:
            n_epochs = len(model["history"].get("loss", []))
            summary += f"  Training:\n"
            summary += f"    Epochs: {n_epochs}\n"

    summary += "\n" + "=" * 80 + "\n"

    return summary


def save_all_plots(models: List[Dict], output_dir: str):
    """
    Generate and save all comparison plots.

    Args:
        models: List of model result dictionaries
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Generating plots in {output_dir}...")

    # Training curves
    fig = plot_training_curves(models)
    fig.savefig(output_path / "training_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ Training curves saved")

    # Confusion matrices
    fig = plot_confusion_matrices(models)
    fig.savefig(output_path / "confusion_matrices.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ Confusion matrices saved")

    # Metrics comparison
    fig = plot_metrics_comparison(models)
    fig.savefig(output_path / "metrics_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ Metrics comparison saved")

    # Learning rate schedule
    fig = plot_learning_rate_schedule(models)
    fig.savefig(
        output_path / "learning_rate_schedule.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)
    print("  ✓ Learning rate schedule saved")

    # Save comparison table
    df = compare_models_table(models)
    df.to_csv(output_path / "model_comparison.csv", index=False)
    print("  ✓ Comparison table saved")

    # Save text summary
    summary = generate_report_summary(models)
    with open(output_path / "summary.txt", "w") as f:
        f.write(summary)
    print("  ✓ Text summary saved")

    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    # Example usage
    print("Analysis utilities loaded successfully!")
    print("\nUsage example:")
    print("  models = [")
    print("      load_model_results('evaluation_result_b0_FullDS'),")
    print("      load_model_results('evaluation_improveBaselien_v1')")
    print("  ]")
    print("  save_all_plots(models, 'report/plots')")
