"""
AIROGS Challenge Report Generator
Loads all results and generates comprehensive figures
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings("ignore")

# Configure plotting
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


class AirogsReportGenerator:
    """Comprehensive report generator for AIROGS challenge results"""

    def __init__(self, output_dir="plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.models = []

    def load_all_results(self):
        """Load all model results from evaluation folders"""

        model_configs = [
            {
                "name": "Baseline",
                "folder": "../evaluation_result_b0_FullDS",
                "has_tta": True,
                "config_file": None,
            },
            {
                "name": "Improved v1 (Focal Loss)",
                "folder": "../evaluation_improveBaselien_v1",
                "has_tta": False,
                "config_file": None,
            },
            {
                "name": "Improved v3 (Enhanced DA)",
                "folder": "../evaluation_boDA_v3",
                "has_tta": True,
                "config_file": "airogs_improved_v3_efficientnet-b0_20251218_114527_config.json",
            },
            {
                "name": "Improved v4 (Advanced)",
                "folder": "../evaluation_boDA_v4",
                "has_tta": True,
                "config_file": "airogs_v4_advanced_efficientnet-b0_20251217_214958_config.json",
            },
            {
                "name": "Improved v5 (Final)",
                "folder": "../evaluation_boDA_v5",
                "has_tta": True,
                "config_file": "airogs_v5_final_efficientnet-b0_20251218_113458_config.json",
            },
        ]

        for config in model_configs:
            try:
                model_data = self._load_single_model(config)
                self.models.append(model_data)
                print(f"✅ Loaded: {config['name']}")
            except Exception as e:
                print(f"⚠️  Could not load {config['name']}: {e}")

        print(f"\n📊 Total models loaded: {len(self.models)}")

    def _load_single_model(self, config: Dict) -> Dict:
        """Load results for a single model"""
        folder = Path(config["folder"])

        # Load evaluation results
        eval_results = self._parse_evaluation_results(folder / "evaluation_results.txt")

        # Load training history
        history = self._load_history(folder)

        # Load configuration if available
        model_config = None
        if config["config_file"]:
            config_path = folder / config["config_file"]
            if config_path.exists():
                with open(config_path, "r") as f:
                    model_config = json.load(f)

        # Load TTA results if available
        tta_results = None
        if config["has_tta"]:
            tta_results = self._load_tta_results(folder)

        return {
            "name": config["name"],
            "folder": str(folder),
            "config": model_config,
            **eval_results,
            "history": history,
            "tta_results": tta_results,
            "roc_curve_path": str(folder / "roc_curve.png"),
            "confusion_matrix_path": str(folder / "confusion_matrix.png"),
            "prediction_dist_path": str(folder / "prediction_distribution.png"),
        }

    def _parse_evaluation_results(self, filepath: Path) -> Dict:
        """Parse evaluation_results.txt file"""
        results = {}

        with open(filepath, "r") as f:
            content = f.read()

        # Parse metrics
        metrics_map = {
            "partial_auc": r"Partial AUC.*?:\s*([\d.]+)",
            "sensitivity_95": r"Sensitivity @ 95% spec.*?:\s*([\d.]+)",
            "auc_roc": r"AUC-ROC.*?:\s*([\d.]+)",
            "accuracy": r"Accuracy.*?:\s*([\d.]+)",
            "precision": r"Precision.*?:\s*([\d.]+)",
            "recall": r"Recall.*?:\s*([\d.]+)",
            "f1_score": r"F1-Score.*?:\s*([\d.]+)",
            "specificity": r"Specificity.*?:\s*([\d.]+)",
            "tp": r"True Positives.*?:\s*(\d+)",
            "fp": r"False Positives.*?:\s*(\d+)",
            "tn": r"True Negatives.*?:\s*(\d+)",
            "fn": r"False Negatives.*?:\s*(\d+)",
        }

        import re

        for key, pattern in metrics_map.items():
            match = re.search(pattern, content)
            if match:
                value = match.group(1)
                results[key] = (
                    int(value) if key in ["tp", "fp", "tn", "fn"] else float(value)
                )

        return results

    def _load_history(self, folder: Path) -> pd.DataFrame:
        """Load training history from CSV or JSON"""
        # Try CSV files first
        csv_files = list(folder.glob("*training*.csv"))
        if csv_files:
            return pd.read_csv(csv_files[0])

        # Fallback to history.json
        json_file = folder / "history.json"
        if json_file.exists():
            with open(json_file, "r") as f:
                history_dict = json.load(f)
            return pd.DataFrame(history_dict)

        return None

    def _load_tta_results(self, folder: Path) -> Dict:
        """Load TTA results from .out files or JSON"""
        # Try .out files
        tta_files = list(folder.glob("*tta*.out"))
        if tta_files:
            return self._parse_tta_from_out(tta_files[0])

        # Try JSON files
        json_files = list(folder.glob("*tta*.json"))
        if json_files:
            with open(json_files[0], "r") as f:
                tta_data = json.load(f)
                # Handle nested structure for baseline
                if "challenge_metrics" in tta_data:
                    return {
                        "partial_auc": tta_data["challenge_metrics"]["partial_auc"],
                        "sensitivity_95": tta_data["challenge_metrics"][
                            "sensitivity_at_95_spec"
                        ],
                        "auc_roc": tta_data["standard_metrics"]["auc_roc"],
                        "accuracy": tta_data["standard_metrics"]["accuracy"],
                        "precision": tta_data["standard_metrics"]["precision"],
                        "recall": tta_data["standard_metrics"]["recall"],
                        "f1_score": tta_data["standard_metrics"]["f1"],
                        "specificity": tta_data["standard_metrics"]["specificity"],
                        "tp": tta_data["confusion_matrix"]["tp"],
                        "fp": tta_data["confusion_matrix"]["fp"],
                        "tn": tta_data["confusion_matrix"]["tn"],
                        "fn": tta_data["confusion_matrix"]["fn"],
                    }
                return tta_data

        return None

    def _parse_tta_from_out(self, filepath: Path) -> Dict:
        """Parse TTA results from .out file"""
        with open(filepath, "r") as f:
            content = f.read()

        import re

        # Extract metrics from the IMPROVED EVALUATION RESULTS section
        results = {}

        patterns = {
            "partial_auc": r"Partial AUC.*?:\s*([\d.]+)",
            "sensitivity_95": r"Sensitivity @ 95% spec.*?:\s*([\d.]+)",
            "auc_roc": r"AUC-ROC.*?:\s*([\d.]+)",
            "accuracy": r"Accuracy.*?:\s*([\d.]+)",
            "precision": r"Precision.*?:\s*([\d.]+)",
            "recall": r"Recall.*?:\s*([\d.]+)",
            "f1_score": r"F1-Score.*?:\s*([\d.]+)",
            "specificity": r"Specificity.*?:\s*([\d.]+)",
            "tp": r"True Positives.*?:\s*(\d+)",
            "fp": r"False Positives.*?:\s*(\d+)",
            "tn": r"True Negatives.*?:\s*(\d+)",
            "fn": r"False Negatives.*?:\s*(\d+)",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                value = match.group(1)
                results[key] = (
                    int(value) if key in ["tp", "fp", "tn", "fn"] else float(value)
                )

        return results

    def generate_comparison_table(self) -> pd.DataFrame:
        """Generate comprehensive comparison table"""
        data = []

        for model in self.models:
            row = {
                "Model": model["name"],
                "pAUC": model.get("partial_auc", np.nan),
                "Sens@95%": model.get("sensitivity_95", np.nan),
                "AUC-ROC": model.get("auc_roc", np.nan),
                "Accuracy": model.get("accuracy", np.nan),
                "Precision": model.get("precision", np.nan),
                "Recall": model.get("recall", np.nan),
                "F1-Score": model.get("f1_score", np.nan),
                "TP": model.get("tp", np.nan),
                "FN": model.get("fn", np.nan),
                "FP": model.get("fp", np.nan),
            }
            data.append(row)

            # Add TTA results if available
            if model.get("tta_results"):
                tta = model["tta_results"]
                row_tta = {
                    "Model": f"{model['name']} + TTA",
                    "pAUC": tta.get("partial_auc", np.nan),
                    "Sens@95%": tta.get("sensitivity_95", np.nan),
                    "AUC-ROC": tta.get("auc_roc", np.nan),
                    "Accuracy": tta.get("accuracy", np.nan),
                    "Precision": tta.get("precision", np.nan),
                    "Recall": tta.get("recall", np.nan),
                    "F1-Score": tta.get("f1_score", np.nan),
                    "TP": tta.get("tp", np.nan),
                    "FN": tta.get("fn", np.nan),
                    "FP": tta.get("fp", np.nan),
                }
                data.append(row_tta)

        df = pd.DataFrame(data)

        # Save to CSV
        df.to_csv(self.output_dir / "model_comparison.csv", index=False)

        return df

    def plot_performance_evolution(self):
        """Plot performance metrics evolution across models"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Prepare data
        base_models = [m for m in self.models]
        model_names = [m["name"].replace("Improved ", "v") for m in base_models]

        pauc_values = [m.get("partial_auc", 0) for m in base_models]
        sens_values = [m.get("sensitivity_95", 0) for m in base_models]
        auc_values = [m.get("auc_roc", 0) for m in base_models]
        f1_values = [m.get("f1_score", 0) for m in base_models]

        # Add TTA values
        tta_indices = []
        tta_pauc = []
        tta_sens = []
        tta_auc = []
        tta_f1 = []

        for i, m in enumerate(base_models):
            if m.get("tta_results"):
                tta_indices.append(i)
                tta = m["tta_results"]
                tta_pauc.append(tta.get("partial_auc", 0))
                tta_sens.append(tta.get("sensitivity_95", 0))
                tta_auc.append(tta.get("auc_roc", 0))
                tta_f1.append(tta.get("f1_score", 0))

        colors = plt.cm.viridis(np.linspace(0, 1, len(base_models)))
        x = np.arange(len(base_models))

        # Plot 1: pAUC
        ax = axes[0, 0]
        bars = ax.bar(
            x, pauc_values, color=colors, edgecolor="black", linewidth=1.5, label="Base"
        )
        if tta_indices:
            ax.scatter(
                tta_indices,
                tta_pauc,
                color="red",
                s=200,
                marker="*",
                label="+ TTA",
                zorder=10,
                edgecolor="black",
                linewidth=2,
            )
        ax.axhline(
            y=0.70,
            color="r",
            linestyle="--",
            linewidth=2,
            alpha=0.5,
            label="Target (0.70)",
        )
        ax.set_xlabel("Model Version", fontsize=13, fontweight="bold")
        ax.set_ylabel("Partial AUC (90-100% spec)", fontsize=13, fontweight="bold")
        ax.set_title(
            "Challenge Metric α: Partial AUC Evolution", fontsize=15, fontweight="bold"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha="right")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")

        for i, (bar, val) in enumerate(zip(bars, pauc_values)):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        # Plot 2: Sensitivity
        ax = axes[0, 1]
        bars = ax.bar(
            x, sens_values, color=colors, edgecolor="black", linewidth=1.5, label="Base"
        )
        if tta_indices:
            ax.scatter(
                tta_indices,
                tta_sens,
                color="red",
                s=200,
                marker="*",
                label="+ TTA",
                zorder=10,
                edgecolor="black",
                linewidth=2,
            )
        ax.axhline(
            y=0.75,
            color="r",
            linestyle="--",
            linewidth=2,
            alpha=0.5,
            label="Target (0.75)",
        )
        ax.set_xlabel("Model Version", fontsize=13, fontweight="bold")
        ax.set_ylabel("Sensitivity @ 95% Specificity", fontsize=13, fontweight="bold")
        ax.set_title(
            "Challenge Metric β: Sensitivity Evolution", fontsize=15, fontweight="bold"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha="right")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")

        for i, (bar, val) in enumerate(zip(bars, sens_values)):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        # Plot 3: AUC-ROC
        ax = axes[1, 0]
        bars = ax.bar(
            x, auc_values, color=colors, edgecolor="black", linewidth=1.5, label="Base"
        )
        if tta_indices:
            ax.scatter(
                tta_indices,
                tta_auc,
                color="red",
                s=200,
                marker="*",
                label="+ TTA",
                zorder=10,
                edgecolor="black",
                linewidth=2,
            )
        ax.set_xlabel("Model Version", fontsize=13, fontweight="bold")
        ax.set_ylabel("AUC-ROC", fontsize=13, fontweight="bold")
        ax.set_title(
            "Standard Metric: AUC-ROC Evolution", fontsize=15, fontweight="bold"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha="right")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim([0.90, 0.98])

        for i, (bar, val) in enumerate(zip(bars, auc_values)):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.002,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        # Plot 4: F1-Score
        ax = axes[1, 1]
        bars = ax.bar(
            x, f1_values, color=colors, edgecolor="black", linewidth=1.5, label="Base"
        )
        if tta_indices:
            ax.scatter(
                tta_indices,
                tta_f1,
                color="red",
                s=200,
                marker="*",
                label="+ TTA",
                zorder=10,
                edgecolor="black",
                linewidth=2,
            )
        ax.set_xlabel("Model Version", fontsize=13, fontweight="bold")
        ax.set_ylabel("F1-Score", fontsize=13, fontweight="bold")
        ax.set_title(
            "Standard Metric: F1-Score Evolution", fontsize=15, fontweight="bold"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha="right")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")

        for i, (bar, val) in enumerate(zip(bars, f1_values)):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "performance_evolution.png", dpi=300, bbox_inches="tight"
        )
        print(f"✅ Saved: performance_evolution.png")

    def plot_confusion_analysis(self):
        """Plot confusion matrix analysis"""
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        # Prepare data
        base_models = [m for m in self.models]
        model_names = [m["name"].replace("Improved ", "v") for m in base_models]

        tp_values = [m.get("tp", 0) for m in base_models]
        fn_values = [m.get("fn", 0) for m in base_models]
        fp_values = [m.get("fp", 0) for m in base_models]

        # Add TTA data
        for i, m in enumerate(base_models):
            if m.get("tta_results"):
                tta = m["tta_results"]
                model_names.insert(i + 1, f"{m['name'].replace('Improved ', 'v')} +TTA")
                tp_values.insert(i + 1, tta.get("tp", 0))
                fn_values.insert(i + 1, tta.get("fn", 0))
                fp_values.insert(i + 1, tta.get("fp", 0))

        x = np.arange(len(model_names))
        colors_extended = plt.cm.viridis(np.linspace(0, 1, len(model_names)))

        # Plot 1: TP vs FN
        ax = axes[0]
        width = 0.35
        bars1 = ax.bar(
            x - width / 2,
            tp_values,
            width,
            label="True Positives",
            color="#2ecc71",
            edgecolor="black",
            linewidth=1.5,
        )
        bars2 = ax.bar(
            x + width / 2,
            fn_values,
            width,
            label="False Negatives",
            color="#e74c3c",
            edgecolor="black",
            linewidth=1.5,
        )
        ax.set_xlabel("Model Version", fontsize=12, fontweight="bold")
        ax.set_ylabel("Count (out of 329 RG cases)", fontsize=12, fontweight="bold")
        ax.set_title(
            "Positive Case Detection: TP vs FN", fontsize=14, fontweight="bold"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=9)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")

        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{int(height)}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

        # Plot 2: False Positives
        ax = axes[1]
        bars = ax.bar(
            x, fp_values, color=colors_extended, edgecolor="black", linewidth=1.5
        )
        ax.set_xlabel("Model Version", fontsize=12, fontweight="bold")
        ax.set_ylabel("False Positive Count", fontsize=12, fontweight="bold")
        ax.set_title(
            "False Positives (out of 11,113 NRG)", fontsize=14, fontweight="bold"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

        for i, (bar, val) in enumerate(zip(bars, fp_values)):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(val)}\n({val/11113*100:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

        # Plot 3: Sensitivity trend
        ax = axes[2]
        sensitivities = [
            tp / (tp + fn) if (tp + fn) > 0 else 0
            for tp, fn in zip(tp_values, fn_values)
        ]
        ax.plot(
            x,
            sensitivities,
            marker="o",
            linewidth=2.5,
            markersize=10,
            color="#3498db",
            markerfacecolor="#e74c3c",
            markeredgewidth=2,
            markeredgecolor="black",
        )
        ax.axhline(
            y=0.75,
            color="r",
            linestyle="--",
            linewidth=2,
            alpha=0.5,
            label="Target (0.75)",
        )
        ax.set_xlabel("Model Version", fontsize=12, fontweight="bold")
        ax.set_ylabel("Sensitivity (Recall)", fontsize=12, fontweight="bold")
        ax.set_title("Sensitivity Progression", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=9)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.6, 0.9])

        for i, (xi, sens) in enumerate(zip(x, sensitivities)):
            ax.text(
                xi,
                sens + 0.015,
                f"{sens:.2%}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "confusion_analysis.png", dpi=300, bbox_inches="tight"
        )
        print(f"✅ Saved: confusion_analysis.png")

    def plot_training_curves(self):
        """Plot training curves for all models"""
        models_with_history = [m for m in self.models if m.get("history") is not None]

        if not models_with_history:
            print("⚠️  No training history available")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        for model in models_with_history:
            history = model["history"]
            name = model["name"].replace("Improved ", "v")

            # Plot Loss (with reasonable y-limits)
            if "loss" in history.columns:
                axes[0, 0].plot(history["loss"], label=f"{name} (train)", linewidth=2)
            if "val_loss" in history.columns:
                # Filter out extreme values for better visualization
                val_loss_clean = history["val_loss"].replace([np.inf, -np.inf], np.nan)
                val_loss_clean = val_loss_clean[
                    val_loss_clean < 3
                ]  # Cap at 3 for visibility
                axes[0, 0].plot(
                    val_loss_clean, label=f"{name} (val)", linewidth=2, linestyle="--"
                )

            # Plot AUC
            auc_col = (
                "auc"
                if "auc" in history.columns
                else "val_auc" if "val_auc" in history.columns else None
            )
            if auc_col:
                axes[0, 1].plot(history[auc_col], label=name, linewidth=2)

            # Plot Precision
            if "precision" in history.columns:
                axes[1, 0].plot(
                    history["precision"], label=f"{name} (train)", linewidth=2
                )
            if "val_precision" in history.columns:
                axes[1, 0].plot(
                    history["val_precision"],
                    label=f"{name} (val)",
                    linewidth=2,
                    linestyle="--",
                )

            # Plot Recall
            if "recall" in history.columns:
                axes[1, 1].plot(history["recall"], label=f"{name} (train)", linewidth=2)
            if "val_recall" in history.columns:
                axes[1, 1].plot(
                    history["val_recall"],
                    label=f"{name} (val)",
                    linewidth=2,
                    linestyle="--",
                )

        # Configure axes
        axes[0, 0].set_xlabel("Epoch", fontsize=12, fontweight="bold")
        axes[0, 0].set_ylabel("Loss", fontsize=12, fontweight="bold")
        axes[0, 0].set_title(
            "Training & Validation Loss", fontsize=14, fontweight="bold"
        )
        axes[0, 0].legend(fontsize=9, loc="best")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(bottom=0, top=0.7)  # Reasonable range for loss

        axes[0, 1].set_xlabel("Epoch", fontsize=12, fontweight="bold")
        axes[0, 1].set_ylabel("AUC", fontsize=12, fontweight="bold")
        axes[0, 1].set_title(
            "Validation AUC Progression", fontsize=14, fontweight="bold"
        )
        axes[0, 1].legend(fontsize=9, loc="best")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0.7, 1.0])  # Typical AUC range

        axes[1, 0].set_xlabel("Epoch", fontsize=12, fontweight="bold")
        axes[1, 0].set_ylabel("Precision", fontsize=12, fontweight="bold")
        axes[1, 0].set_title(
            "Training & Validation Precision", fontsize=14, fontweight="bold"
        )
        axes[1, 0].legend(fontsize=9, loc="best")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 1.0])

        axes[1, 1].set_xlabel("Epoch", fontsize=12, fontweight="bold")
        axes[1, 1].set_ylabel("Recall", fontsize=12, fontweight="bold")
        axes[1, 1].set_title(
            "Training & Validation Recall", fontsize=14, fontweight="bold"
        )
        axes[1, 1].legend(fontsize=9, loc="best")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim([0, 1.0])

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "training_curves.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
        print(f"✅ Saved: training_curves.png")

    def copy_existing_figures(self):
        """Copy ROC curves, confusion matrices, and prediction distributions"""
        import shutil

        for model in self.models:
            folder = Path(model["folder"])
            name_clean = (
                model["name"].replace(" ", "_").replace("(", "").replace(")", "")
            )

            # Copy ROC curve
            roc_src = folder / "roc_curve.png"
            if roc_src.exists():
                roc_dst = self.output_dir / f"roc_{name_clean}.png"
                shutil.copy(roc_src, roc_dst)

            # Copy confusion matrix
            cm_src = folder / "confusion_matrix.png"
            if cm_src.exists():
                cm_dst = self.output_dir / f"cm_{name_clean}.png"
                shutil.copy(cm_src, cm_dst)

            # Copy prediction distribution
            pd_src = folder / "prediction_distribution.png"
            if pd_src.exists():
                pd_dst = self.output_dir / f"pred_dist_{name_clean}.png"
                shutil.copy(pd_src, pd_dst)

        print(f"✅ Copied existing figures to {self.output_dir}")

    def generate_all_figures(self, verbose=True):
        """Generate all figures for the report"""
        if verbose:
            print("\n" + "=" * 70)
            print("GENERATING ALL REPORT FIGURES")
            print("=" * 70)

        if verbose:
            print("\n📊 Generating comparison table...")
        df = self.generate_comparison_table()
        if verbose:
            print(f"✅ Comparison table saved")

        if verbose:
            print("\n📈 Generating performance evolution plots...")
        self.plot_performance_evolution()

        if verbose:
            print("\n🔍 Generating confusion analysis...")
        self.plot_confusion_analysis()

        if verbose:
            print("\n📉 Generating training curves...")
        self.plot_training_curves()

        if verbose:
            print("\n📁 Copying existing figures...")
        self.copy_existing_figures()

        if verbose:
            print("\n" + "=" * 70)
            print("✅ ALL FIGURES GENERATED SUCCESSFULLY!")
            print(f"📁 Output directory: {self.output_dir.absolute()}")
            print("=" * 70)

        return df


# Main execution
if __name__ == "__main__":
    generator = AirogsReportGenerator(output_dir="plots")
    generator.load_all_results()
    df = generator.generate_all_figures()

    print("\n📊 Model Comparison Summary:")
    print(df.to_string(index=False))
