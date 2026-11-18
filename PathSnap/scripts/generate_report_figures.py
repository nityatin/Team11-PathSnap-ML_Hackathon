#!/usr/bin/env python3
"""Generate report-ready PathSnap model-comparison figures from saved CSV results."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CV = PROJECT_ROOT / "experiments/results/model_comparison/model_comparison.csv"
DEFAULT_STUDENT = PROJECT_ROOT / "experiments/results/student_testing/student_test_model_comparison.csv"
DEFAULT_OUT = PROJECT_ROOT / "reports/figures/model_comparison"


def require_columns(df: pd.DataFrame, columns: list[str], file_path: Path) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing expected columns in {file_path}: {missing}.\n"
            f"Available columns: {list(df.columns)}"
        )


def nice_model(name: str) -> str:
    return str(name).replace("HistGradientBoosting", "Hist. Gradient Boosting")


def save_bar(df: pd.DataFrame, metric: str, title: str, ylabel: str,
             out_path: Path, percent: bool = False, lower_better: bool = False,
             annotate: bool = True, log_scale: bool = False) -> None:
    plot_df = df[["model", metric]].dropna().copy()
    plot_df["model"] = plot_df["model"].map(nice_model)
    plot_df = plot_df.sort_values(metric, ascending=lower_better)

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    bars = ax.bar(plot_df["model"], plot_df[metric], width=0.65)
    ax.set_title(title, fontsize=15, weight="bold", pad=16)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Machine learning model")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", rotation=18)
    if percent:
        ax.set_ylim(0, max(1.0, float(plot_df[metric].max()) * 1.15))
        labels = [f"{v:.2%}" for v in plot_df[metric]]
    else:
        labels = [f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}" for v in plot_df[metric]]
    if log_scale:
        ax.set_yscale("log")
        ax.set_ylabel(ylabel + " (log scale)")
    if annotate:
        offset = 0.015 * max(plot_df[metric].max(), 1e-9)
        for bar, label in zip(bars, labels):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + offset,
                    label, ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_grouped_top1(cv: pd.DataFrame, student: pd.DataFrame, out_path: Path) -> None:
    merged = cv[["model", "observation_top1_accuracy"]].merge(
        student[["model", "observation_top1_accuracy"]],
        on="model", suffixes=("_cv", "_student"), validate="one_to_one"
    )
    merged["model"] = merged["model"].map(nice_model)
    merged = merged.sort_values("observation_top1_accuracy_cv", ascending=False)
    x = np.arange(len(merged))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 6.4))
    b1 = ax.bar(x - width / 2, merged["observation_top1_accuracy_cv"], width,
                label="5-fold GroupKFold (training data)")
    b2 = ax.bar(x + width / 2, merged["observation_top1_accuracy_student"], width,
                label="Student-test (10 trips)")
    ax.set_title("Observation-Level Top-1 Accuracy: CV vs Student-Test", fontsize=14, weight="bold", pad=16)
    ax.set_ylabel("Top-1 accuracy")
    ax.set_xlabel("Machine learning model")
    ax.set_xticks(x, merged["model"], rotation=18, ha="right")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    for bars in (b1, b2):
        ax.bar_label(bars, labels=[f"{b.get_height():.1%}" for b in bars], padding=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_precision_recall(cv: pd.DataFrame, out_path: Path) -> None:
    data = cv[["model", "candidate_precision", "candidate_recall"]].copy()
    data["model"] = data["model"].map(nice_model)
    data = data.sort_values("candidate_recall", ascending=False)
    x = np.arange(len(data))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 6.4))
    b1 = ax.bar(x - width / 2, data["candidate_precision"], width, label="Precision")
    b2 = ax.bar(x + width / 2, data["candidate_recall"], width, label="Recall")
    ax.set_title("Candidate-Level Precision and Recall (Cross-Validation)", fontsize=14, weight="bold", pad=16)
    ax.set_ylabel("Score")
    ax.set_xlabel("Machine learning model")
    ax.set_xticks(x, data["model"], rotation=18, ha="right")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    for bars in (b1, b2):
        ax.bar_label(bars, labels=[f"{b.get_height():.2f}" for b in bars], padding=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT,
                        help="Path to the PathSnap project directory")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Override output directory")
    args = parser.parse_args()

    root = args.project_root.resolve()
    cv_file = root / "experiments/results/model_comparison/model_comparison.csv"
    student_file = root / "experiments/results/student_testing/student_test_model_comparison.csv"
    output_dir = (args.output_dir or root / "reports/figures/model_comparison").resolve()

    for path in (cv_file, student_file):
        if not path.is_file():
            raise FileNotFoundError(f"Required results file not found: {path}")

    cv = pd.read_csv(cv_file)
    student = pd.read_csv(student_file)
    require_columns(cv, ["model", "candidate_macro_f1", "candidate_precision",
                         "candidate_recall", "candidate_log_loss",
                         "observation_top1_accuracy", "total_training_time_seconds"], cv_file)
    require_columns(student, ["model", "candidate_macro_f1", "candidate_log_loss",
                              "observation_top1_accuracy"], student_file)

    if cv["model"].duplicated().any() or student["model"].duplicated().any():
        raise ValueError("Expected one aggregate row per model in each comparison CSV.")
    if set(cv["model"]) != set(student["model"]):
        raise ValueError("CV and student-test files do not contain the same model names.")

    output_dir.mkdir(parents=True, exist_ok=True)
    save_bar(cv, "observation_top1_accuracy",
             "Observation-Level Top-1 Accuracy (5-Fold GroupKFold)",
             "Top-1 accuracy", output_dir / "01_cv_observation_top1_accuracy.png", percent=True)
    save_bar(cv, "candidate_macro_f1",
             "Candidate-Level Macro F1 (Cross-Validation)",
             "Macro F1", output_dir / "02_cv_candidate_macro_f1.png")
    save_bar(cv, "candidate_log_loss",
             "Candidate-Level Log Loss (Cross-Validation)",
             "Log Loss (lower is better)", output_dir / "03_cv_candidate_log_loss.png", lower_better=True)
    save_bar(cv, "total_training_time_seconds",
             "Model Training Time (Five-Fold Experiment)",
             "Total training time in seconds", output_dir / "04_cv_training_time.png", log_scale=True)
    save_bar(student, "observation_top1_accuracy",
             "Observation-Level Top-1 Accuracy (Student-Test)",
             "Top-1 accuracy", output_dir / "05_student_test_top1_accuracy.png", percent=True)
    save_bar(student, "candidate_macro_f1",
             "Candidate-Level Macro F1 (Student-Test)",
             "Macro F1", output_dir / "06_student_test_macro_f1.png")
    save_grouped_top1(cv, student, output_dir / "07_cv_vs_student_test_top1.png")
    save_precision_recall(cv, output_dir / "08_cv_candidate_precision_recall.png")

    notes = output_dir / "FIGURE_NOTES.md"
    notes.write_text(
        "# PathSnap Model Comparison Figures\n\n"
        "Figures are generated from the saved aggregate CSV result files, not manually entered chart values.\n\n"
        "- Figures 01–04 and 08 use `model_comparison.csv` (five-fold GroupKFold on training trips).\n"
        "- Figures 05–06 use `student_test_model_comparison.csv` (10 student-test trips).\n"
        "- Figure 07 compares observation-level Top-1 from both result files.\n\n"
        "Interpretation: higher Top-1 accuracy and Macro F1 are better; lower Log Loss is better.\n"
        "Training time is the total time reported by the five-fold model comparison experiment.\n"
        "Student-test labels use the project's independent nearest-reference-arc matching and 50 m eligibility filter;\n"
        "they should be described as development metrics, not fully sequence-aligned ground truth.\n",
        encoding="utf-8"
    )
    print(f"Generated 8 figures and FIGURE_NOTES.md in: {output_dir}")
    for path in sorted(output_dir.glob("*.png")):
        print(f"  {path.name} ({path.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
