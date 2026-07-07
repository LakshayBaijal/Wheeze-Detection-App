"""
Phase 5 - Measure how good the model is.

For a wheeze detector the important questions are:
  * Sensitivity (recall) - of the children who really wheeze, how many did we
    catch? (Missing a sick child is the worst kind of error.)
  * Specificity - of the healthy children, how many did we correctly clear?
  * ROC-AUC - one overall score (0.5 = coin flip, 1.0 = perfect).

This file computes those numbers, prints them, and saves two pictures:
a confusion matrix and an ROC curve.
"""

import json

import numpy as np
import matplotlib

matplotlib.use("Agg")  # lets plots save to file without a screen
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from config import OUTPUT_DIR


def compute_metrics(y_true, y_pred, y_score):
    """Return a dictionary of the standard clinical/ML metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0  # = recall for wheeze
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "sensitivity_recall": sensitivity,
        "specificity": specificity,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else float("nan"),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def print_metrics(name, metrics):
    print(f"\n----- {name} : test-set results -----")
    print(f"  Accuracy    : {metrics['accuracy']:.3f}")
    print(f"  Sensitivity : {metrics['sensitivity_recall']:.3f}  (caught wheezers)")
    print(f"  Specificity : {metrics['specificity']:.3f}  (cleared non-wheezers)")
    print(f"  Precision   : {metrics['precision']:.3f}")
    print(f"  F1 score    : {metrics['f1']:.3f}")
    print(f"  ROC-AUC     : {metrics['roc_auc']:.3f}")
    cm = metrics["confusion_matrix"]
    print(f"  Confusion   : TP={cm['tp']} FN={cm['fn']} | FP={cm['fp']} TN={cm['tn']}")


def save_plots(name, y_true, y_score, metrics):
    """Save a confusion-matrix picture and an ROC-curve picture."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe = name.lower().replace(" ", "_")

    # --- Confusion matrix ---
    cm = metrics["confusion_matrix"]
    grid = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(grid, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Pred no-wheeze", "Pred wheeze"])
    ax.set_yticks([0, 1], labels=["True no-wheeze", "True wheeze"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, grid[i, j], ha="center", va="center", fontsize=14)
    ax.set_title(f"{name}\nConfusion matrix")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"confusion_{safe}.png", dpi=120)
    plt.close(fig)

    # --- ROC curve ---
    if len(set(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="grey")
        ax.set_xlabel("False-positive rate (1 - specificity)")
        ax.set_ylabel("True-positive rate (sensitivity)")
        ax.set_title(f"{name}\nROC curve")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"roc_{safe}.png", dpi=120)
        plt.close(fig)


def save_results_json(all_metrics):
    """Save every model's numbers into one results file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n[saved] All results -> {path}")
