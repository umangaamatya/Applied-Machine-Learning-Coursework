# src/evaluation.py
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

def evaluate_binary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob_spam: np.ndarray,
    class_names=("ham", "spam"),
    title_prefix="",
    save_dir=None,
    save_prefix=None
) -> dict:
    """
    Produces:
    - printed metrics + classification report
    - confusion matrix plot
    - ROC curve plot
    - PR curve plot
    Saves plots if save_dir & save_prefix provided.
    Returns metric dict.
    """
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    report = classification_report(
        y_true, y_pred, target_names=list(class_names), zero_division=0
    )

    print(f"{title_prefix} Accuracy : {acc:.4f}")
    print(f"{title_prefix} Precision: {precision:.4f}")
    print(f"{title_prefix} Recall   : {recall:.4f}")
    print(f"{title_prefix} F1-score : {f1:.4f}")
    print("\nClassification report:\n", report)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(class_names))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    disp.plot(ax=ax)
    plt.title(f"{title_prefix} Confusion Matrix")
    if save_dir and save_prefix:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{save_prefix}_confusion_matrix.png")
        plt.savefig(path, bbox_inches="tight")
        print("Saved:", path)
    plt.show()

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob_spam)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title_prefix} ROC Curve (AUC={roc_auc:.4f})")
    if save_dir and save_prefix:
        path = os.path.join(save_dir, f"{save_prefix}_roc_curve.png")
        plt.savefig(path, bbox_inches="tight")
        print("Saved:", path)
    plt.show()

    # PR Curve
    prec, rec, _ = precision_recall_curve(y_true, y_prob_spam)
    ap = average_precision_score(y_true, y_prob_spam)
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title_prefix} Precision–Recall Curve (AP={ap:.4f})")
    if save_dir and save_prefix:
        path = os.path.join(save_dir, f"{save_prefix}_pr_curve.png")
        plt.savefig(path, bbox_inches="tight")
        print("Saved:", path)
    plt.show()

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "ap": float(ap),
    }