# src/train_bert.py
import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Dict, Tuple

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)
from datasets import Dataset

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    classification_report,
)


def _device_info() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def _label_mapping(train_df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[int, str]]:
    labels = sorted(train_df["label"].unique().tolist())
    if len(labels) != 2:
        raise ValueError(f"Expected binary labels, found: {labels}")
    label_to_id = {labels[0]: 0, labels[1]: 1}
    id_to_label = {0: labels[0], 1: labels[1]}
    return label_to_id, id_to_label


def _compute_metrics(eval_pred):
    """
    Trainer passes (logits, labels).
    We compute accuracy/precision/recall/f1.
    ROC/PR handled later in final test evaluation.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}


def _save_eval_plots_and_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob_spam: np.ndarray,
    class_names=("ham", "spam"),
    plots_dir="outputs/plots",
    prefix="bert",
) -> Dict[str, float]:
    _ensure_dirs(plots_dir)

    # Classification report
    print("\nClassification report:\n")
    print(classification_report(y_true, y_pred, target_names=list(class_names), zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(class_names))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    disp.plot(ax=ax)
    plt.title("DistilBERT (Test) - Confusion Matrix")
    cm_path = os.path.join(plots_dir, f"{prefix}_confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches="tight")
    plt.show()
    print("Saved:", cm_path)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob_spam)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"DistilBERT (Test) - ROC Curve (AUC={roc_auc:.4f})")
    roc_path = os.path.join(plots_dir, f"{prefix}_roc_curve.png")
    plt.savefig(roc_path, bbox_inches="tight")
    plt.show()
    print("Saved:", roc_path)

    # PR curve
    prec, rec, _ = precision_recall_curve(y_true, y_prob_spam)
    ap = average_precision_score(y_true, y_prob_spam)
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"DistilBERT (Test) - Precision–Recall Curve (AP={ap:.4f})")
    pr_path = os.path.join(plots_dir, f"{prefix}_pr_curve.png")
    plt.savefig(pr_path, bbox_inches="tight")
    plt.show()
    print("Saved:", pr_path)

    # Scalar metrics
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

    return {
        "accuracy": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "ap": float(ap),
    }


def _plot_training_curves(log_history: list, plots_dir: str, prefix: str = "bert"):
    """
    Creates bert_training_curves.png using Trainer log_history.
    Plots:
    - train loss (loss)
    - eval loss (eval_loss)
    - eval f1 (eval_f1)
    """
    _ensure_dirs(plots_dir)

    steps = []
    train_loss = []
    eval_steps = []
    eval_loss = []
    eval_f1 = []

    for item in log_history:
        if "step" in item:
            if "loss" in item and "eval_loss" not in item:
                steps.append(item["step"])
                train_loss.append(item["loss"])
            if "eval_loss" in item:
                eval_steps.append(item["step"])
                eval_loss.append(item["eval_loss"])
                if "eval_f1" in item:
                    eval_f1.append(item["eval_f1"])

    plt.figure()
    if steps and train_loss:
        plt.plot(steps, train_loss, label="train_loss")
    if eval_steps and eval_loss:
        plt.plot(eval_steps, eval_loss, label="eval_loss")
    if eval_steps and eval_f1:
        # If eval_f1 shorter than eval_steps, align to min length
        m = min(len(eval_steps), len(eval_f1))
        plt.plot(eval_steps[:m], eval_f1[:m], label="eval_f1")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title("DistilBERT Training Curves")
    plt.legend()
    out_path = os.path.join(plots_dir, f"{prefix}_training_curves.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()
    print("Saved:", out_path)


@dataclass
class DistilBertConfig:
    model_name: str = "distilbert-base-uncased"
    max_length: int = 128
    seed: int = 42
    num_train_epochs: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    warmup_ratio: float = 0.0


def run_distilbert_pipeline(
    train_csv: str,
    val_csv: str,
    test_csv: str,
    model_dir: str,
    plots_dir: str,
    preds_dir: str,
    cfg: DistilBertConfig = DistilBertConfig(),
):
    """
    Fine-tune DistilBERT on train, validate each epoch, select best checkpoint by eval_f1.
    Then evaluate on test set and save plots + sample predictions + metadata/logs.
    """
    _ensure_dirs(model_dir, plots_dir, preds_dir)
    set_seed(cfg.seed)

    device = _device_info()
    print(f"Using device: {device}")

    # Load splits
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    label_to_id, id_to_label = _label_mapping(train_df)
    class_names = (id_to_label[0], id_to_label[1])

    # Convert labels to ids
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df["label_id"] = train_df["label"].map(label_to_id).astype(int)
    val_df["label_id"] = val_df["label"].map(label_to_id).astype(int)
    test_df["label_id"] = test_df["label"].map(label_to_id).astype(int)

    # Hugging Face datasets
    train_ds = Dataset.from_pandas(train_df[["text_clean", "label_id"]].rename(columns={"label_id": "labels"}))
    val_ds = Dataset.from_pandas(val_df[["text_clean", "label_id"]].rename(columns={"label_id": "labels"}))
    test_ds = Dataset.from_pandas(test_df[["text_clean", "label_id"]].rename(columns={"label_id": "labels"}))

    # Tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    def tokenize(batch):
        return tokenizer(
            batch["text_clean"],
            truncation=True,
            max_length=cfg.max_length,
        )

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=2,
        id2label=id_to_label,
        label2id=label_to_id,
    )

    # Training args
    output_checkpoint_dir = os.path.join(model_dir, "checkpoints")
    best_checkpoint_dir = os.path.join(model_dir, "best_checkpoint")

    common_args = dict(
        output_dir=output_checkpoint_dir,
        save_strategy="epoch",        
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        num_train_epochs=cfg.num_train_epochs,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        fp16=False,  
        seed=cfg.seed,
    )

    try:
        args = TrainingArguments(evaluation_strategy="epoch", **common_args)
    except TypeError:
        args = TrainingArguments(eval_strategy="epoch", **common_args)

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=_compute_metrics,
        )

    # Train
    trainer.train()

    # Save best model + tokenizer
    trainer.save_model(best_checkpoint_dir)
    tokenizer.save_pretrained(best_checkpoint_dir)
    print(" Saved best checkpoint to:", best_checkpoint_dir)

    # Save logs
    logs_path = os.path.join(model_dir, "training_logs.json")
    with open(logs_path, "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    print("Saved training logs:", logs_path)

    # Plot training curves
    _plot_training_curves(trainer.state.log_history, plots_dir=plots_dir, prefix="bert")

    # Final test predictions (logits)
    test_out = trainer.predict(test_ds)
    logits = test_out.predictions
    y_true = test_out.label_ids
    y_pred = np.argmax(logits, axis=1)

    # Probabilities for spam class (id=1)
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    y_prob_spam = probs[:, 1]

    metrics = _save_eval_plots_and_report(
        y_true=y_true,
        y_pred=y_pred,
        y_prob_spam=y_prob_spam,
        class_names=class_names,
        plots_dir=plots_dir,
        prefix="bert",
    )
    print("DistilBERT test metrics:", metrics)

    # Save sample predictions
    sample = test_df.copy()
    sample["true_label"] = sample["label_id"].map(id_to_label)
    sample["pred_label"] = pd.Series(y_pred).map(id_to_label)
    sample["prob_spam"] = y_prob_spam

    sample_path = os.path.join(preds_dir, "bert_sample_predictions.csv")
    sample[["text", "text_clean", "true_label", "pred_label", "prob_spam"]].sample(
        30, random_state=cfg.seed
    ).to_csv(sample_path, index=False)
    print("Saved sample predictions:", sample_path)

    # Save metadata
    meta = {
        "model": cfg.model_name,
        "max_length": cfg.max_length,
        "seed": cfg.seed,
        "epochs": cfg.num_train_epochs,
        "learning_rate": cfg.learning_rate,
        "batch_size_train": cfg.per_device_train_batch_size,
        "batch_size_eval": cfg.per_device_eval_batch_size,
        "weight_decay": cfg.weight_decay,
        "warmup_ratio": cfg.warmup_ratio,
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "sizes": {
            "train": int(train_df.shape[0]),
            "val": int(val_df.shape[0]),
            "test": int(test_df.shape[0]),
        },
        "metrics_test": metrics,
        "device": device,
        "best_checkpoint_dir": best_checkpoint_dir,
    }

    meta_path = os.path.join(model_dir, "bert_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved metadata:", meta_path)

    return metrics