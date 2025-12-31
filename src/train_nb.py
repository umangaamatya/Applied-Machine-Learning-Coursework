# src/train_nb.py
import os
import json
import pickle
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support

from src.evaluation import evaluate_binary


class MultinomialNBFromScratch:
    """
    Multinomial Naive Bayes using log priors + log likelihoods with Laplace smoothing.
    Works with nonnegative features (counts or TF-IDF).
    """
    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.class_log_prior_ = None         # (2,)
        self.feature_log_prob_ = None        # (2, vocab)

    def fit(self, X, y):
        y = np.asarray(y).astype(int)
        class_counts = np.bincount(y, minlength=2).astype(float)
        self.class_log_prior_ = np.log(class_counts / class_counts.sum())

        X0 = X[y == 0]
        X1 = X[y == 1]

        fc0 = np.asarray(X0.sum(axis=0)).ravel()
        fc1 = np.asarray(X1.sum(axis=0)).ravel()

        sm0 = fc0 + self.alpha
        sm1 = fc1 + self.alpha

        p0 = sm0 / sm0.sum()
        p1 = sm1 / sm1.sum()

        self.feature_log_prob_ = np.vstack([np.log(p0), np.log(p1)])
        return self

    def predict_log_proba(self, X):
        jll0 = X @ self.feature_log_prob_[0].T + self.class_log_prior_[0]
        jll1 = X @ self.feature_log_prob_[1].T + self.class_log_prior_[1]
        jll = np.vstack([np.asarray(jll0).ravel(), np.asarray(jll1).ravel()]).T

        # log-softmax
        max_jll = np.max(jll, axis=1, keepdims=True)
        logsumexp = max_jll + np.log(np.sum(np.exp(jll - max_jll), axis=1, keepdims=True))
        return jll - logsumexp

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        return np.argmax(self.predict_log_proba(X), axis=1)


def run_naive_bayes_pipeline(
    train_csv: str,
    val_csv: str,
    test_csv: str,
    models_dir: str,
    plots_dir: str,
    preds_dir: str,
    seed: int = 42,
    alphas=(0.1, 0.5, 1.0),
    tfidf_min_df=2,
    tfidf_max_df=0.95,
    tfidf_ngram_range=(1, 2),
):
    # Load splits
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    # Determine label names (expects two labels)
    label_names = sorted(train_df["label"].unique().tolist())
    if len(label_names) != 2:
        raise ValueError(f"Expected 2 classes, found {label_names}")

    # Map labels -> {0,1}
    label_to_id = {label_names[0]: 0, label_names[1]: 1}
    id_to_label = {0: label_names[0], 1: label_names[1]}

    y_train = train_df["label"].map(label_to_id).values
    y_val   = val_df["label"].map(label_to_id).values
    y_test  = test_df["label"].map(label_to_id).values

    # TF-IDF (fit on train only for tuning)
    tfidf = TfidfVectorizer(
        ngram_range=tfidf_ngram_range,
        min_df=tfidf_min_df,
        max_df=tfidf_max_df
    )

    X_train = tfidf.fit_transform(train_df["text_clean"])
    X_val   = tfidf.transform(val_df["text_clean"])
    X_test  = tfidf.transform(test_df["text_clean"])

    # Tune alpha on validation by F1
    best_alpha = None
    best_f1 = -1.0

    for a in alphas:
        nb = MultinomialNBFromScratch(alpha=a).fit(X_train, y_train)
        val_pred = nb.predict(X_val)
        _, _, f1, _ = precision_recall_fscore_support(y_val, val_pred, average="binary", zero_division=0)
        print(f"alpha={a} -> val F1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_alpha = a

    print(f"Best alpha = {best_alpha} (val F1={best_f1:.4f})")

    # Final training on train+val
    trainval_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)
    y_trainval = trainval_df["label"].map(label_to_id).values

    tfidf_final = TfidfVectorizer(
        ngram_range=tfidf_ngram_range,
        min_df=tfidf_min_df,
        max_df=tfidf_max_df
    )
    X_trainval = tfidf_final.fit_transform(trainval_df["text_clean"])
    X_test_final = tfidf_final.transform(test_df["text_clean"])

    final_nb = MultinomialNBFromScratch(alpha=best_alpha).fit(X_trainval, y_trainval)

    # Evaluate on test
    test_probs = final_nb.predict_proba(X_test_final)[:, 1]
    test_pred  = final_nb.predict(X_test_final)

    metrics = evaluate_binary(
        y_true=y_test,
        y_pred=test_pred,
        y_prob_spam=test_probs,
        class_names=(label_names[0], label_names[1]),
        title_prefix="Naive Bayes (Final) - Test",
        save_dir=plots_dir,
        save_prefix="nb"
    )

    # Save sample predictions (for screenshots)
    os.makedirs(preds_dir, exist_ok=True)
    sample_df = test_df.copy()
    sample_df["true_label"] = y_test
    sample_df["pred_label"] = test_pred
    sample_df["prob_spam"] = test_probs
    sample_df["true_label"] = sample_df["true_label"].map(id_to_label)
    sample_df["pred_label"] = sample_df["pred_label"].map(id_to_label)

    sample_out = os.path.join(preds_dir, "nb_sample_predictions.csv")
    sample_df[["text", "true_label", "pred_label", "prob_spam"]].sample(
        30, random_state=seed
    ).to_csv(sample_out, index=False)
    print("Saved sample predictions:", sample_out)

    # Interpretability: top spam-indicative features
    feature_names = np.array(tfidf_final.get_feature_names_out())
    logp0 = final_nb.feature_log_prob_[0]
    logp1 = final_nb.feature_log_prob_[1]
    indicative = logp1 - logp0

    topk = 20
    top_spam_idx = np.argsort(indicative)[-topk:][::-1]
    top_spam_terms = feature_names[top_spam_idx]
    top_spam_scores = indicative[top_spam_idx].tolist()

    # Save top features plot (simple)
    import matplotlib.pyplot as plt
    os.makedirs(plots_dir, exist_ok=True)
    plt.figure()
    plt.bar(range(topk), indicative[top_spam_idx])
    plt.xticks(range(topk), top_spam_terms, rotation=90)
    plt.title("Top Spam-Indicative Features (Naive Bayes)")
    plt.tight_layout()
    feat_plot = os.path.join(plots_dir, "nb_top_features.png")
    plt.savefig(feat_plot, bbox_inches="tight")
    print("Saved:", feat_plot)
    plt.show()

    # Save artifacts
    os.makedirs(models_dir, exist_ok=True)
    vec_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
    model_path = os.path.join(models_dir, "nb_model.pkl")
    meta_path = os.path.join(models_dir, "nb_metadata.json")

    with open(vec_path, "wb") as f:
        pickle.dump(tfidf_final, f)

    with open(model_path, "wb") as f:
        pickle.dump(final_nb, f)

    meta = {
        "model": "Multinomial Naive Bayes (from scratch)",
        "best_alpha": best_alpha,
        "alphas_tested": list(alphas),
        "tfidf": {
            "ngram_range": list(tfidf_ngram_range),
            "min_df": tfidf_min_df,
            "max_df": tfidf_max_df
        },
        "label_names": label_names,
        "label_to_id": label_to_id,
        "seed": seed,
        "sizes": {
            "train": int(train_df.shape[0]),
            "val": int(val_df.shape[0]),
            "test": int(test_df.shape[0]),
        },
        "metrics_test": metrics,
        "top_spam_terms": list(top_spam_terms),
        "top_spam_scores": top_spam_scores
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("Saved vectorizer:", vec_path)
    print("Saved model     :", model_path)
    print("Saved metadata  :", meta_path)

    return metrics