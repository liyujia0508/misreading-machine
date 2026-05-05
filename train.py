"""
Emotion Recognition Model Training
===================================
Affective Computing Homework

Dataset : dair-ai/emotion  (HuggingFace)
          16 000 train / 2 000 val / 2 000 test
Labels  : sadness, joy, love, anger, fear, surprise
Model   : TF-IDF (1-3 grams, 100 k features) + Logistic Regression
Expected: ~93 % accuracy on test set
"""

import os
import sys
import re
import string
import time

import numpy as np
import pandas as pd


# ── Emotion meta ─────────────────────────────────────────────────────────────
EMOTION_LABELS = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']


# ── Helpers ───────────────────────────────────────────────────────────────────
def print_banner():
    print()
    print("=" * 62)
    print("   EMOTION RECOGNITION  ·  Affective Computing Homework")
    print("=" * 62)
    print()


def preprocess_text(text: str) -> str:
    """Lowercase, strip URLs / mentions / punctuation."""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ── Steps ─────────────────────────────────────────────────────────────────────
def load_data():
    print("[1/4] Loading dataset from HuggingFace …")

    try:
        from datasets import load_dataset
    except ImportError:
        print("  ✗  'datasets' package not installed.")
        print("     Run: pip install -r requirements.txt")
        sys.exit(1)

    try:
        dataset = load_dataset("dair-ai/emotion")
    except Exception as exc:
        print(f"  ✗  Download failed: {exc}")
        print("     Check your internet connection and retry.")
        sys.exit(1)

    train_df = dataset['train'].to_pandas()
    val_df   = dataset['validation'].to_pandas()
    test_df  = dataset['test'].to_pandas()

    print(f"  Training samples  : {len(train_df):>6,}")
    print(f"  Validation samples: {len(val_df):>6,}")
    print(f"  Test samples      : {len(test_df):>6,}")
    print()
    return train_df, val_df, test_df


def preprocess_data(train_df, val_df, test_df):
    print("[2/4] Preprocessing text …")
    for df in [train_df, val_df, test_df]:
        df['text_clean'] = df['text'].apply(preprocess_text)
    print("  Done.\n")
    return train_df, val_df, test_df


def train_model(train_df, val_df):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    print("[3/4] Training TF-IDF + Logistic Regression …")

    # Combine train + val for final model
    X = pd.concat([train_df['text_clean'], val_df['text_clean']], ignore_index=True)
    y = pd.concat([train_df['label'],      val_df['label']],      ignore_index=True)
    print(f"  Samples : {len(X):,}  |  Features: up to 100 000 (1-3 grams)")

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=100_000,
            min_df=2,
            sublinear_tf=True,
            strip_accents='unicode',
        )),
        ('clf', LogisticRegression(
            C=5.0,
            max_iter=1000,
            solver='lbfgs',
            n_jobs=-1,
        )),
    ])

    t0 = time.time()
    pipeline.fit(X, y)
    print(f"  Completed in {time.time() - t0:.1f}s\n")
    return pipeline


def evaluate_model(pipeline, test_df):
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix,
    )

    print("[4/4] Evaluating on test set …")

    y_pred = pipeline.predict(test_df['text_clean'])
    y_true = test_df['label']

    acc = accuracy_score(y_true, y_pred)
    print(f"\n  Test Accuracy : {acc:.4f}  ({acc * 100:.2f} %)\n")

    print("  Classification Report:")
    print("  " + "-" * 56)
    report = classification_report(y_true, y_pred, target_names=EMOTION_LABELS, digits=3)
    for line in report.split('\n'):
        print("  " + line)

    cm = confusion_matrix(y_true, y_pred)
    print("  Confusion Matrix (rows = true, cols = predicted):")
    header = "  {:12s}".format("") + "  ".join(f"{e[:6]:>7}" for e in EMOTION_LABELS)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {EMOTION_LABELS[i]:<12}" + "  ".join(f"{v:>7}" for v in row))
    print()

    # Optional: save confusion matrix image
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS, ax=ax,
        )
        ax.set_title('Confusion Matrix – Emotion Classifier', fontsize=13, pad=12)
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        plt.tight_layout()
        os.makedirs('model', exist_ok=True)
        fig.savefig('model/confusion_matrix.png', dpi=150)
        plt.close(fig)
        print("  Confusion matrix image saved → model/confusion_matrix.png\n")
    except ImportError:
        pass  # matplotlib / seaborn not required

    return acc


def save_model(pipeline):
    import joblib

    os.makedirs('model', exist_ok=True)
    path = 'model/emotion_classifier.pkl'
    joblib.dump(pipeline, path)
    size_mb = os.path.getsize(path) / 1_048_576
    print(f"  Model saved → {path}  ({size_mb:.1f} MB)\n")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    print_banner()
    train_df, val_df, test_df = load_data()
    train_df, val_df, test_df = preprocess_data(train_df, val_df, test_df)
    pipeline = train_model(train_df, val_df)
    acc = evaluate_model(pipeline, test_df)
    save_model(pipeline)

    print("=" * 62)
    print(f"  Training complete!  Accuracy: {acc * 100:.2f} %")
    print("  Start the demo:  python run.py")
    print("=" * 62)
    print()


if __name__ == '__main__':
    main()
