# src/evaluation.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, roc_auc_score
)

# Clinical AAMI class order, used for figures, tables, and markdown text
# throughout the notebooks (N, S, V, F, Q).
CLASSES = ['N', 'S', 'V', 'F', 'Q']

# IMPORTANT: sklearn's LabelEncoder.fit(CLASSES) sorts classes ALPHABETICALLY
# internally, regardless of the order of the input list. So le.classes_ is
# actually ['F', 'N', 'Q', 'S', 'V'], and le.transform() assigns:
#   F -> 0, N -> 1, Q -> 2, S -> 3, V -> 4
# This is DIFFERENT from the clinical order in CLASSES above. Any function
# that receives already-encoded integer labels (y_true/y_pred as 0-4) must
# use this alphabetical order for target_names, NOT the clinical CLASSES
# order, or every column will be mislabeled (e.g. the report will print
# "N" over the support/values that actually belong to "F").
CLASSES_ALPHABETICAL = sorted(CLASSES)  # ['F', 'N', 'Q', 'S', 'V']

CLASS_NAMES = {
    'N': 'Normal',
    'S': 'Supraventricular',
    'V': 'Ventricular (PVC)',
    'F': 'Fusion',
    'Q': 'Unknown/Paced',
}


def print_metrics(y_true, y_pred, y_prob=None, model_name='Model', labels=None, target_names=None):
    """
    Print classification report and compute F1-macro and AUC-ROC.
    F1-macro is the primary metric: it weights all classes equally,
    penalizing equally for poor performance on minority classes.

    Defaults assume y_true/y_pred are LabelEncoder-encoded integers
    (0-4), and use CLASSES_ALPHABETICAL (matching le.classes_ order) for
    target_names -- NOT the clinical CLASSES order -- so that each column
    is labeled with the class it actually corresponds to.

    If y_true/y_pred are string labels already (e.g. 'N', 'V', ...),
    pass labels=CLASSES explicitly instead.
    """
    if labels is None:
        labels = np.arange(len(CLASSES_ALPHABETICAL))
    if target_names is None:
        target_names = CLASSES_ALPHABETICAL

    print(f"\n{'=' * 55}")
    print(f" {model_name} — Evaluation Results")
    print(f"{'=' * 55}")
    print(classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0
    ))

    f1 = f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
    print(f" F1-macro : {f1:.4f}")

    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
            print(f" AUC-ROC  : {auc:.4f}")
        except Exception as e:
            print(f" AUC-ROC  : could not compute ({e})")


def plot_confusion_matrix(y_true, y_pred, labels, model_name='Model', save_path=None):
    """
    Plot a normalized confusion matrix.
    Normalization by true label (rows) shows recall per class,
    which is the clinically relevant metric for arrhythmia detection.
    A false negative on class V (missing a PVC) is more dangerous
    than a false positive, so recall is prioritized over precision.

    `labels` must be the actual label values present in y_true/y_pred
    (either encoded integers matching le.classes_ order, or string
    letters if working with decoded labels). CLASS_NAMES maps single
    letters to full names for display; if `labels` are integers, map
    them to letters via CLASSES_ALPHABETICAL before calling this.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    label_names = [CLASS_NAMES.get(l, l) for l in labels]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(
        cm, annot=True, fmt='.2f', cmap='Blues',
        xticklabels=label_names, yticklabels=label_names,
        linewidths=0.5, ax=ax, vmin=0, vmax=1
    )
    ax.set_xlabel('Predicted class', fontsize=11)
    ax.set_ylabel('True class', fontsize=11)
    ax.set_title(f'{model_name} — Confusion matrix (normalized by true label)', fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()


def plot_training_curves(history, save_path=None):
    """
    Plot loss and accuracy curves for CNN training.
    Used in Notebook 05 to diagnose overfitting or underfitting.
    A large gap between train and val curves = overfitting.
    Curves that never converge = learning rate too low or model too small.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history['train_loss'], label='Train loss', color='#3B82F6')
    ax1.plot(history['val_loss'], label='Val loss', color='#EF4444')
    ax1.set_title('Loss curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-entropy loss')
    ax1.legend()

    ax2.plot(history['train_acc'], label='Train acc', color='#3B82F6')
    ax2.plot(history['val_acc'], label='Val acc', color='#EF4444')
    ax2.set_title('Accuracy curves')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.suptitle('CNN Training history', fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()


def compare_models(results_dict, save_path=None):
    """
    Plot a bar chart comparing F1-macro across all models.
    results_dict format: {'Model name': {'F1': 0.85, 'AUC': 0.92}, ...}
    """
    names = list(results_dict.keys())
    f1s = [results_dict[n]['F1'] for n in names]
    colors = ['#3B82F6', '#F59E0B', '#10B981', '#8B5CF6'][:len(names)]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(names, f1s, color=colors, edgecolor='white', width=0.5)
    for bar, val in zip(bars, f1s):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f'{val:.3f}', ha='center', fontsize=10
        )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('F1-macro', fontsize=11)
    ax.set_title(
        'Model comparison — F1-macro (primary metric)\n'
        'Higher = better performance across all AAMI classes equally',
        fontsize=10
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()
