# src/evaluation.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, roc_auc_score
)

CLASS_NAMES = {
    'N': 'Normal',
    'S': 'Supraventricular',
    'V': 'Ventricular (PVC)',
    'F': 'Fusion',
    'Q': 'Unknown/Paced'
}


def print_metrics(y_true, y_pred, y_prob=None, model_name='Model'):
    """
    Print classification report and compute F1-macro and AUC-ROC.
    F1-macro is the primary metric: it weights all classes equally,
    penalizing equally for poor performance on minority classes.
    """
    print(f"\n{'='*55}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'='*55}")
    print(classification_report(y_true, y_pred, zero_division=0))

    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print(f"  F1-macro : {f1:.4f}")

    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob, multi_class='ovr',
                                average='weighted')
            print(f"  AUC-ROC  : {auc:.4f}")
        except Exception as e:
            print(f"  AUC-ROC  : could not compute ({e})")


def plot_confusion_matrix(y_true, y_pred, labels, model_name='Model', save_path=None):
    """
    Plot a normalized confusion matrix.
    Normalization by true label (rows) shows recall per class,
    which is the clinically relevant metric for arrhythmia detection.
    A false negative on class V (missing a PVC) is more dangerous
    than a false positive, so recall is prioritized over precision.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    label_names = [CLASS_NAMES.get(l, l) for l in labels]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names,
                linewidths=0.5, ax=ax, vmin=0, vmax=1)
    ax.set_xlabel('Predicted class', fontsize=11)
    ax.set_ylabel('True class', fontsize=11)
    ax.set_title(f'{model_name} — Confusion matrix (normalized by true label)\n'
                 f'Diagonal = recall per class', fontsize=10)
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
    ax1.plot(history['val_loss'],   label='Val loss',   color='#EF4444')
    ax1.set_title('Loss curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-entropy loss')
    ax1.legend()

    ax2.plot(history['train_acc'], label='Train acc', color='#3B82F6')
    ax2.plot(history['val_acc'],   label='Val acc',   color='#EF4444')
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
    names  = list(results_dict.keys())
    f1s    = [results_dict[n]['F1'] for n in names]
    colors = ['#3B82F6', '#F59E0B', '#10B981', '#8B5CF6'][:len(names)]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(names, f1s, color=colors, edgecolor='white', width=0.5)
    for bar, val in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('F1-macro', fontsize=11)
    ax.set_title('Model comparison — F1-macro (primary metric)\n'
                 'Higher = better performance across all AAMI classes equally', fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()