
"""
Evaluation Module
=================
Functions for computing metrics (accuracy, macro F1, precision, recall),
generating confusion matrices, and plotting training curves.
"""

# Allow running as a script: add project root to sys.path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, classification_report,
                             confusion_matrix)

from src.data_loading import LABEL_MAP


def evaluate_rnn_model(model, data_loader, device="cpu"):
    """
    Run inference on a DataLoader and collect predictions and true labels.

    Args:
        model (nn.Module): Trained RNN model.
        data_loader (DataLoader): Validation DataLoader.
        device (str): Device.

    Returns:
        all_preds (list): Predicted labels.
        all_labels (list): True labels.
        all_probs (list): Prediction probabilities.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for texts, labels in data_loader:
            texts, labels = texts.to(device), labels.to(device)
            logits = model(texts)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return all_preds, all_labels, all_probs


def evaluate_bert_model(model, tokenizer, texts, labels, max_len=128,
                        batch_size=16, device="cpu"):
    """
    Run inference with a BERT model and collect predictions.

    Args:
        model: Fine-tuned BERT model.
        tokenizer: Corresponding tokenizer.
        texts (list): Text samples.
        labels (list): True labels.
        max_len (int): Max token length.
        batch_size (int): Batch size.
        device (str): Device.

    Returns:
        all_preds (list): Predicted labels.
        all_labels (list): True labels.
        all_probs (list): Prediction probabilities.
    """
    from torch.utils.data import DataLoader, TensorDataset

    model.eval()
    model = model.to(device)

    encodings = tokenizer(texts, truncation=True, padding=True,
                          max_length=max_len, return_tensors="pt")
    dataset = TensorDataset(
        encodings["input_ids"],
        encodings["attention_mask"],
        torch.tensor(labels, dtype=torch.long)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_labels_out = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            input_ids, attention_mask, batch_labels = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels_out.extend(batch_labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return all_preds, all_labels_out, all_probs


def compute_metrics(y_true, y_pred):
    """
    Compute classification metrics.

    Args:
        y_true (list): True labels.
        y_pred (list): Predicted labels.

    Returns:
        metrics (dict): Dictionary with accuracy, macro F1, and per-class metrics.
    """
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)

    target_names = [LABEL_MAP[i] for i in sorted(LABEL_MAP.keys())]
    report = classification_report(y_true, y_pred, target_names=target_names,
                                   zero_division=0)

    metrics = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "class_precision": {LABEL_MAP[i]: float(precision[i])
                            for i in range(len(precision))},
        "class_recall": {LABEL_MAP[i]: float(recall[i])
                         for i in range(len(recall))},
        "classification_report": report
    }

    print(f"\nOverall Accuracy: {accuracy:.4f}")
    print(f"Macro F1-Score:   {macro_f1:.4f}")
    print(f"\nClassification Report:\n{report}")

    return metrics


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """
    Plot a confusion matrix heatmap.

    Args:
        y_true (list): True labels.
        y_pred (list): Predicted labels.
        title (str): Plot title.

    Returns:
        fig: Matplotlib figure.
    """
    target_names = [LABEL_MAP[i] for i in sorted(LABEL_MAP.keys())]
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_training_history(history, model_name="Model"):
    """
    Plot training and validation loss/accuracy curves.

    Args:
        history (dict): Training history with train_loss, val_loss,
                        train_acc, val_acc.
        model_name (str): Name for the plot title.

    Returns:
        fig: Matplotlib figure.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], "r-o", label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{model_name} - Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy plot
    axes[1].plot(epochs, history["train_acc"], "b-o", label="Train Accuracy")
    axes[1].plot(epochs, history["val_acc"], "r-o", label="Val Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{model_name} - Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    return fig


def compare_models(results_dict):
    """
    Create a comparison table and bar chart of model performances.

    Args:
        results_dict (dict): {model_name: metrics_dict} where metrics_dict
                             contains 'accuracy' and 'macro_f1'.

    Returns:
        fig: Matplotlib figure with comparison bar chart.
    """
    model_names = list(results_dict.keys())
    accuracies = [results_dict[m]["accuracy"] for m in model_names]
    f1_scores = [results_dict[m]["macro_f1"] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, accuracies, width, label="Accuracy",
                   color="steelblue")
    bars2 = ax.bar(x + width / 2, f1_scores, width, label="Macro F1",
                   color="coral")

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f"{height:.3f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f"{height:.3f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    return fig
