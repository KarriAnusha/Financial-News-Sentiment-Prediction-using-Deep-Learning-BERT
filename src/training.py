"""
Training Module
===============
Functions for training deep learning models with early stopping
and BERT fine-tuning.
"""

import time
import copy

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from transformers import get_linear_schedule_with_warmup


def train_rnn_model(model, train_loader, val_loader, num_epochs=15,
                    learning_rate=1e-3, patience=3, device="cpu", class_weights=None):
    """
    Train an RNN-based model with early stopping.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): Training DataLoader.
        val_loader (DataLoader): Validation DataLoader.
        num_epochs (int): Maximum number of epochs.
        learning_rate (float): Learning rate.
        patience (int): Early stopping patience (epochs without improvement).
        device (str): Device to train on ('cpu' or 'cuda').
        class_weights (torch.Tensor): Weights for each class.

    Returns:
        model (nn.Module): Best model (by validation loss).
        history (dict): Training history with train/val loss and accuracy.
    """
    model = model.to(device)
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    best_model_state = None
    epochs_no_improve = 0

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": []
    }

    for epoch in range(num_epochs):
        start_time = time.time()

        # --- Training Phase ---
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0

        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(texts)
            loss = criterion(logits, labels)
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item() * texts.size(0)
            preds = logits.argmax(dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += texts.size(0)

        avg_train_loss = total_train_loss / total_train
        train_acc = correct_train / total_train

        # --- Validation Phase ---
        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for texts, labels in val_loader:
                texts, labels = texts.to(device), labels.to(device)
                logits = model(texts)
                loss = criterion(logits, labels)

                total_val_loss += loss.item() * texts.size(0)
                preds = logits.argmax(dim=1)
                correct_val += (preds == labels).sum().item()
                total_val += texts.size(0)

        avg_val_loss = total_val_loss / total_val
        val_acc = correct_val / total_val

        elapsed = time.time() - start_time
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"Time: {elapsed:.1f}s")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1} "
                      f"(no improvement for {patience} epochs)")
                break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model with val loss: {best_val_loss:.4f}")

    return model, history


def train_bert_model(model, tokenizer, train_texts, train_labels,
                     val_texts, val_labels, num_epochs=3,
                     learning_rate=2e-5, batch_size=16, max_len=128,
                     patience=2, device="cpu"):
    """
    Fine-tune a BERT model for sequence classification.

    Args:
        model: Hugging Face model for sequence classification.
        tokenizer: Corresponding tokenizer.
        train_texts (list): Training text samples.
        train_labels (list): Training labels.
        val_texts (list): Validation text samples.
        val_labels (list): Validation labels.
        num_epochs (int): Number of fine-tuning epochs.
        learning_rate (float): Learning rate.
        batch_size (int): Batch size.
        max_len (int): Max token length for BERT.
        patience (int): Early stopping patience.
        device (str): Device to train on.

    Returns:
        model: Fine-tuned model.
        history (dict): Training history.
    """
    model = model.to(device)

    # Tokenize all texts
    train_encodings = tokenizer(train_texts, truncation=True, padding=True,
                                max_length=max_len, return_tensors="pt")
    val_encodings = tokenizer(val_texts, truncation=True, padding=True,
                              max_length=max_len, return_tensors="pt")

    train_dataset = TensorDataset(
        train_encodings["input_ids"],
        train_encodings["attention_mask"],
        torch.tensor(train_labels, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        val_encodings["input_ids"],
        val_encodings["attention_mask"],
        torch.tensor(val_labels, dtype=torch.long)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    best_val_loss = float("inf")
    best_model_state = None
    epochs_no_improve = 0

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": []
    }

    for epoch in range(num_epochs):
        start_time = time.time()

        # --- Training ---
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0

        num_batches = len(train_loader)
        for batch_idx, batch in enumerate(train_loader):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item() * input_ids.size(0)
            preds = logits.argmax(dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += input_ids.size(0)

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == num_batches:
                print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{num_batches} | "
                      f"Loss: {loss.item():.4f}", flush=True)

        avg_train_loss = total_train_loss / total_train
        train_acc = correct_train / total_train

        # --- Validation ---
        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
                outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                labels=labels)
                loss = outputs.loss
                logits = outputs.logits

                total_val_loss += loss.item() * input_ids.size(0)
                preds = logits.argmax(dim=1)
                correct_val += (preds == labels).sum().item()
                total_val += input_ids.size(0)

        avg_val_loss = total_val_loss / total_val
        val_acc = correct_val / total_val

        elapsed = time.time() - start_time
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"Time: {elapsed:.1f}s")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best BERT model with val loss: {best_val_loss:.4f}")

    return model, history
