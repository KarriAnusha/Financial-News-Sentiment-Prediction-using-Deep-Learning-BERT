"""
Models Module
=============
Deep learning model architectures for financial sentiment classification.
Includes: Simple RNN, LSTM, GRU (baseline models) and BERT fine-tuning wrapper.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data_loading import LABEL_MAP, LABEL_TO_ID


class SimpleRNNClassifier(nn.Module):
    """
    Embedding + Simple RNN for 3-class sentiment classification.
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 num_layers=1, dropout=0.3, pad_idx=0):
        super(SimpleRNNClassifier, self).__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        output, hidden = self.rnn(embedded)
        lengths = (x != self.pad_idx).sum(dim=1).clamp(min=1) - 1
        hidden = output[torch.arange(output.size(0), device=x.device), lengths]
        hidden = self.dropout(hidden)
        logits = self.fc(hidden)  # (batch_size, num_classes)
        return logits


class LSTMClassifier(nn.Module):
    """
    Embedding + LSTM for 3-class sentiment classification.
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 num_layers=1, dropout=0.3, bidirectional=False, pad_idx=0):
        super(LSTMClassifier, self).__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0,
                            bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        direction_factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * direction_factor, num_classes)

    def forward(self, x):
        # x: (batch_size, seq_len)
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        lengths = (x != self.pad_idx).sum(dim=1).clamp(min=1) - 1
        hidden = output[torch.arange(output.size(0), device=x.device), lengths]
        hidden = self.dropout(hidden)
        logits = self.fc(hidden)
        return logits


class GRUClassifier(nn.Module):
    """
    Embedding + GRU for 3-class sentiment classification.
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 num_layers=1, dropout=0.3, bidirectional=False, pad_idx=0):
        super(GRUClassifier, self).__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0,
                          bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        direction_factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * direction_factor, num_classes)

    def forward(self, x):
        # x: (batch_size, seq_len)
        embedded = self.embedding(x)
        output, hidden = self.gru(embedded)
        lengths = (x != self.pad_idx).sum(dim=1).clamp(min=1) - 1
        hidden = output[torch.arange(output.size(0), device=x.device), lengths]
        hidden = self.dropout(hidden)
        logits = self.fc(hidden)
        return logits


def get_rnn_model(model_type, vocab_size, embed_dim=128, hidden_dim=128,
                  num_classes=3, num_layers=1, dropout=0.3, pad_idx=0):
    """
    Factory function to create an RNN-based model.

    Args:
        model_type (str): One of 'simple_rnn', 'lstm', 'gru'.
        vocab_size (int): Size of the vocabulary.
        embed_dim (int): Embedding dimension.
        hidden_dim (int): Hidden state dimension.
        num_classes (int): Number of output classes.
        num_layers (int): Number of RNN layers.
        dropout (float): Dropout rate.
        pad_idx (int): Padding index.

    Returns:
        nn.Module: The requested model.
    """
    if model_type == "simple_rnn":
        return SimpleRNNClassifier(vocab_size, embed_dim, hidden_dim,
                                   num_classes, num_layers, dropout, pad_idx)
    elif model_type == "lstm":
        return LSTMClassifier(vocab_size, embed_dim, hidden_dim,
                              num_classes, num_layers, dropout,
                              bidirectional=False, pad_idx=pad_idx)
    elif model_type == "gru":
        return GRUClassifier(vocab_size, embed_dim, hidden_dim,
                             num_classes, num_layers, dropout,
                             bidirectional=False, pad_idx=pad_idx)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. "
                         f"Choose from 'simple_rnn', 'lstm', 'gru'.")


def load_bert_model(model_name="ahmedrachid/FinancialBERT-Sentiment-Analysis",
                    num_labels=3):
    """
    Load a pre-trained BERT/FinBERT model for sequence classification.

    Args:
        model_name (str): Hugging Face model name.
        num_labels (int): Number of classification labels.

    Returns:
        model: The pre-trained model.
        tokenizer: The corresponding tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, ignore_mismatched_sizes=True
    )
    model.config.id2label = LABEL_MAP
    model.config.label2id = LABEL_TO_ID
    return model, tokenizer
