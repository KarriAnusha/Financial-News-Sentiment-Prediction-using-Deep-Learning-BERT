"""
Preprocessing Module
====================
Functions for cleaning financial tweets, building vocabulary,
tokenizing, and preparing PyTorch tensors for training.
"""

import re
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


# Special tokens
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX = 0
UNK_IDX = 1


def clean_tweet(text):
    """
    Clean a financial tweet while preserving tickers and financial terms.

    Steps:
        - Remove URLs
        - Remove @mentions
        - Remove extra whitespace
        - Keep $TICKER symbols and financial terms
        - Lowercase the text

    Args:
        text (str): Raw tweet text.

    Returns:
        str: Cleaned tweet text.
    """
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)
    # Remove @mentions
    text = re.sub(r"@\w+", "", text)
    # Remove HTML entities
    text = re.sub(r"&\w+;", " ", text)
    # Keep $TICKER patterns but normalize spacing
    text = re.sub(r"\s+", " ", text)
    # Lowercase
    text = text.lower().strip()
    return text


def tokenize(text):
    """
    Simple whitespace + punctuation tokenizer.
    Splits on whitespace and separates punctuation while keeping $tickers intact.

    Args:
        text (str): Cleaned text.

    Returns:
        list: List of tokens.
    """
    # Split on whitespace, then separate punctuation except $ attached to words
    tokens = re.findall(r"\$\w+|\w+|[^\w\s]", text)
    return tokens


def build_vocabulary(texts, min_freq=2):
    """
    Build a vocabulary from a list of texts.

    Args:
        texts (list): List of cleaned text strings.
        min_freq (int): Minimum frequency for a token to be included.

    Returns:
        word2idx (dict): Token to index mapping.
        idx2word (dict): Index to token mapping.
        vocab_size (int): Size of the vocabulary.
    """
    counter = Counter()
    for text in texts:
        tokens = tokenize(text)
        counter.update(tokens)

    # Start with special tokens
    word2idx = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    idx = 2

    for word, freq in counter.most_common():
        if freq >= min_freq:
            word2idx[word] = idx
            idx += 1

    idx2word = {v: k for k, v in word2idx.items()}
    vocab_size = len(word2idx)

    print(f"Vocabulary size: {vocab_size} (min_freq={min_freq})")
    return word2idx, idx2word, vocab_size


def encode_texts(texts, word2idx, max_len=64):
    """
    Convert texts to padded/truncated sequences of token indices.

    Args:
        texts (list): List of cleaned text strings.
        word2idx (dict): Vocabulary mapping.
        max_len (int): Maximum sequence length.

    Returns:
        torch.Tensor: Tensor of shape (num_samples, max_len).
    """
    encoded = []
    for text in texts:
        tokens = tokenize(text)
        indices = [word2idx.get(t, UNK_IDX) for t in tokens[:max_len]]
        # Pad if shorter than max_len
        if len(indices) < max_len:
            indices += [PAD_IDX] * (max_len - len(indices))
        encoded.append(indices)

    return torch.tensor(encoded, dtype=torch.long)


class SentimentDataset(Dataset):
    """
    PyTorch Dataset for sentiment classification.
    """

    def __init__(self, encoded_texts, labels):
        """
        Args:
            encoded_texts (torch.Tensor): Encoded and padded text tensor.
            labels (torch.Tensor): Label tensor.
        """
        self.encoded_texts = encoded_texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.encoded_texts[idx], self.labels[idx]


def create_dataloaders(train_texts, train_labels, val_texts, val_labels,
                       word2idx, max_len=64, batch_size=64):
    """
    Create PyTorch DataLoaders for training and validation.

    Args:
        train_texts (list): List of cleaned training texts.
        train_labels (list): List of training labels.
        val_texts (list): List of cleaned validation texts.
        val_labels (list): List of validation labels.
        word2idx (dict): Vocabulary mapping.
        max_len (int): Maximum sequence length.
        batch_size (int): Batch size for DataLoaders.

    Returns:
        train_loader (DataLoader): Training DataLoader.
        val_loader (DataLoader): Validation DataLoader.
    """
    # Encode texts
    train_encoded = encode_texts(train_texts, word2idx, max_len)
    val_encoded = encode_texts(val_texts, word2idx, max_len)

    # Convert labels to tensors
    train_label_tensor = torch.tensor(train_labels, dtype=torch.long)
    val_label_tensor = torch.tensor(val_labels, dtype=torch.long)

    # Create datasets
    train_dataset = SentimentDataset(train_encoded, train_label_tensor)
    val_dataset = SentimentDataset(val_encoded, val_label_tensor)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False)

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    return train_loader, val_loader
