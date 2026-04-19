"""
Inference Module
================
Functions for running sentiment prediction on raw financial text
using trained RNN or BERT models.
"""

import torch
import torch.nn.functional as F

from src.data_loading import LABEL_MAP
from src.preprocessing import clean_tweet, tokenize, PAD_IDX, UNK_IDX


def predict_rnn(text, model, word2idx, max_len=64, device="cpu"):
    """
    Predict sentiment for a single text using a trained RNN model.

    Args:
        text (str): Raw financial text/tweet.
        model (nn.Module): Trained RNN model.
        word2idx (dict): Vocabulary mapping.
        max_len (int): Maximum sequence length.
        device (str): Device.

    Returns:
        predicted_label (str): Predicted sentiment label.
        probabilities (dict): Class probabilities.
    """
    model.eval()
    # Clean and tokenize
    cleaned = clean_tweet(text)
    tokens = tokenize(cleaned)
    indices = [word2idx.get(t, UNK_IDX) for t in tokens[:max_len]]

    # Pad
    if len(indices) < max_len:
        indices += [PAD_IDX] * (max_len - len(indices))

    # Convert to tensor
    input_tensor = torch.tensor([indices], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()

    predicted_id = int(probs.argmax())
    predicted_label = LABEL_MAP[predicted_id]

    probabilities = {LABEL_MAP[i]: float(probs[i]) for i in range(len(probs))}

    return predicted_label, probabilities


def predict_bert(text, model, tokenizer, max_len=128, device="cpu"):
    """
    Predict sentiment for a single text using a fine-tuned BERT model.

    Args:
        text (str): Raw financial text/tweet.
        model: Fine-tuned BERT model.
        tokenizer: Corresponding tokenizer.
        max_len (int): Max token length.
        device (str): Device.

    Returns:
        predicted_label (str): Predicted sentiment label.
        probabilities (dict): Class probabilities.
    """
    model.eval()
    model = model.to(device)

    encoding = tokenizer(text, truncation=True, padding=True,
                         max_length=max_len, return_tensors="pt")

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()

    predicted_id = int(probs.argmax())
    predicted_label = LABEL_MAP[predicted_id]

    probabilities = {LABEL_MAP[i]: float(probs[i]) for i in range(len(probs))}

    return predicted_label, probabilities


def predict_paragraph(text, model, word2idx=None, tokenizer=None,
                      model_type="rnn", max_len=64, device="cpu"):
    """
    Automated script that takes a raw financial paragraph and outputs sentiment.
    This is the final deliverable function.

    Args:
        text (str): Raw financial paragraph/tweet.
        model: Trained model (RNN or BERT).
        word2idx (dict): Vocabulary mapping (for RNN models).
        tokenizer: Tokenizer (for BERT models).
        model_type (str): 'rnn' or 'bert'.
        max_len (int): Maximum sequence length.
        device (str): Device.

    Returns:
        result (dict): Dictionary with predicted label, probabilities, and input text.
    """
    if model_type == "rnn":
        if word2idx is None:
            raise ValueError("word2idx is required for RNN models.")
        predicted_label, probabilities = predict_rnn(
            text, model, word2idx, max_len, device
        )
    elif model_type == "bert":
        if tokenizer is None:
            raise ValueError("tokenizer is required for BERT models.")
        predicted_label, probabilities = predict_bert(
            text, model, tokenizer, max_len, device
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    result = {
        "input_text": text,
        "predicted_sentiment": predicted_label,
        "probabilities": probabilities
    }

    print(f"\nInput:  {text}")
    print(f"Sentiment: {predicted_label}")
    print(f"Probabilities: {probabilities}")

    return result
