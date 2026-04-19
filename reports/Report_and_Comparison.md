# Financial News Sentiment Prediction — Report & Comparison

## 1. Project Overview

This project builds a sentiment classification system that labels finance-related tweets as **Bearish**, **Bullish**, or **Neutral** using deep learning models. We compare three RNN-based baselines against a fine-tuned BERT model on the [Twitter Financial News Sentiment](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) dataset from Hugging Face.

- **Training samples:** 9,543
- **Validation samples:** 2,388
- **Classes:** Bearish (0), Bullish (1), Neutral (2)
- **Class imbalance:** Neutral dominates at ~65.6% of all samples

---

## 2. RNN/LSTM/GRU Baseline Comparison

### Architecture Details

All three RNN-based models share the same architecture template:

| Component | Configuration |
|-----------|--------------|
| Embedding Dimension | 128 |
| Hidden Dimension | 128 |
| Number of Layers | 1 |
| Dropout | 0.3 |
| Learning Rate | 1e-3 |
| Early Stopping Patience | 3 epochs |
| Max Sequence Length | 64 |
| Vocabulary Size | 7,532 tokens |

The only difference is the recurrent cell type:
- **Simple RNN:** Basic recurrent unit (`nn.RNN`)
- **LSTM:** Long Short-Term Memory with forget, input, and output gates (`nn.LSTM`)
- **GRU:** Gated Recurrent Unit with reset and update gates (`nn.GRU`)

### Results

| Model | Parameters | Accuracy | Macro F1 | Bearish P/R | Bullish P/R | Neutral P/R |
|-------|-----------|----------|----------|-------------|-------------|-------------|
| Simple RNN | ~1.05M | 65.58% | 0.2636 | 0.00/0.00 | 0.00/0.00 | 0.66/1.00 |
| LSTM | ~1.12M | 65.58% | 0.2636 | 0.00/0.00 | 0.00/0.00 | 0.66/1.00 |
| GRU | ~1.10M | 65.58% | 0.2636 | 0.00/0.00 | 0.00/0.00 | 0.66/1.00 |

### Analysis: Which RNN/LSTM Variant Works Best and Why

All three RNN variants achieved identical performance (65.58% accuracy, 0.264 Macro F1), with all models collapsing to **majority-class prediction** — predicting "Neutral" for every input. This is a well-documented failure mode when:

1. **Severe class imbalance exists:** The Neutral class comprises ~65.6% of the dataset. The models learn that always predicting Neutral minimizes cross-entropy loss quickly, and early stopping triggers before the model escapes this local minimum.

2. **Randomly initialized embeddings lack semantic knowledge:** Unlike pre-trained models, these RNN models start with random word embeddings. Financial tweets are short (avg ~15 words) and contain domain-specific language ($tickers, market jargon) that random embeddings cannot represent meaningfully within a few training epochs.

3. **Simple architecture limitations:** With only 1 layer and 128-dimensional hidden states, the models have limited capacity to capture the subtle contextual cues that differentiate bearish from bullish sentiment (e.g., "surges" vs. "plummets").

**In theory**, LSTM and GRU should outperform Simple RNN because:
- **LSTM** uses forget/input/output gates to selectively remember and forget information, addressing the vanishing gradient problem
- **GRU** achieves similar gating with fewer parameters (reset + update gates), often training faster

However, in this dataset, the class imbalance overwhelms any architectural advantage. To make RNN models competitive, one would need: class-weighted loss, oversampling minority classes, pre-trained word embeddings (GloVe/Word2Vec), or a larger model with more training epochs.

---

## 3. BERT Performance Improvement

### Architecture

| Component | Configuration |
|-----------|--------------|
| Base Model | `ahmedrachid/FinancialBERT-Sentiment-Analysis` |
| Parameters | 109,754,115 (~110M) |
| Fine-tuning Epochs | 2 |
| Learning Rate | 2e-5 |
| Batch Size | 32 |
| Max Sequence Length | 128 |

### Results

| Model | Accuracy | Macro F1 | Bearish P/R | Bullish P/R | Neutral P/R |
|-------|----------|----------|-------------|-------------|-------------|
| **BERT (FinBERT)** | **82.83%** | **0.7598** | 0.80/0.61 | 0.78/0.64 | 0.84/0.93 |

### How Much BERT Improves Performance

| Metric | Best RNN | BERT | Improvement |
|--------|----------|------|-------------|
| Accuracy | 65.58% | 82.83% | +17.25 pp |
| Macro F1 | 0.2636 | 0.7598 | **+187.8%** |
| Bearish Recall | 0.00 | 0.61 | ∞ (from zero) |
| Bullish Recall | 0.00 | 0.64 | ∞ (from zero) |

BERT achieves a **+187.8% improvement in Macro F1** over the best RNN model. The key reasons:

1. **Pre-trained contextual embeddings:** FinBERT was pre-trained on financial text corpora, so it already understands financial terminology, ticker symbols, and sentiment-bearing phrases before fine-tuning.

2. **Transfer learning:** Instead of learning from scratch with ~9.5K samples, BERT leverages knowledge from millions of financial documents. Fine-tuning only adapts the final classification layers.

3. **Attention mechanism:** The Transformer's self-attention allows BERT to weigh important tokens (e.g., "surges", "plummets", "crash") regardless of their position, unlike RNNs which must propagate information sequentially.

4. **Subword tokenization:** BERT's WordPiece tokenizer handles out-of-vocabulary words gracefully by breaking them into subwords, unlike the word-level vocabulary used by RNN models.

---

## 4. Streamlit Dashboard

### Description

The Streamlit dashboard (`app/streamlit_app.py`) provides an interactive web interface for real-time sentiment prediction.

### Features

1. **Model Selection (Sidebar):**
   - Users can switch between the RNN baseline and BERT model via a dropdown
   - Model info panel shows architecture type, parameter count, vocab size, and max sequence length

2. **Text Input (Left Panel):**
   - Free-text input area where users type any financial tweet or headline
   - Three quick-test sample buttons for instant demonstration:
     - Bullish example: "$AAPL Apple stock surges to all-time high..."
     - Bearish example: "$TSLA Tesla shares plummet 15%..."
     - Neutral example: "The Federal Reserve will maintain current interest rates"

3. **Prediction Results (Right Panel):**
   - Predicted sentiment label with color-coded emoji (🔴 Bearish, 🟢 Bullish, 🔵 Neutral)
   - Confidence percentage for the predicted class
   - Three metric cards showing individual class probabilities
   - Interactive Plotly bar chart visualizing all class probabilities

4. **Validation Distribution (Bottom):**
   - Bar chart showing the overall class distribution from the validation dataset
   - Helps users understand the data composition and class imbalance

### How to Run

```bash
cd "Financial news sentiment prediction using DL and BERT"
streamlit run app/streamlit_app.py
```

The app opens at `http://localhost:8501`.

### App Layout Description

```
┌──────────────────────────────────────────────────────────────┐
│  📈 Financial News Sentiment Predictor                       │
├──────────────┬───────────────────────────────────────────────┤
│  Sidebar:    │  Left Column:        │  Right Column:         │
│  - Model     │  - Text input box    │  - 🟢 Bullish         │
│    selector  │  - [Predict] button  │  - Confidence: 96.2%   │
│  - Model     │  - Quick test        │  - Bearish: 1.2%       │
│    info      │    samples           │  - Bullish: 96.2%      │
│              │                      │  - Neutral: 2.5%       │
│              │                      │  - [Bar Chart]         │
├──────────────┴───────────────────────┴───────────────────────┤
│  Validation Set Class Distribution                           │
│  [Bar chart: Bearish ~500 | Bullish ~350 | Neutral ~1540]    │
└──────────────────────────────────────────────────────────────┘
```

### Key Observations from the Dashboard

- **BERT model** correctly identifies sentiment with high confidence (>90% for clear cases)
- **RNN model** predicts "Neutral" for all inputs with ~63.5% probability — consistent with majority-class collapse
- Switching between models demonstrates the dramatic quality difference in real-time

---

## 5. Conclusion

| Aspect | RNN Baselines | BERT (FinBERT) |
|--------|--------------|----------------|
| Training Time | ~2 min (all three) | ~3 hrs (CPU, 2 epochs) |
| Parameters | ~1M | ~110M |
| Accuracy | 65.6% | 82.8% |
| Macro F1 | 0.264 | 0.760 |
| Handles Imbalance | No (majority-class collapse) | Yes (learns all classes) |
| Real-world Usability | Not usable | Production-ready |

**BERT/FinBERT is the clear winner**, achieving 82.8% accuracy and 0.76 Macro F1 with meaningful predictions across all three sentiment classes. The RNN baselines serve as a useful demonstration of why pre-trained language models have become the standard for NLP tasks — especially on small, imbalanced datasets where learning from scratch is insufficient.
