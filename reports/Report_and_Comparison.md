# Financial News Sentiment Prediction — Report & Comparison

## 1. Project Overview

This project builds a sentiment classification system that labels finance-related tweets as **Bearish**, **Bullish**, or **Neutral** using deep learning models. We compare three RNN-based baselines against a fine-tuned BERT model on the [Twitter Financial News Sentiment](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) dataset from Hugging Face.

- **Training samples used:** 9,543
- **Validation samples used:** 2,388
- **Classes:** Bearish (0), Bullish (1), Neutral (2)
- **Class imbalance:** Neutral dominates at ~65.6% of all samples

**Dataset count note:** The project uses the official Hugging Face
`zeroshot/twitter-financial-news-sentiment` train/validation splits returned by
`datasets.load_dataset`. The proposal/dataset-card text lists 9,938 training and
2,486 validation instances, but the current Hugging Face dataset viewer and
local CSVs expose 9,543 and 2,388 rows respectively. The code now warns when the
loaded split sizes differ from the proposal counts.

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
| Simple RNN | ~1.05M | 72.03% | 0.6322 | 0.45/0.59 | 0.56/0.56 | 0.86/0.80 |
| LSTM | ~1.12M | 78.27% | 0.7093 | 0.55/0.63 | 0.66/0.69 | 0.89/0.84 |
| GRU | ~1.10M | 79.10% | 0.7117 | 0.57/0.60 | 0.67/0.68 | 0.88/0.87 |

### Analysis: Which RNN/LSTM Variant Works Best and Why

The initial review found that the recurrent models were reading the final padded
timestep instead of the last real token. After correcting that padding handling
and retraining, the baselines no longer collapse to Neutral. GRU performs best
with 79.10% accuracy and 0.7117 macro F1, narrowly ahead of LSTM.

GRU and LSTM outperform Simple RNN because their gates preserve useful context
from short financial tweets more reliably. The remaining gap versus BERT is
expected because these baselines still rely on randomly initialized embeddings
and a small word-level vocabulary, while BERT starts with pre-trained financial
language knowledge and subword tokenization.


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

The saved BERT config now uses the project label order:
`0=Bearish`, `1=Bullish`, `2=Neutral`. This prevents Hugging Face config-based
inference from confusing Bullish and Neutral labels.

### Results

| Model | Accuracy | Macro F1 | Bearish P/R | Bullish P/R | Neutral P/R |
|-------|----------|----------|-------------|-------------|-------------|
| **BERT (FinBERT)** | **82.83%** | **0.7598** | 0.80/0.61 | 0.78/0.64 | 0.84/0.93 |

### How Much BERT Improves Performance

| Metric | Best RNN | BERT | Improvement |
|--------|----------|------|-------------|
| Accuracy | 79.10% | 82.83% | +3.73 pp |
| Macro F1 | 0.7117 | 0.7598 | **+6.76%** |
| Bearish Recall | 0.60 | 0.61 | +0.01 |
| Bullish Recall | 0.68 | 0.64 | -0.04 |

BERT achieves a **+6.76% improvement in Macro F1** over the best corrected RNN model. The key reasons:

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
- **Corrected RNN models** now produce class-specific predictions instead of collapsing to Neutral
- Switching between models demonstrates BERT's smaller but still meaningful quality advantage in real-time

---

## 5. Conclusion

| Aspect | RNN Baselines | BERT (FinBERT) |
|--------|--------------|----------------|
| Training Time | ~2 min (all three) | ~3 hrs (CPU, 2 epochs) |
| Parameters | ~1M | ~110M |
| Accuracy | 79.1% | 82.8% |
| Macro F1 | 0.712 | 0.760 |
| Handles Imbalance | Partially, with class-weighted loss | Yes (learns all classes) |
| Real-world Usability | Prototype baseline | Stronger prototype |

**BERT/FinBERT remains the best model**, achieving 82.8% accuracy and 0.76 Macro F1. The corrected GRU baseline is now competitive at 79.1% accuracy and 0.712 Macro F1, but BERT still benefits from financial pre-training, attention, and subword tokenization.



