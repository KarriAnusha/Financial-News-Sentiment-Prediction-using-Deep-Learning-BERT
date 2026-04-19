# Financial News Sentiment Prediction using Deep Learning & BERT

A sentiment classification system that labels finance-related tweets as **Bearish**, **Bullish**, or **Neutral** using deep learning and transformer models.

## Project Overview

| Item | Detail |
|------|--------|
| **Domain** | Finance & FinTech |
| **Task** | Multi-class sentiment classification (3 classes) |
| **Dataset** | [Twitter Financial News Sentiment](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) (11,932 tweets) |
| **Models** | Simple RNN, LSTM, GRU, BERT (FinBERT) |

## Labels

| Label ID | Name | Description |
|----------|------|-------------|
| 0 | Bearish | Negative / pessimistic sentiment |
| 1 | Bullish | Positive / optimistic sentiment |
| 2 | Neutral | Objective / mixed sentiment |

## Project Structure

```
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
├── notebooks/
│   └── Financial_News_Sentiment_Prediction.ipynb  # Main notebook (all steps)
├── src/
│   ├── __init__.py
│   ├── data_loading.py        # Dataset loading and statistics
│   ├── preprocessing.py       # Text cleaning, tokenization, vocabulary
│   ├── models.py              # RNN/LSTM/GRU/BERT model architectures
│   ├── training.py            # Training loops with early stopping
│   ├── evaluation.py          # Metrics, confusion matrix, plots
│   └── inference.py           # Prediction on raw text
├── app/
│   └── streamlit_app.py       # Interactive Streamlit dashboard
├── models/
│   └── saved_models/          # Saved model weights (created after training)
└── reports/
    └── comparison_report.pdf  # Generated comparison report
```

## Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd "Financial news sentiment prediction using DL and BERT"
```

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## How to Run

### Step 1: Run the Jupyter Notebook

The notebook executes the complete pipeline in order:

```bash
cd notebooks
jupyter notebook Financial_News_Sentiment_Prediction.ipynb
```

Run all cells sequentially. The notebook will:
1. Load and explore the dataset from Hugging Face
2. Preprocess tweets (clean text, build vocabulary, create tensors)
3. Train 3 baseline models (Simple RNN, LSTM, GRU) with early stopping
4. Fine-tune a BERT/FinBERT model (optional section)
5. Evaluate all models (accuracy, macro F1, confusion matrices)
6. Save trained models to `models/saved_models/`
7. Generate a 2-page PDF comparison report in `reports/`
8. Run automated inference on sample financial texts

### Step 2: Launch the Streamlit Dashboard

After training (models must be saved):

```bash
streamlit run app/streamlit_app.py
```

The dashboard provides:
- **Text input box**: Type any financial tweet/headline
- **Sentiment prediction**: Shows predicted label and confidence
- **Probability bar chart**: Visual breakdown of class probabilities
- **Class distribution chart**: Overall validation data distribution

## Dependencies

- Python 3.8+
- PyTorch >= 2.0.0
- Hugging Face Datasets >= 2.14.0
- Hugging Face Transformers >= 4.30.0
- scikit-learn >= 1.3.0
- Streamlit >= 1.28.0
- matplotlib, seaborn, plotly
- fpdf2 (for report generation)

## Evaluation Metrics

- **Overall Accuracy** on validation data
- **Macro F1-Score** across three classes (treats each class equally)
- **Class-wise Precision and Recall** for Bearish, Bullish, and Neutral
- **Confusion Matrices** for each model
- **Training curves** (loss and accuracy per epoch)

## Technical Tags

Python, PyTorch, Hugging Face Datasets, Hugging Face Transformers, BERT, FinBERT, Text Classification, Sentiment Analysis, Finance NLP, RNN, LSTM, GRU, Embedding Layers, Streamlit

## ⚠️ Model Files Not Included

Due to GitHub's file size limits, large model files (such as `.safetensors` for BERT/FinBERT) are **not included** in this repository.

**For local evaluation or live demo:**

- Download the required model files from your own backup, cloud storage, or retrain using the notebook.
- Place the files in the appropriate folder, e.g.:
  - `notebooks/models/saved_models/bert_finetuned/model.safetensors`
  - `notebooks/models/saved_models/bert_finetuned_new/model.safetensors`
- The app and notebook will load these files if present. If missing, you will see an error or a message to download the model.

**Tip:** You can share model files via Google Drive, Dropbox, or Hugging Face Hub and provide the download link here for collaborators.

Example download instruction:
```
Download the fine-tuned BERT model from [YOUR_LINK_HERE] and place it in `notebooks/models/saved_models/bert_finetuned_new/`.
```
