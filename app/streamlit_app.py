"""
Streamlit Dashboard
===================
Interactive sentiment prediction dashboard for financial news/tweets.

Features:
- Text input box for user to type a tweet/headline
- Predicted sentiment with probabilities
- Bar chart of class probabilities
- Overall class distribution from validation data

Run from project root: streamlit run app/streamlit_app.py
"""

import sys
import os
import json

import streamlit as st
import torch
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loading import (LABEL_MAP, LABEL_TO_ID, NUM_CLASSES,
                              load_financial_sentiment_data)
from src.preprocessing import clean_tweet, tokenize, PAD_IDX, UNK_IDX
from src.models import get_rnn_model, load_bert_model
from src.inference import predict_rnn, predict_bert

# --- Page Configuration ---
st.set_page_config(
    page_title="Financial Sentiment Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Financial News Sentiment Predictor")
st.markdown(
    "Predict the sentiment of financial tweets and headlines as "
    "**Bearish**, **Bullish**, or **Neutral** using deep learning models."
)

# --- Sidebar: Model Selection ---
st.sidebar.header("Model Configuration")

# Models saved by notebook are inside notebooks/models/saved_models/
MODEL_DIR = os.path.join(PROJECT_ROOT, 'notebooks', 'models', 'saved_models')


# Check available RNN models
rnn_model_files = {
    "Simple RNN": "simple_rnn_model.pt",
    "LSTM": "lstm_model.pt",
    "GRU": "gru_model.pt",
    "RNN (Best Baseline)": "best_rnn_model.pt"
}
available_models = []
for display_name, fname in rnn_model_files.items():
    if os.path.exists(os.path.join(MODEL_DIR, fname)):
        available_models.append(display_name)
bert_available = os.path.exists(os.path.join(MODEL_DIR, 'bert_finetuned'))
if bert_available:
    available_models.append("BERT (FinBERT)")

if not available_models:
    st.error(
        "No trained models found! Please run the Jupyter notebook first "
        "to train and save models."
    )
    st.stop()


selected_model = st.sidebar.selectbox("Select Model", available_models)



@st.cache_resource
def load_rnn_model_cached(model_file):
    """Load the selected RNN model and vocabulary."""
    checkpoint = torch.load(
        os.path.join(MODEL_DIR, model_file),
        map_location='cpu',
        weights_only=False
    )
    with open(os.path.join(MODEL_DIR, 'word2idx.json'), 'r') as f:
        word2idx = json.load(f)

    model = get_rnn_model(
        model_type=checkpoint['model_type'],
        vocab_size=checkpoint['vocab_size'],
        embed_dim=checkpoint['embed_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        num_classes=checkpoint['num_classes']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, word2idx, checkpoint['max_len']


@st.cache_resource
def load_bert_model_cached():
    """Load the fine-tuned BERT model."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    model_path = os.path.join(MODEL_DIR, 'bert_finetuned')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.config.id2label = LABEL_MAP
    model.config.label2id = LABEL_TO_ID
    model.eval()
    return model, tokenizer


@st.cache_data
def load_validation_distribution():
    """Load validation set class distribution for the distribution chart."""
    local_valid_path = os.path.join(PROJECT_ROOT, 'sent_valid.csv')
    if os.path.exists(local_valid_path):
        val_df = pd.read_csv(local_valid_path)
    else:
        _, val_df = load_financial_sentiment_data()

    counts = val_df['label'].value_counts().sort_index()
    dist_df = pd.DataFrame({
        'Sentiment': [LABEL_MAP[i] for i in counts.index],
        'Count': counts.values
    })
    return dist_df




# --- Load Selected Model ---
with st.spinner("Loading model..."):
    checkpoint = None
    if selected_model in rnn_model_files:
        # Load RNN model and also get checkpoint for sidebar info
        checkpoint_path = os.path.join(MODEL_DIR, rnn_model_files[selected_model])
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        with open(os.path.join(MODEL_DIR, 'word2idx.json'), 'r') as f:
            word2idx = json.load(f)
        model = get_rnn_model(
            model_type=checkpoint['model_type'],
            vocab_size=checkpoint['vocab_size'],
            embed_dim=checkpoint['embed_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            num_classes=checkpoint['num_classes']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        max_len = checkpoint['max_len']
        model_type = "rnn"
    else:
        model, tokenizer = load_bert_model_cached()
        model_type = "bert"
        max_len = 128
        checkpoint = None

st.sidebar.success(f"Model loaded: {selected_model}")

# Sidebar model info
st.sidebar.markdown("---")
st.sidebar.subheader("Model Info")
if model_type == "rnn":
    total_params = sum(p.numel() for p in model.parameters())
    if selected_model != 'RNN (Best Baseline)':
        arch_name = selected_model
    else:
        arch_name = checkpoint['model_type'].replace('_', ' ').title() if checkpoint else 'RNN'
    st.sidebar.markdown(f"- **Architecture:** Embedding + {arch_name}")
    st.sidebar.markdown(f"- **Parameters:** {total_params:,}")
    st.sidebar.markdown(f"- **Max Sequence Length:** {max_len}")
    st.sidebar.markdown(f"- **Vocab Size:** {len(word2idx):,}")
else:
    total_params = sum(p.numel() for p in model.parameters())
    st.sidebar.markdown(f"- **Architecture:** FinBERT (Transformer)")
    st.sidebar.markdown(f"- **Parameters:** {total_params:,}")
    st.sidebar.markdown(f"- **Max Sequence Length:** {max_len}")
    st.sidebar.markdown(f"- **Pre-trained on:** Financial text")

# --- Main Content ---
col1, col2 = st.columns([1, 1])

# Initialise session state for text input
if "text_area_input" not in st.session_state:
    st.session_state["text_area_input"] = ""

with col1:
    st.subheader("Enter Financial Text")

    # Sample texts for quick testing (rendered before text_area so clicks
    # update the default value before the widget is drawn this run)
    st.markdown("**Quick Test Samples:**")
    sample_texts = [
        "$AAPL Apple stock surges to all-time high after strong earnings beat",
        "$TSLA Tesla shares plummet 15% amid production delays",
        "The Federal Reserve will maintain current interest rates"
    ]
    for index, sample in enumerate(sample_texts):
        if st.button(sample, key=f"sample_{index}"):
            st.session_state["text_area_input"] = sample

    user_input = st.text_area(
        "Type a financial tweet or headline:",
        placeholder="e.g., $AAPL Apple stock surges after strong earnings report",
        height=120,
        key="text_area_input"
    )

    predict_btn = st.button("Predict Sentiment", type="primary", use_container_width=True)

with col2:
    if predict_btn and user_input.strip():
        st.subheader("Prediction Results")

        # Run prediction
        if model_type == "rnn":
            predicted_label, probabilities = predict_rnn(
                user_input, model, word2idx, max_len
            )
        else:
            predicted_label, probabilities = predict_bert(
                user_input, model, tokenizer, max_len
            )

        # Sentiment color mapping
        color_map = {"Bearish": "🔴", "Bullish": "🟢", "Neutral": "🔵"}

        # Display prediction
        st.markdown(
            f"### {color_map[predicted_label]} Predicted Sentiment: "
            f"**{predicted_label}**"
        )
        st.markdown(
            f"**Confidence:** {probabilities[predicted_label]*100:.1f}%"
        )

        # Probability detail metrics
        pcol1, pcol2, pcol3 = st.columns(3)
        pcol1.metric("Bearish", f"{probabilities['Bearish']*100:.1f}%")
        pcol2.metric("Bullish", f"{probabilities['Bullish']*100:.1f}%")
        pcol3.metric("Neutral", f"{probabilities['Neutral']*100:.1f}%")

        # Bar chart of class probabilities
        prob_df = pd.DataFrame({
            'Sentiment': list(probabilities.keys()),
            'Probability': list(probabilities.values())
        })

        fig = px.bar(
            prob_df, x='Sentiment', y='Probability',
            color='Sentiment',
            color_discrete_map={
                'Bearish': '#e74c3c', 'Bullish': '#2ecc71', 'Neutral': '#3498db'
            },
            title='Class Probabilities',
            text=prob_df['Probability'].apply(lambda x: f'{x:.3f}')
        )
        fig.update_layout(
            yaxis_range=[0, 1],
            showlegend=False,
            height=350
        )
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    elif predict_btn:
        st.warning("Please enter some text to analyze.")

# --- Validation Distribution Section ---
st.markdown("---")
st.subheader("Validation Set Class Distribution")

try:
    dist_df = load_validation_distribution()
    fig = px.bar(
        dist_df, x='Sentiment', y='Count',
        color='Sentiment',
        color_discrete_map={
            'Bearish': '#e74c3c', 'Bullish': '#2ecc71', 'Neutral': '#3498db'
        },
        title='Overall Class Distribution (Validation Data)',
        text='Count'
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.info(f"Could not load validation distribution: {e}")

# --- Footer ---
st.markdown("---")
st.markdown(
    "*Built with PyTorch, Hugging Face Transformers, and Streamlit | "
    "Financial News Sentiment Prediction using Deep Learning & BERT*"
)
