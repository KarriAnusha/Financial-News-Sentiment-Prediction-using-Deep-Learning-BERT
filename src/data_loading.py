"""
Data Loading Module
===================
Functions for loading the Twitter Financial News Sentiment dataset
from Hugging Face and inspecting basic statistics.
"""

from datasets import load_dataset
import pandas as pd


# Label mapping as specified in the dataset
LABEL_MAP = {0: "Bearish", 1: "Bullish", 2: "Neutral"}
LABEL_TO_ID = {"Bearish": 0, "Bullish": 1, "Neutral": 2}
NUM_CLASSES = 3


def load_financial_sentiment_data():
    """
    Load the Twitter Financial News Sentiment dataset from Hugging Face.
    Uses the 'sentiment' subset with official train/validation splits.

    Returns:
        train_df (pd.DataFrame): Training split with columns ['text', 'label'].
        val_df (pd.DataFrame): Validation split with columns ['text', 'label'].
    """
    dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")

    train_df = pd.DataFrame(dataset["train"])
    val_df = pd.DataFrame(dataset["validation"])

    return train_df, val_df


def get_class_distribution(df, split_name="Dataset"):
    """
    Print and return class distribution for a given dataframe.

    Args:
        df (pd.DataFrame): DataFrame with a 'label' column.
        split_name (str): Name of the split for display purposes.

    Returns:
        pd.DataFrame: Distribution counts and percentages.
    """
    counts = df["label"].value_counts().sort_index()
    percentages = df["label"].value_counts(normalize=True).sort_index() * 100

    dist_df = pd.DataFrame({
        "Label_ID": counts.index,
        "Label_Name": [LABEL_MAP[i] for i in counts.index],
        "Count": counts.values,
        "Percentage": percentages.values.round(2)
    })

    print(f"\n--- {split_name} Class Distribution ---")
    print(f"Total samples: {len(df)}")
    print(dist_df.to_string(index=False))

    return dist_df


def get_basic_stats(df, split_name="Dataset"):
    """
    Print basic text statistics for the dataset.

    Args:
        df (pd.DataFrame): DataFrame with a 'text' column.
        split_name (str): Name of the split for display purposes.
    """
    text_lengths = df["text"].str.len()
    word_counts = df["text"].str.split().str.len()

    print(f"\n--- {split_name} Basic Stats ---")
    print(f"Number of samples: {len(df)}")
    print(f"Avg character length: {text_lengths.mean():.1f}")
    print(f"Avg word count: {word_counts.mean():.1f}")
    print(f"Max word count: {word_counts.max()}")
    print(f"Min word count: {word_counts.min()}")
