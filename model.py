import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
# import altair as alt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
import collections
import re
import string
import requests
import datetime
from bs4 import BeautifulSoup

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from torch.nn.functional import softmax
import torch

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split


df = pd.read_csv('survey_data.csv')
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y', errors='coerce')

# Text Preprocessing of the texts column using NLTK
def preprocess_text(text):
    """
    Preprocess a text string for sentiment analysis.

    Parameters
    ----------
    text : str
        The text string to preprocess.

    Returns
    -------
    str
        The preprocessed text string.
    """

    # Define the denoise_text function
    def denoise_text(text):
        text = strip_html(text)
        return text

    # Define the strip_html function
    def strip_html(text):
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()

    # Apply denoising functions
    text = denoise_text(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs, hashtags, mentions, and special characters
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
    text = re.sub(r"[^\w\s]", "", text)

    # Remove numbers/digits
    text = re.sub(r'\b[0-9]+\b\s*', '', text)

    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Join tokens back into a single string
    return ' '.join(tokens)
    
X_preprocessed = [preprocess_text(text) for text in df['feedback']]

# model name
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# calculate sentiment scoring
def sentiment_score(text, model, tokenizer, label_mapping={1: 'Negative', 2: 'Neutral', 3: 'Positive'}):
    try:
        # Tokenize the input text
        tokens = tokenizer.encode(text, return_tensors='pt')

        # Get model predictions
        with torch.no_grad():
            result = model(tokens)

        # Obtain predicted class index
        predicted_index = torch.argmax(result.logits).item()

        # Map scores to labels
        if label_mapping is not None:
            predicted_label = label_mapping.get(predicted_index + 1, f'Class {predicted_index + 1}')

        # Calculate confidence percentage
        probabilities = softmax(result.logits, dim=1)
        confidence_percentage = str(probabilities[0, predicted_index].item() * 100) + '%'

        # Return results
        return {
            'predicted_label': predicted_label,
            'predicted_index': predicted_index + 1,
            'confidence_percentage': confidence_percentage
        }

    except Exception as e:
        return {
            'error': str(e)
        }


# Function to apply sentiment scoring to a single feedback
def apply_sentiment_scoring(feedback):
    # Apply sentiment scoring
    result = sentiment_score(feedback, model, tokenizer, label_mapping={1: 'Negative', 2: 'Neutral', 3: 'Positive'})

    # Return the sentiment scoring results
    # return {
    #     'sentiments': result.get('predicted_label', None),
    #     'sentiments_index': result.get('predicted_index', None),
    #     'percentage_confidence': result.get('confidence_percentage', None)
    # }

    return pd.Series({
        'sentiments': result.get('predicted_label', None),
        'sentiments_index': result.get('predicted_index', None),
        'percentage_confidence': result.get('confidence_percentage', None)
    })
