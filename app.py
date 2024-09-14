import os
import subprocess
import sys

# Function to install packages
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install nltk if it's not already installed
try:
    import nltk
except ImportError:
    install("nltk")
    import nltk

import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = text.split()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

import streamlit as st

st.title('Sentiment Analysis App')

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data['text'] = data['Summary'] + " " + data['Review']
    data['cleaned_text'] = data['text'].apply(preprocess_text)
    
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(data['cleaned_text']).toarray()
    y = data['Sentiment']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {
        "Naive Bayes": MultinomialNB(),
        "Support Vector Machine (SVM)": SVC(kernel='linear')
    }
    
    accuracy_results = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_results[model_name] = accuracy

        st.write(f"\nModel: {model_name}")
        st.write(f"Accuracy: {accuracy}")
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f"Confusion Matrix - {model_name}")
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)

    st.write("Model Comparison")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(list(accuracy_results.keys()), list(accuracy_results.values()), color='skyblue')
    ax.set_xlabel('Accuracy')
    ax.set_title('Model Comparison')
    ax.set_xlim(0, 1)
    st.pyplot(fig)
