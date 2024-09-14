import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocess text function
def preprocess_text(text):
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()

    # Tokenization
    words = text.split()

    # Remove stop words and lemmatize
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    return ' '.join(words)

# Load and preprocess dataset
def load_and_preprocess_data(data_path):
    # Load data
    data = pd.read_csv(data_path)

    # Combine 'Summary' and 'Review' into a single text column for analysis
    data['text'] = data['Summary'] + " " + data['Review']
    data['cleaned_text'] = data['text'].apply(preprocess_text)

    return data

# Train and evaluate models
def train_and_evaluate_models(data):
    # Convert text to numerical data using TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(data['cleaned_text']).toarray()
    y = data['Sentiment']  # Using the 'Sentiment' column

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize models
    models = {
        "Naive Bayes": MultinomialNB(),
        "Support Vector Machine (SVM)": SVC(kernel='linear')
    }
    
    # Store accuracy results
    accuracy_results = {}

    # Train and evaluate each model
    for model_name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_results[model_name] = accuracy

        print(f"\nModel: {model_name}")
        print(f"Accuracy: {accuracy}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
    
    return accuracy_results

# Plot model comparison
def plot_model_comparison(accuracy_results):
    models = list(accuracy_results.keys())
    accuracies = list(accuracy_results.values())

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.barh(models, accuracies, color='skyblue')
    plt.xlabel('Accuracy')
    plt.title('Model Comparison')
    plt.xlim(0, 1)  # Accuracy is between 0 and 1
    plt.show()

# Main function
if __name__ == "__main__":
    # Path to your CSV file containing the dataset
    data_path = 'Dataset-SA.csv'  # Ensure this file is in the same directory
    
    # Load and preprocess data
    data = load_and_preprocess_data(data_path)
    
    # Train and evaluate models
    accuracy_results = train_and_evaluate_models(data)
    
    # Plot model comparison
    plot_model_comparison(accuracy_results)
